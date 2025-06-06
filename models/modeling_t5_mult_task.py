import copy
from typing import Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import gc

from torch.utils import checkpoint
import warnings
from dataclasses import dataclass
from transformers.models.t5.modeling_t5 import (
    # T5PreTrainedModel,
    T5ForConditionalGeneration,
    T5Config,
    T5Stack,
    T5_START_DOCSTRING,
    # T5_ENCODER_INPUTS_DOCSTRING,
    T5_INPUTS_DOCSTRING,
    PARALLELIZE_DOCSTRING,
    DEPARALLELIZE_DOCSTRING,
    __HEAD_MASK_WARNING_MSG,
    _CONFIG_FOR_DOC
)
from transformers.modeling_outputs import (
    Seq2SeqLMOutput, 
    BaseModelOutput
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.utils.logging import get_logger
logger = get_logger("transformers")


class InteractionModule(nn.Module):
    def __init__(self, hid_dim):
        super(InteractionModule, self).__init__()
        # 初始化权重矩阵
        self.linear = nn.Linear(hid_dim, hid_dim)
    
    def forward(self, inputs, mask1=None, mask2=None):
        # inputs: [batch_size, seq_len, hid_dim]
        
        transformed_inputs = self.linear(inputs)  # [batch_size, seq_len, hid_dim]
        
        # 计算input1和变换后的input2的点积
        # 需要调整矩阵乘法来处理三维张量
        interaction_matrix = torch.einsum('bik,bjk->bij', inputs, transformed_inputs)  # [batch_size, seq_len, seq_len]

        # 使用掩码调整交互矩阵，忽略填充部分
        if mask1 is not None and mask2 is not None:
            mask1 = mask1.unsqueeze(2)  # [batch_size, seq_len, 1]
            mask2 = mask2.unsqueeze(1)  # [batch_size, 1, seq_len]
            mask = mask1 * mask2  # [batch_size, seq_len, seq_len]，仅当两者都有效时结果为1
            interaction_matrix *= mask  # 将填充部分的交互设为0
        
        # 将所有交互值求和
        output = interaction_matrix.mean(dim=(1, 2))  # 对seq_len1和seq_len2维度求和，结果为[batch_size]
        
        return output

@dataclass
class MySeq2SeqLMOutput(Seq2SeqLMOutput):
    pred_logits: torch.FloatTensor = None


@add_start_docstrings(
    """T5 Model with non-linear mapping layer on top. Use soft nearest neighbor loss for contrastive learning.
    """, 
    T5_START_DOCSTRING
)
class T5ForConditionalGenerationMultTask(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        encoder_config.use_cache = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        classifier_dropout = (
            config.classifier_dropout if hasattr(config, 'classifier_dropout') else config.dropout_rate
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.classifier = InteractionModule(config.d_model)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.classifier = self.classifier.to(self.encoder.first_device)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.classifier = self.classifier.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=MySeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        drug1_mask: Optional[torch.FloatTensor] = None,
        drug2_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        cls_labels: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        Return:
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask


        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0] # last hidden state; bs x seq_len x hs

        ########## Use Encoder Reps to predict DDI #############
        pred_output = self.classifier(hidden_states, drug1_mask, drug2_mask)
        pred_loss = None
        if cls_labels is not None:
            pred_criterion = nn.BCEWithLogitsLoss()
            pred_loss = pred_criterion(pred_output, cls_labels)
        ########################################################


        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)


        decoder_loss = None
        if labels is not None:
            # decoder loss
            decoder_loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            decoder_loss = decoder_loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        
        if decoder_loss is not None and pred_loss is not None:
            loss = 0.5 * decoder_loss + 0.5 * pred_loss
        else:
            loss = None

        if not return_dict:
            output = (lm_logits, pred_output) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        
        return MySeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            pred_logits = pred_output,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    


