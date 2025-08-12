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
    T5PreTrainedModel,
    # T5ForConditionalGeneration,
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
    SequenceClassifierOutput
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




@add_start_docstrings(
    """T5 Model with non-linear mapping layer on top. Use soft nearest neighbor loss for contrastive learning.
    """, 
    T5_START_DOCSTRING
)
class T5ForDDIClassification(T5PreTrainedModel):
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

        # decoder_config = copy.deepcopy(config)
        # decoder_config.is_decoder = True
        # decoder_config.is_encoder_decoder = False
        # decoder_config.num_layers = config.num_decoder_layers
        # self.decoder = T5Stack(decoder_config, self.shared)

        # self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

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
        # self.decoder.parallelize(self.device_map)
        self.classifier = self.classifier.to(self.encoder.first_device)
        # self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        # self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        # self.decoder = self.decoder.to("cpu")
        self.classifier = self.classifier.to("cpu")
        # self.lm_head = self.lm_head.to("cpu")
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
        labels: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], SequenceClassifierOutput]:
        r"""
        Return:
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


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

        loss = None
        ########## Use Encoder Reps to predict DDI #############
        logits = self.classifier(hidden_states, drug1_mask, drug2_mask)
        if labels is not None:
            pred_criterion = nn.BCEWithLogitsLoss()
            loss = pred_criterion(logits, labels.squeeze())
        ########################################################


        if not return_dict:
            output = (logits, ) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )
    


