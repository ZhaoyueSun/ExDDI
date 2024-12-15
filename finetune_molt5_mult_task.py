#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Jointly generate the prediction and explanation.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import time
import random
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
import evaluate

import numpy as np
import datasets
from datasets import load_dataset
datasets.disable_caching()
datasets.disable_progress_bar()

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process,PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback
from tokenizers.models import BPE
from models.modeling_t5_mult_task import T5ForConditionalGenerationMultTask


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    from_checkpoint: bool = field(
        default=False, metadata={"help": "Whether load from checkpoint to continue learning"}
    )


@dataclass
class DataTrainingArguments:

    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    input_format: str = field(
        default="smiles",
        metadata={
            "help": "whether to input SMILES, description or both. options: smiles"
        },
    )
    interrupt_ratio: float = field(
        default=0.0,
        metadata={
            "help": "whether to input SMILES, description or both. options: smiles, description, smiles_and_description, smiles_and_description_interrupt"
        },
    )
    task: str = field(
        default="summarization",
        metadata={
            "help": "The name of the task, should be summarization (or summarization_{dataset} for evaluating "
                    "pegasus) or translation (or translation_{xx}_to_{yy})."
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge/sacreblue) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge/sacreblue) on "
                    "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocess: bool = field(
        default=True,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessed_folder: Optional[str] = field(
        default=None,
        metadata={
            "help": "Folder to preprocessed data"
        },
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_prefix_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum prefix length."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    meta_negative: int = field(
        default=-1, metadata={"help": "Negative Schema Number in Training."}
    )
    ordered_prompt: bool = field(
        default=True,
        metadata={
            "help": "Whether to sort the spot prompt and asoc prompt or not."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

    early_stop_epoch_num: int = field(
        default=3, metadata={"help": "Stop training when metric_for_best_model has not improved for early_stop_epoch_num epochs"}
    )

    data_source: Optional[str] = field(
        default="ddinter", metadata={"help": "The DDI explanation source: ddinter/drugbank."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # write the training config to the output folder
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        root_dir, base = os.path.split(training_args.output_dir)
        if not base:
            root_dir, base = os.path.split(root_dir)

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        if training_args.do_train:
            with open(sys.argv[1]) as f:
                with open(os.path.join(root_dir, "trainer_config.json"), "w") as fout:
                    fout.write(f.read())
    

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    # logger.setLevel(logging.ERROR)

    logger.info("Options:")
    logger.info(model_args)
    logger.info(data_args)
    logger.info(training_args)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        # transformers.utils.logging.set_verbosity(transformers.logging.ERROR)
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `record_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]

    logger.info(data_files)
    datasets = load_dataset("json", data_files=data_files, download_mode="force_redownload")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    logger.info(datasets)
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    logger.info("Load Config: %s" % model_args.config_name if model_args.config_name else model_args.model_name_or_path)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.max_length = data_args.max_target_length

    tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            sep_token = '<sep>'
    )

    to_remove_token_list = list()
    if tokenizer.bos_token:
        to_remove_token_list += [tokenizer.bos_token]
    if tokenizer.eos_token:
        to_remove_token_list += [tokenizer.eos_token]
    if tokenizer.pad_token:
        to_remove_token_list += [tokenizer.pad_token]

    model = T5ForConditionalGenerationMultTask.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            mirror='tuna'
        )

    logger.info(tokenizer)

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # To serialize preprocess_function below, each of those four variables needs to be defined (even if we won't use
    # them all).

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.error(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )


    def preprocess_function(example):
        drug1_rep = example['drug1_smiles']
        drug2_rep = example['drug2_smiles']
        label = "positive" if example['label'] else "negative"
        if data_args.data_source == 'ddinter':
            explanation = example['ddinter_explanation']
        elif data_args.data_source == 'drugbank':
            explanation = example['drugbank_explanation']
        else:
            raise NotImplementedError

        if data_args.input_format == 'smiles':
            inputs = "DRUG1 %s ; DRUG2 %s"%(drug1_rep, drug2_rep)
        else:
            raise NotImplementedError

        outputs = "%s explanation: %s"%(label, explanation)

        # pad all source to max length for easy drug mask computing
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding="max_length", truncation=True)
        drug1_str = "DRUG1 %s"%drug1_rep
        drug1_inputs = tokenizer(drug1_str, max_length=data_args.max_source_length, padding="max_length", truncation=True)
        drug1_mask = drug1_inputs['attention_mask']
        drug2_mask = [x - y for (x, y) in zip(model_inputs['attention_mask'], drug1_mask)]
        if sum(drug1_mask) < data_args.max_source_length:
            drug2_mask[sum(drug1_mask)] = 0  # ;

        # if sum(model_inputs['attention_mask']) >= data_args.max_source_length:
        #     # logger.warn("truncate input")
        #     pass
        # if sum(drug1_mask) >= data_args.max_source_length:
        #     logger.warn("truncate drug1")


        with tokenizer.as_target_tokenizer():
            labels = tokenizer(outputs, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(_label if _label != tokenizer.pad_token_id else -100) for _label in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["cls_labels"] = float(1) if label == 'positive' else float(0)
        model_inputs["drug1_mask"] = drug1_mask
        model_inputs["drug2_mask"] = drug2_mask

        return model_inputs


    logger.info("Start Data Preprocessing ...")

    if training_args.do_train:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))


        train_dataset = train_dataset.map(
            preprocess_function,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache
        )



    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))


        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache
        )



    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))


        test_dataset = test_dataset.map(
            preprocess_function,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache
        )


    logger.info("End Data Preprocessing ...")

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    def postprocess_text(x_str):
        # Clean `bos` `eos` `pad` for cleaned text
        for to_remove_token in to_remove_token_list:
            x_str = x_str.replace(to_remove_token, '')
        
        return x_str.strip()
    
    exp_metric = evaluate.load("bleu")
    cls_metric = evaluate.combine(["accuracy", "recall", "precision", "f1"])

    
    def compute_metrics(eval_preds):

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        decoded_preds = [postprocess_text(x) for x in decoded_preds]
        decoded_labels = [postprocess_text(x) for x in decoded_labels]

        gold_labels = []
        pred_labels = []
        gold_exps = []
        pred_exps = []
        broken_format = 0

        for pred, gold in zip(decoded_preds, decoded_labels):
            if pred.startswith("positive"):
                pred_labels.append(1)
                pred_exp = pred.replace("positive", "", 1).strip()
            elif pred.startswith("negative"):
                pred_labels.append(0)
                pred_exp = pred.replace("negative", "", 1).strip()
            else:
                pred_labels.append(0)
                pred_exp = pred
                broken_format += 1
            if pred_exp.startswith("explanation:"):
                pred_exp = pred_exp.replace("explanation:", "", 1).strip()

            pred_exps.append(pred_exp)

            if gold.startswith("positive"):
                gold_labels.append(1)
                gold_exp = gold.replace("positive", "", 1).strip()
            elif gold.startswith("negative"):
                gold_labels.append(0)
                gold_exp = gold.replace("negative", "", 1).strip()
            else:
                raise Exception("Illegal gold explanation format!")
            if gold_exp.startswith("explanation:"):
                gold_exp = gold_exp.replace("explanation:", "", 1).strip()
            
            gold_exps.append(gold_exp)
            

        pred_exps_pos = [pred_exps[i] for i, value in enumerate(gold_labels) if value == 1]  # pred exps for gold label==1
        gold_exps_pos = [gold_exps[i] for i, value in enumerate(gold_labels) if value == 1]
        bleu_pos = exp_metric.compute(predictions=pred_exps_pos, references=gold_exps_pos)

        pred_exps_neg = [pred_exps[i] for i, value in enumerate(gold_labels) if value == 0]  # pred exps for gold label==0
        gold_exps_neg = [gold_exps[i] for i, value in enumerate(gold_labels) if value == 0]
        bleu_neg = exp_metric.compute(predictions=pred_exps_neg, references=gold_exps_neg)

        exp_result = exp_metric.compute(predictions=pred_exps, references=gold_exps)
        result = cls_metric.compute(predictions=pred_labels, references=gold_labels)

        result['bleu'] = exp_result['bleu']
        result['bleu_pos'] = bleu_pos['bleu']
        result['bleu_neg'] = bleu_neg['bleu']
        result["broken_pred"] = broken_format

        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    
    start_time = time.time()
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None
    )

    trainer.add_callback(EarlyStoppingCallback(data_args.early_stop_epoch_num))

    

    # Training
    if training_args.do_train:
        if model_args.from_checkpoint:
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
            elif os.path.isdir(model_args.model_name_or_path):
                checkpoint = model_args.model_name_or_path
        checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        end_time = time.time()
        training_duration = end_time - start_time
        train_result.metrics['total_training_time'] = training_duration
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
                for key, value in sorted(vars(model_args).items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
                for key, value in sorted(vars(training_args).items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
                for key, value in sorted(vars(data_args).items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))



    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        results = trainer.evaluate(max_length=data_args.val_max_target_length, num_beams=data_args.num_beams)
        results = {k: round(v, 4) for k, v in results.items()}

        eval_results = trainer.predict(
            eval_dataset,
            metric_key_prefix="eval",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams
        )

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            if training_args.predict_with_generate:
                eval_preds = tokenizer.batch_decode(
                    eval_results.predictions, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                output_test_preds_file = os.path.join(training_args.output_dir, "eval_preds.txt")
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(eval_preds))

    if training_args.do_predict:
        logger.info("*** Test ***")

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        test_metrics = test_results.metrics
        test_metrics["test_loss"] = round(test_metrics["test_loss"], 4)

        output_test_result_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_result_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in sorted(test_metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                output_test_preds_file = os.path.join(training_args.output_dir, "test_preds.txt")
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(test_preds))

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":

    # sys.argv.extend(['configs/molt5_finetune_joint_train.json'])
    sys.argv.extend(['configs/molt5_finetune_joint_train_inference.json'])
    main()
