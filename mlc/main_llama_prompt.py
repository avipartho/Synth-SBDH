
#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import os
import json
import torch
import sys
import numpy as np

import logging
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

import numpy as np

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
    EvalPrediction,
    BitsAndBytesConfig
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.cache_utils import Cache

from trainer.data import DataCollatorForMambaDataset, proc_labels, proc_note
from trainer.metrics import all_metrics, find_threshold_micro
from trainer.config_labels import label2index
from trainer_hf.modeling_llama import LlamaForMaskedLM

from datasets import load_dataset, disable_caching
disable_caching()

torch.autograd.set_detect_anomaly(True)
import wandb
logger = logging.getLogger(__name__)



import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint

def proc_prompt(index2prompt,mask_token):
    num_prompts = len(index2prompt)
    prompts = []
    for i in range(num_prompts):
        prompta = index2prompt[i]
        prompts.append(prompta)
    prompts = f"{mask_token}\n".join(prompts)
    prompts += mask_token
    return prompts


class EnsureMinLR(TrainerCallback):
    """A callback to ensure the learning rate never goes below a minimum threshold."""
    def __init__(self, min_lr):
        self.min_lr = min_lr

    def on_step_begin(self, args, state, control, **kwargs):
        """Adjust the learning rate at the beginning of each step."""
        for param_group in kwargs['optimizer'].param_groups:
            if param_group['lr'] < self.min_lr:
                param_group['lr'] = self.min_lr

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    data_path: Optional[str] = field(
        default=None, metadata={"help": "path to json data folder"}
    )
    global_attention_strides: Optional[int] = field(
        default=3,
        metadata={
            "help": "how many gap between each (longformer) golabl attention token in prompt code descriptions, set to 1 for maximum accuracy, but requires more gpu memory."
        },
    )


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
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
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
    finetune_terms: str = field(
        default="no",
        metadata={"help": "what terms to train like bitfit (bias)."},
    )
    do_rope: bool = field(
        default=False,
        metadata={"help": "Do rope for position or not."},
    )
    
@dataclass
class CustomTrainingArguments(TrainingArguments):
    lora_module_path: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained Lora ckpt path."},
    )

def main(): 
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.broadcast_buffers = False
    training_args.gradient_checkpointing = True
    # if 'Meta-Llama-3-8B' not in model_args.model_name_or_path and 'Meta-Llama-2-8B' not in model_args.model_name_or_path:
    #     raise ValueError(f"Provided model {model_args.model_name_or_path} is not supported.")
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    if is_main_process(training_args.local_rank):
       wandb.init(name=training_args.run_name, project='sdoh_multi_label', entity="avijit")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint, lora_module_ckpt = None, None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if training_args.lora_module_path and not training_args.do_train and training_args.do_predict:
        with open(f'{training_args.lora_module_path}/trainer_state.json') as f:
            lora_module_ckpt = json.load(f)['best_model_checkpoint']
            
    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer.padding_side='right'
    tokenizer.truncation_side='left'
    # tokenizer.eos_token = "</s>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.mask_token = tokenizer.unk_token if 'Llama-2' in model_args.model_name_or_path else '<|reserved_special_token_0|>'
    tokenizer.mask_token_id = tokenizer.unk_token_id if 'Llama-2' in model_args.model_name_or_path else 128002

    # note, we don't use a leading space here as that creates two tokens for llama
    label_no = tokenizer("no",add_special_tokens=False)['input_ids'][0] 
    label_yes = tokenizer("yes",add_special_tokens=False)['input_ids'][0]

    data_files = {
        'train':data_args.data_path+"_train.json",
        'validation':data_args.data_path+"_valid.json",
        'test':data_args.data_path+"_test.json"
    }

    # Preprocess text data
    raw_dataset = load_dataset('json', data_files=data_files, field='data')
    column_names = [k for k,_ in raw_dataset['train'].features.items()]
    label2index_ = label2index[data_args.data_path.split('/')[-1]]
    index2prompt = {
        i: f"Any indication of '{' '.join(sbdh.split('_')[1:])}'?" for sbdh,i in label2index_.items()
    }
    def tokenize_function(examples):
        notes = proc_note(examples["text"]) if 'mimic' in data_args.data_path else examples["text"]
        prompts = proc_prompt(index2prompt,tokenizer.mask_token)
        notes = ["Text: "+a+"\nAnalysis:\n"+prompts for a in notes]
        results = tokenizer(notes, padding="max_length", max_length=data_args.max_seq_length, truncation=True, return_tensors="pt")
        # results = tokenizer(notes, padding="longest", return_tensors="pt")
        labels = proc_labels(label_no, label_yes, examples, label2index_)
        results["label_ids"] = labels
        return results 
    
    processed_dataset =  raw_dataset.map(tokenize_function, batched=True, batch_size=1000, remove_columns=column_names)
    print('### Check processed data ###')
    for k in processed_dataset.keys():
        print('***',k,'***')
        print(processed_dataset[k])
        print(processed_dataset[k][0])
        print(tokenizer.decode(processed_dataset[k][0]['input_ids']))
        
    train_dataset = processed_dataset['train']
    eval_dataset = processed_dataset['validation']
    test_dataset = processed_dataset['test']

    # load config, model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=2,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.label_yes = label_yes #4754
    config.label_no = label_no #642
    config.mask_token_id = tokenizer.mask_token_id 

    if model_args.do_rope:
        orig_rope_scaling = getattr(config, "rope_scaling", None)
        if orig_rope_scaling is None:
            orig_rope_scaling = {"factor": 1}
        orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling_factor
            if data_args.max_seq_length > orig_ctx_len:
                scaling_factor = float(math.ceil(data_args.max_seq_length / orig_ctx_len))
                config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    if training_args.do_train: # train starting from base model
        model = LlamaForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            quantization_config=nf4_config
        )
        if training_args.lora_module_path is not None: # train starting from pretrained lora module ckpt
            print('Loading and merging LoRA adapter weights to base weights from {:}'.format(training_args.lora_module_path))
            model = PeftModel.from_pretrained(model,training_args.lora_module_path,is_trainable=True)
            model.merge_and_unload()
        
        peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.1,
            r=64,
            bias="none",
                target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            task_type="CAUSAL_LM",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif training_args.do_predict: # inference
        model = PeftModel.from_pretrained(model,lora_module_ckpt)

    data_collator = DataCollatorForMambaDataset(tokenizer)


    # Get the metric function
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        y = p.label_ids==config.label_yes
        threshold = find_threshold_micro(preds, y)
        logger.info("best logits threshold:")
        logger.info(str(threshold))
        result = all_metrics(y, preds, k=[5, 8, 15], threshold=threshold)
        return result

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EnsureMinLR(min_lr=1e-5)]
    )


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # trainer.save_model(training_args.output_dir)  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_predict:  
        predict_result = trainer.predict(test_dataset)
        metrics = predict_result.metrics
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        if trainer.is_world_process_zero():
            predictions = predict_result.predictions
            y = predict_result.label_ids==config.label_yes
            output_prediction_file = os.path.join(training_args.output_dir, "predictions.npy")
            np.save(output_prediction_file,predictions)
            output_prediction_file = os.path.join(training_args.output_dir, "labels.npy")
            np.save(output_prediction_file,y)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()