
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
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import sys
import numpy as np
# from train_parser import generate_parser, print_metrics
# from train_utils import generate_output_folder_name, generate_model
# from find_threshold import find_threshold_micro

import logging
import random
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import expit
from sklearn.metrics import classification_report

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
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from trainer.model import MambaLMHForPT
from trainer.trainer_mamba import MambaTrainer
from trainer.data import DataCollatorForMambaDataset, proc_note, proc_labels 
from trainer.metrics import all_metrics, find_threshold_micro
from trainer.config_labels import label2index


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
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead, MaskedLMOutput

from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, SequenceClassifierOutput

class RobertaForMultiLabelClassification(RobertaForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
    # global_attention_strides: Optional[int] = field(
    #     default=3,
    #     metadata={
    #         "help": "how many gap between each (longformer) golabl attention token in prompt code descriptions, set to 1 for maximum accuracy, but requires more gpu memory."
    #     },
    # )


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

def main(): 
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.broadcast_buffers = False
        
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
    last_checkpoint = None
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

    if os.path.isdir(training_args.output_dir) and not training_args.do_train and training_args.do_predict:
        with open(f'{training_args.output_dir}/trainer_state.json') as f:
            best_model_checkpoint = json.load(f)['best_model_checkpoint']
            
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
    
    data_files = {
        # 'train':os.path.join(data_args.data_path,"sbdh_gpt4_msf_multilabel_train.json"),
        # 'validation':os.path.join(data_args.data_path,"sbdh_gpt4_msf_multilabel_valid.json"),
        # 'test':os.path.join(data_args.data_path,"sbdh_gpt4_msf_multilabel_test.json")
        # 'test':'/home/avijit/playground/sdoh/annotation-dataset-sdoh/mimic_test.json'
        'train':data_args.data_path+"_train.json",
        'validation':data_args.data_path+"_valid.json",
        'test':data_args.data_path+"_test.json"
    }

    raw_dataset = load_dataset('json', data_files=data_files, field='data')
    column_names = [k for k,v in raw_dataset['train'].features.items()]
    def tokenize_function(examples):
        notes = proc_note(examples["text"]) if 'mimic' in data_args.data_path else examples["text"]
        results = tokenizer(notes, padding="max_length", max_length=data_args.max_seq_length, truncation=True, return_tensors="pt")
        # results = tokenizer(notes, padding="longest", return_tensors="pt")
        labels = proc_labels(None, None, examples, label2index[data_args.data_path.split('/')[-1]], return_raw=True)
        results["label_ids"] = labels
           
        return results 
    
    processed_dataset =  raw_dataset.map(tokenize_function, batched=True, batch_size=1000, remove_columns=column_names)
    print('### Check processed data ###')
    for k in processed_dataset.keys():
        print('***',k,'***')
        print(processed_dataset[k])
        print(processed_dataset[k][0])
        print(tokenizer.decode(processed_dataset[k][0]['input_ids']))
        print(np.array(processed_dataset[k]['label_ids']).sum(axis=0))
        
    train_dataset = processed_dataset['train']
    eval_dataset = processed_dataset['validation']
    test_dataset = processed_dataset['test']

    # load config, model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=len(label2index[data_args.data_path.split('/')[-1]]),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # config.label_yes = label_yes #4754
    # config.label_no = label_no #642
    # config.mask_token_id = tokenizer.mask_token_id #50254
    config.problem_type = "multi_label_classification"
    
    model = RobertaForMultiLabelClassification.from_pretrained(
        model_args.model_name_or_path if training_args.do_train else best_model_checkpoint,
        ignore_mismatched_sizes=True,
        config=config
    )

    # data_collator = DataCollatorForMambaDataset(tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer)

    # Get the metric function
    def compute_metrics(p: EvalPrediction):
        preds = expit(p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions)
        y = p.label_ids
        threshold = find_threshold_micro(preds, y)
        logger.info("best logits threshold:")
        logger.info(str(threshold))
        result = all_metrics(y, preds, k=[5, 8, 15], threshold=threshold)
        # print(classification_report(y, y_pred_aux, digits=4, zero_division=0))
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
            predictions = expit(predict_result.predictions)
            output_prediction_file = os.path.join(training_args.output_dir, "predictions.npy")
            np.save(output_prediction_file,predictions)
            output_prediction_file = os.path.join(training_args.output_dir, "labels.npy")
            np.save(output_prediction_file,predict_result.label_ids)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()