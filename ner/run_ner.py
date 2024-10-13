#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning a Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path
import numpy as np

import datasets
import torch
from datasets import ClassLabel, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import classification_report as sklearn_classification_report
from seqeval_custom import classification_report as cr_custom

import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from model_roberta import RobertaNer
from utils import write_file

logger = logging.getLogger(__name__)

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--logfile", type=str, default=None, help="Save logs in this file."
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lenght` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--do_aux_task",
        action="store_true",
        help="If passed, perform an auxilary task to classify if a sequence contains multi-word named entity.",
    )
    parser.add_argument(
        "--use_accelerator",
        action="store_true",
        help="If passed, use accelerator otherwise use pytorch's dataparallel method."
    )
    parser.add_argument(
        "--no_cuda",
        action='store_true',
        help="If passed, do not use GPU."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--train_size", type=float, default=1.0, help="How much (in %) of the training data to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_proportion", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--best_result_file", type=str, default=None, help="Where to store the best results.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args

def evaluate(eval_dataloader, model, args, accelerator, get_labels, device=None):
    y_true, y_pred = None, None
    y_true_aux, y_pred_aux = None, None
    for step, batch in tqdm(enumerate(eval_dataloader),total=math.ceil(len(eval_dataloader)/args.gradient_accumulation_steps),
                            disable=not accelerator.is_local_main_process if args.use_accelerator else False):
        with torch.no_grad():
            if args.use_accelerator:
                outputs = model(**batch, output_hidden_states=True)
            else:
                input_ids = batch['input_ids'].to(device), 
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(
                    input_ids = input_ids[0], 
                    attention_mask = attention_mask,
                    output_hidden_states=True
                    )
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        if args.do_aux_task: 
            aux_predictions = outputs.aux_logits.argmax(dim=-1)
            aux_labels = batch["aux_labels"]
        
        if args.use_accelerator:
            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                if args.do_aux_task: 
                    aux_predictions = accelerator.pad_across_processes(aux_predictions, dim=1, pad_index=-100)
                    aux_labels = accelerator.pad_across_processes(aux_labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)
        else:
            predictions_gathered = predictions
            labels_gathered = labels

        preds, refs = get_labels(predictions_gathered, labels_gathered)
        if args.do_aux_task: 
            if args.use_accelerator:
                aux_preds = accelerator.gather(aux_predictions)
                aux_preds = aux_preds.detach().cpu().clone().numpy()
                aux_labels = accelerator.gather(aux_labels)
                aux_labels = aux_labels.detach().cpu().clone().numpy()
            else:
                aux_preds = aux_predictions.detach().cpu().clone().numpy()
                aux_labels = aux_labels.detach().cpu().clone().numpy()

        # If we are in a multiprocess environment, the last batch has duplicates
        if args.use_accelerator:
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions_gathered = predictions_gathered[: len(eval_dataloader.dataset) - samples_seen]
                    labels_gathered = labels_gathered[: len(eval_dataloader.dataset) - samples_seen]
                    if args.do_aux_task: 
                        aux_preds = aux_preds[: len(eval_dataloader.dataset) - samples_seen]
                        aux_labels = aux_labels[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += labels_gathered.shape[0]

        if y_true is None:
            y_true = refs
            y_pred = preds
            if args.do_aux_task: 
                y_true_aux = aux_labels
                y_pred_aux = aux_preds
        else:
            y_true += refs                
            y_pred += preds
            if args.do_aux_task: 
                y_true_aux = np.concatenate((y_true_aux,aux_labels)) 
                y_pred_aux = np.concatenate((y_pred_aux,aux_preds)) 
    
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    macro_f = f1_score(y_true, y_pred, average='macro')
    relaxed_report = cr_custom(y_true, y_pred, digits=4, criteria='relaxed')[0]
    if args.do_aux_task: 
        report_aux = sklearn_classification_report(y_true_aux, y_pred_aux, digits=4, zero_division=0)

    return (report, relaxed_report, macro_f, report_aux) if args.do_aux_task else (report, relaxed_report, macro_f)

def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(filename = args.logfile , filemode ='a', 
        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt = '%m/%d/%Y %H:%M:%S',
        level = logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    if args.use_accelerator:
        if args.no_cuda:
            raise ValueError("To run on cpu, remove `use_accelerator` flag and use `no_cuda` flag")
        
        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        accelerator = Accelerator()
        logger.info(accelerator.state)

        # Setup logging, we only want one process per machine to log things on the screen.
        # accelerator.is_local_main_process is only True for one process per machine.
        logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
        device = None # a placeholder, not necessary when accelerator is used.
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

        # Setup logging.
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()

        accelerator = None # a placeholder, not necessary when accelerator is not used.

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    logger.info(f'Using seed {args.seed}')
    
    # accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = args.train_file.split(".")[-1]
        try: raw_datasets = load_dataset(extension, data_files=data_files, field='data')
        except: raw_datasets = load_dataset(extension, data_files=data_files)

    if args.train_size < 1.0:
        raw_datasets["train"] = raw_datasets['train'].train_test_split(test_size=1.0-args.train_size, seed=0)['train']
        
    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    if args.text_column_name is not None:
        text_column_name = args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if args.label_column_name is not None:
        label_column_name = args.label_column_name
    elif f"{args.task_name}_tags" in column_names:
        label_column_name = f"{args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i, _ in enumerate(label_list)} # labels are already in numerical order
    elif args.dataset_name is not None and "tner" in args.dataset_name:
        if "mit_restaurant" in args.dataset_name:
            label_to_id = {"O": 0, "B-Rating": 1, "I-Rating": 2, "B-Amenity": 3, "I-Amenity": 4, "B-Location": 5, "I-Location": 6, "B-Restaurant_Name": 7, "I-Restaurant_Name": 8, "B-Price": 9, "B-Hours": 10, "I-Hours": 11, "B-Dish": 12, "I-Dish": 13, "B-Cuisine": 14, "I-Price": 15, "I-Cuisine": 16}
        elif "mit_movie" in args.dataset_name:
            label_to_id = {"O": 0, "B-Actor": 1, "I-Actor": 2, "B-Plot": 3, "I-Plot": 4, "B-Opinion": 5, "I-Opinion": 6, "B-Award": 7, "I-Award": 8, "B-Year": 9, "B-Genre": 10, "B-Origin": 11, "I-Origin": 12, "B-Director": 13, "I-Director": 14, "I-Genre": 15, "I-Year": 16, "B-Soundtrack": 17, "I-Soundtrack": 18, "B-Relationship": 19, "I-Relationship": 20, "B-Character_Name": 21, "I-Character_Name": 22, "B-Quote": 23, "I-Quote": 24}
        label_list = list(label_to_id.keys())
        label_to_id = {i: i for i, _ in enumerate(label_list)} # labels are already in numerical order
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Map that sends B-Xxx label to its I-Xxx counterpart
    # b_to_i_label = []
    # for idx, label in enumerate(label_list):
    #     if label.startswith("B-") and label.replace("B-", "I-") in label_list:
    #         b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
    #     else:
    #         b_to_i_label.append(idx)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, num_labels=num_labels)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    if tokenizer.pad_token is None: # for gpt2
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if args.model_name_or_path:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            quantization_config=nf4_config if 'llama' in args.model_name_or_path else None
        )
        # model = RobertaNer.from_pretrained(
        #     args.model_name_or_path,
        #     from_tf=bool(".ckpt" in args.model_name_or_path),
        #     config=config,
        #     do_aux_task=args.do_aux_task
        # )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForTokenClassification.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
        
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels, aux_labels = [], []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids, aux_label_id = [], 0
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    # if label_ids[-1]%2 == 0 and label_ids[-1]!=0:
                    #     aux_label_id = 1
                    if args.do_aux_task and label[word_idx].startswith("I-"):
                        aux_label_id = 1
                # For the other tokens in a word, we set the label to -100.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
            aux_labels.append(aux_label_id)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["aux_labels"] = aux_labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["validation"]
    test_dataset = processed_raw_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        if args.use_accelerator: pad_to_multiple_of = 8 if accelerator.use_fp16 else None
        else: pad_to_multiple_of = None # no support for fp16 when accelerator is not used
        data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=pad_to_multiple_of
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    # device = accelerator.device
    # model.to(device)

    if args.use_accelerator:
        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, test_dataloader
        )
    else:
        model.to(device)
        if n_gpu>1:
            model = torch.nn.DataParallel(model)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name = args.lr_scheduler_type,
        optimizer = optimizer,
        num_warmup_steps = int(args.warmup_proportion * args.max_train_steps),
        num_training_steps = args.max_train_steps,
    )

    def get_labels(predictions, references):
        # Transform predictions and references tensos to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    # Train!
    if args.use_accelerator:
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    else:
        total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    completed_steps = 0
    best_macro_f, best_eval_report, best_epoch = 0, None, 0

    for epoch in range(args.num_train_epochs):
        logger.info(f"Training epoch = {epoch}")
        model.train()
        tr_loss = 0

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                disable=not accelerator.is_local_main_process if args.use_accelerator else False
                ) 
        for step, batch in progress_bar:
            if not args.use_accelerator:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(
                    input_ids = input_ids, 
                    attention_mask = attention_mask,
                    labels = labels
                    )
                loss = outputs[0].mean() if n_gpu>1 else outputs[0]
            else:
                outputs = model(**batch)
                loss = outputs.loss

            loss = loss / args.gradient_accumulation_steps
            if args.use_accelerator:
                accelerator.backward(loss)
            else:
                loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            progress_bar.set_description("Epoch {:}, loss: {:.4f}".format(epoch,tr_loss))
            tr_loss += loss.item()

            if completed_steps >= args.max_train_steps:
                break
            
        logger.info(f"Training loss = {tr_loss}")
        
        # Evaluate on dev data
        model.eval()
        logger.info("##### Validation set result #####")
        if args.do_aux_task: 
            eval_report, eval_relaxed_report, macro_f, eval_report_aux = evaluate(eval_dataloader, model, args, accelerator, get_labels, device)      
            logger.info("\n***Exact Matching***\n%s", eval_report)
            logger.info("\n***Relaxed Matching***\n%s", eval_relaxed_report)
            logger.info("\n%s", eval_report_aux)
        else:
            eval_report, eval_relaxed_report, macro_f = evaluate(eval_dataloader, model, args, accelerator, get_labels, device)      
            logger.info("\n***Exact Matching***\n%s", eval_report)
            logger.info("\n***Relaxed Matching***\n%s", eval_relaxed_report)

        # Save predictions when the model improves ner performance on dev set.
        if args.output_dir is not None and macro_f>=best_macro_f:
            best_macro_f = macro_f
            best_eval_report = eval_report
            best_epoch = epoch
            logger.info("##### Test set result #####")
            if args.do_aux_task:
                test_report, test_relaxed_report, _, test_report_aux = evaluate(test_dataloader, model, args, accelerator, get_labels, device)  
                logger.info("\n***Exact Matching***\n%s", test_report)
                logger.info("\n***Relaxed Matching***\n%s", test_relaxed_report)
                logger.info("\n%s", test_report_aux)
            else:
                test_report, test_relaxed_report, _ = evaluate(test_dataloader, model, args, accelerator, get_labels, device)  
                logger.info("\n***Exact Matching***\n%s", test_report)
                logger.info("\n***Relaxed Matching***\n%s", test_relaxed_report)

            if not args.use_accelerator or accelerator.is_local_main_process:
                    report = eval_report+'\n'+eval_relaxed_report+'\n'+test_report+'\n'+test_relaxed_report
                    write_file(report,args.best_result_file,mode='w')
                    
            if args.use_accelerator:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )
                    tokenizer.save_pretrained(args.output_dir)
            else:
                if args.no_cuda or torch.cuda.device_count()==1:
                    model.save_pretrained(args.output_dir)
                else:
                    model.module.save_pretrained(args.output_dir)
            
                tokenizer.save_pretrained(args.output_dir)

    logger.info("Finished training, best result achieved at epoch {:}".format(best_epoch))
    # logger.info("\n%s", best_eval_report)
    # logger.info("\n%s", report)
    # print("\n%s", best_eval_report)
    # print("\n%s", report)


if __name__ == "__main__":
    main()
