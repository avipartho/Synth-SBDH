#!/usr/bin/env python
# coding=utf-8
# @author: Avijit Mitra 

"""
Fine-tuning an  E2E Transformer-based model on cqr.
"""

import argparse
from dataclasses import field
import logging
import math
import os
import random

import datasets
import nltk
import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel,load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock
from utils import write_file
from evaluate import evaluate
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    Adafactor,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import is_offline_mode

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
IGNORE_INDEX = -100

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune an  E2E Transformer-based model on cqr.")
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
        "--test_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--logfile", type=str, default=None, help="Save logs in this file."
    )
    parser.add_argument(
        "--pred_file", type=str, default=None, help="Save predictions and references in this file."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
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
        "--text_column_name",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the input.",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the ner labels.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the Tokenizers library).",
    )
    parser.add_argument(
        "--use_accelerate",
        action="store_true",
        help="If passed, use accelerate package otherwise use pytorch's dataparallel method."
    )
    parser.add_argument(
        "--no_cuda",
        action='store_true',
        help="If passed, do not use GPU."
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="If passed, will perform training on the training set and validation on the dev set.",
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
        help="If passed, will perform evaluation on the test set.",
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
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--result_file", type=str, default=None, help="Where to store the results.")
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
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--prefix_scheme",
        type=str,
        default="v1",
        help="How to add prefix. \
            v1: no prefix, \
            v2: uses prefix - 'Find all entity types from the following text.\nEntity types: ent_type_0 ent_type_1 ... ent_type_n\nText:',\
            ",
        choices=["v1", "v2"],
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(filename = args.logfile , filemode ='a', 
        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt = '%m/%d/%Y %H:%M:%S',
        level = logging.INFO)
    logger = logging.getLogger(__name__)

    if args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        args.source_prefix = ''

    if args.use_accelerate:
        if args.no_cuda:
            raise ValueError("To run on cpu, remove `use_accelerate` flag and use `no_cuda` flag")

        # Initialize the accelerator to handle device placement.
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
        n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    
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
    
    if os.path.isfile(args.pred_file) and accelerator.is_local_main_process:
        logger.info("Found an existing prediction file. Will overwrite.")
        os.remove(args.pred_file)
    
    if os.path.isfile(args.result_file) and accelerator.is_local_main_process:
        logger.info("Found an existing result file. Will overwrite.")
        os.remove(args.result_file)

    # Load the dataset.
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
        raw_datasets = load_dataset(extension, data_files=data_files, field='data')

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
    label_names = [l[2:] for l in label_list if l.startswith('B-')] + ['O']
    label_to_special_token = {l: 'B-<'+l[2:]+'>' for _, l in enumerate(label_list) if l.startswith('B-')}

    # conll2003 >> 'B-PER':'B-<PER>','B-ORG':'B-<ORG>','B-LOC':'B-<LOC>','B-MISC':'B-<MISC>'
    # bleeding >> 'B-BLEEDING_ANATOMIC_SITE': 'B-<BLEEDING_ANATOMIC_SITE>', 'B-BLEEDING_EVENT': 'B-<BLEEDING_EVENT>', 'B-BLEEDING_LAB_EVAL': 'B-<BLEEDING_LAB_EVAL>', 'B-DRUGNAME': 'B-<DRUGNAME>', 'B-SEVERITY': 'B-<SEVERITY>', 'B-TRIGGER_ALTERNATIVE_CAUSE': 'B-<TRIGGER_ALTERNATIVE_CAUSE>'


    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if 'bart' in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=not args.use_slow_tokenizer, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=not args.use_slow_tokenizer)
    
    ner_labels = list(i[2:] for i in label_to_special_token.values())
    tokenizer.add_tokens(ner_labels)
    if tokenizer.pad_token is None: # for gpt2
        tokenizer.pad_token = tokenizer.eos_token

    if args.model_name_or_path!='gpt-2':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    elif args.model_name_or_path=='gpt-2':
        model = AutoModelForCausalLM(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocess the datasets.
    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    # This function expects only one argument. Moving it outside the main() will require passing additional arguments 
    # which goes against the use case of this function. 
    def preprocess_function(examples):
        inputs = examples[text_column_name]
        inputs = [inp for inp in inputs]
        model_inputs = tokenizer(
            inputs, 
            max_length=args.max_source_length, 
            padding=padding, 
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True
            )
        # this includes eos token as well, so exclude that depending on the model
        # for gpt-2 there is no eos token added
        # model_inputs['seq_len'] = [
        #     sum(np.array(input_id)!=tokenizer.pad_token_id) for input_id in model_inputs.input_ids
        #     ]

        gen_labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = model_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            gen_label = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    pass
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    if label_list[label_to_id[label[word_idx]]] != 'O':
                        if label_list[label_to_id[label[word_idx]]].startswith('I-'):
                            try:
                                if 'B-' in gen_label[-1]:
                                    token, tag = gen_label[-1].split(' B-')
                                else:
                                    token, tag = gen_label[-1].split(' I-')
                                gen_label[-1] = token+' '+examples[text_column_name][i][word_idx]+' B-'+tag
                            except: # erroneous label, first token has I- tag    
                                tag = label_to_special_token['B-'+label_list[label_to_id[label[word_idx]]][2:]]
                                gen_label.append(examples[text_column_name][i][word_idx]+' '+tag)
                        else:
                            gen_label.append(
                                examples[text_column_name][i][word_idx]+
                                ' '+ label_to_special_token[label_list[label_to_id[label[word_idx]]]]
                                )
                # For the other tokens in a word, we set the label to -100.
                else:
                    pass
                previous_word_idx = word_idx

            gen_label = ', '.join(gen_label).replace('B-','')
            gen_label = 'none' if gen_label=='' else gen_label
            if args.model_name_or_path=='gpt2':
                gen_label = examples[text_column_name][i]+' '+gen_label
            gen_labels.append(gen_label)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(gen_labels, max_length=max_target_length, padding=padding, truncation=True)
        # if args.model_name_or_path=='gpt2':
        #     for seq_len,label in zip(model_inputs['seq_len'],labels):
        #         label[:seq_len] = IGNORE_INDEX

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        # if accelerator.is_local_main_process: print(gen_labels[-2:])
        return model_inputs

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=column_names, load_from_cache_file=not args.overwrite_cache
    )

    if args.do_train:
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]
        test_dataset = processed_datasets["test"]
        
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    test_dataset = processed_datasets["test"]

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if args.use_accelerate: pad_to_multiple_of = 8 if accelerator.use_fp16 else None
    else: pad_to_multiple_of = None # no support for fp16 when accelerator is not used
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    train_batch_size = args.per_device_train_batch_size if args.use_accelerate else args.per_device_train_batch_size * n_gpu
    eval_batch_size = args.per_device_eval_batch_size if args.use_accelerate else args.per_device_eval_batch_size * n_gpu
    if args.do_train:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_batch_size
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=eval_batch_size)

    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=eval_batch_size)

    # Metric
    metric_bleu = load_metric("bleu")
    metric_rouge = load_metric("rouge")
    metrics = (metric_rouge, metric_bleu)

    if args.do_train:
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
        # if 't5_base' in args.model_name_or_path:
        #     optimizer = Adafactor(optimizer_grouped_parameters, lr=0.001, scale_parameter=False, relative_step=False)
        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8, lr=args.learning_rate)

        if args.use_accelerate:
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
        
        num_warmup_steps = int(args.warmup_proportion * args.max_train_steps)
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # Train!
        if args.use_accelerate:
            total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        else:
            total_batch_size = args.per_device_train_batch_size * n_gpu * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"Using model: {args.model_name_or_path}")
        logger.info(f"Num examples = {len(train_dataset)}")
        logger.info(f"Num Epochs = {args.num_train_epochs}")
        logger.info(f"Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"Total optimization steps = {args.max_train_steps}")
        
        completed_steps = 0
        best_macro_f = 0
        
        for epoch in range(args.num_train_epochs):
            logger.info(f"Training epoch = {epoch}")
            model.train()
            tr_loss = 0
            
            # Only show the progress bar once on each machine when using accelerator.
            progress_bar = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                disable=not accelerator.is_local_main_process if args.use_accelerate else False
                )  

            for step, batch in progress_bar:
                if not args.use_accelerate:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(
                        input_ids = input_ids, 
                        attention_mask = attention_mask,
                        labels=labels)
                    loss = outputs[0].mean() 
                else:
                    outputs = model(**batch)
                    loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                if args.use_accelerate:
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

            # Evaluate on dev & test data!
            logger.info("***** Running evaluation on dev data *****")
            # write_predictions = (epoch == args.num_train_epochs-1) # flag to decide whether to write the model predictions
            if not args.use_accelerate or accelerator.is_local_main_process:
                write_file(f'##### Epoch {epoch} #####\n',args.result_file)
                write_file(f'***** Dev set results *****\n',args.result_file)
            macro_f, dev_result_string = evaluate(model, config, tokenizer, eval_dataloader, accelerator, device, logger, n_gpu, args, metrics, label_names, write_predictions=False)
        
            # Save predictions when the model improves ner performance on dev set.
            if args.output_dir is not None and macro_f>=best_macro_f:
                # Run evaluation on test set if the model ner performance improves on dev set.
                logger.info("***** Running evaluation on test data *****")
                if not args.use_accelerate or accelerator.is_local_main_process:
                    write_file('***** Test set results *****\n',args.result_file)
                test_macro_f, test_result_string = evaluate(model, config, tokenizer, test_dataloader, accelerator, device, logger, n_gpu, args, metrics, label_names, write_predictions=True)
                result_string = dev_result_string+'\n\n'+test_result_string
                write_file(result_string,args.best_result_file,mode='w')
                
                best_macro_f = macro_f
                if args.use_accelerate:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                else:
                    if args.no_cuda:
                        model.save_pretrained(args.output_dir)
                    else:
                        model.module.save_pretrained(args.output_dir)
                
                tokenizer.save_pretrained(args.output_dir)

        logger.info("Best macro-f score on val set: {:.4f}".format(best_macro_f))

    if args.do_test and not args.do_train: # inference only mode
        
        # Load saved model
        model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir)
        config = AutoConfig.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, use_fast=not args.use_slow_tokenizer)

        # Test!
        if args.use_accelerate:
            model, test_dataloader = accelerator.prepare(model, test_dataloader)
        else:
            model.to(device)
            if n_gpu>1:
                model = torch.nn.DataParallel(model)
        
        # Evaluate on test data!
        logger.info("***** Running evaluation on test data *****")
        if not args.use_accelerate or accelerator.is_local_main_process:
            write_file('***** Test set results *****\n',args.result_file)
        evaluate(model, config, tokenizer, test_dataloader, accelerator, device, logger, n_gpu, args, metrics, label_names, write_predictions=True)


if __name__ == "__main__":
    main()