# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import shutil
import logging
import json

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration
from model_t5_pg import t5_pg
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed

from model_utils import TaskPrefixDataCollator, TaskPrefixDataCollatorV2, TaskPrefixTrainer, TaskPrefixTrainerV2, Seq2SeqTrainerNER


def get_config_dir(args):
    return f'{args.dataset}/{args.from_pretrained.split("/")[1]}/{args.model_type}/{args.llm}/{args.subsample}/{args.label_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}'


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics, ner_labels):
    set_seed(run)
    
    config_dir = get_config_dir(args)
    output_dir = f'ckpts/{config_dir}/{run}'  # for model ckpts
    logging_dir = f'logs/{config_dir}/{run}'  # for training logs
    if os.path.isdir(output_dir) and not args.do_train and args.do_predict:
        with open(f'{output_dir}/trainer_state.json') as f:
            best_model_checkpoint = json.load(f)['best_model_checkpoint']
    
    if '-ptr' in args.from_pretrained:
        print('######### Using T5 with pointer module. #########')
        model = t5_pg.from_pretrained(args.from_pretrained.replace('-ptr','') if args.do_train else best_model_checkpoint,
                                      ner_tag_input_ids=tokenizer.convert_tokens_to_ids(ner_labels),
                                      use_pgen=True)
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained if args.do_train else best_model_checkpoint)
    
    if args.dataset in ['sbdh_gpt4_v2', 'sbdh_gpt4_v3', 'sbdh_gpt4_msf', 'sbdh_gpt4_msf_v3']: model.resize_token_embeddings(len(tokenizer))

    if args.parallelize:
        model.parallelize()
    
    if args.logging_strategy == 'no': logging_dir = None

    # clear output dir if already exists
    if os.path.exists(output_dir) and args.do_train:
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir,
        remove_unused_columns = False,
        evaluation_strategy = args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_dir=logging_dir,
        logging_strategy=args.logging_strategy,
        logging_steps=args.eval_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        load_best_model_at_end=args.load_best_model_at_end,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        seed=run,
        local_rank=args.local_rank,
        bf16=args.bf16,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        prediction_loss_only=False,
    )

    if args.model_type == 'task_prefix':
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    elif args.model_type == 'task_prefix_v2':
        data_collator = TaskPrefixDataCollatorV2(tokenizer=tokenizer, model=model)
    elif args.model_type == 'standard':
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        raise ValueError


    trainer_kwargs = {
        'alpha': args.alpha,
        'output_rationale': args.output_rationale,
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': tokenized_datasets["valid"],
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }
    
    if args.model_type == 'task_prefix':
        trainer = TaskPrefixTrainer(**trainer_kwargs)
    elif args.model_type == 'task_prefix_v2':
        trainer = TaskPrefixTrainerV2(**trainer_kwargs)
    elif args.model_type == 'standard':
        trainer_kwargs.pop('alpha')
        trainer_kwargs.pop('output_rationale')
        if args.dataset in ['sbdh_gpt4_v2', 'sbdh_gpt4_v3', 'sbdh_gpt4_msf', 'sbdh_gpt4_msf_v3']:
            trainer = Seq2SeqTrainerNER(**trainer_kwargs)
        else:
            trainer = Seq2SeqTrainer(**trainer_kwargs)
    else:
        raise ValueError
    
    if args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    if args.do_predict:
        predict_result = trainer.predict(
                tokenized_datasets["test"], max_length=args.generation_max_length, num_beams=args.generation_num_beams
            )
        metrics = predict_result.metrics
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)