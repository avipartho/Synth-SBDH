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


import argparse

from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader, SBDHDatasetLoader
from metrics import compute_text_acc, compute_equation_acc, compute_metrics_text, compute_metrics_equation, compute_metrics_text_aux, compute_metrics_equation_aux, compute_metrics_ner
from train_utils import train_and_evaluate


def run(args):
    #### Prepare datasets
    if args.dataset == 'cqa':
        dataset_loader = CQADatasetLoader()
    elif args.dataset == 'svamp':
        dataset_loader = SVAMPDatasetLoader()
    elif args.dataset == 'esnli':
        dataset_loader = ESNLIDatasetLoader()
    elif args.dataset == 'anli1':
        dataset_loader = ANLI1DatasetLoader()
    elif args.dataset == 'asdiv':  # NOTE: for augmenting SVAMP only
        dataset_loader = SVAMPDatasetLoader()
        dataset_loader_svamp = SVAMPDatasetLoader()
        dataset_loader_asdiv = ASDivDatasetLoader()
    elif args.dataset in ['sbdh_gpt4_v2', 'sbdh_gpt4_v3', 'sbdh_gpt4_msf', 'sbdh_gpt4_msf_v3']:
        dataset_loader = SBDHDatasetLoader(args.dataset)
    else:
        raise ValueError

    if args.dataset == 'asdiv':
        datasets_svamp = dataset_loader_svamp.load_from_json()
        datasets_asdiv = dataset_loader_asdiv.load_from_json()
        datasets = DatasetDict({
            'train': concatenate_datasets([datasets_svamp['train'], datasets_asdiv['train']]),
            'test': datasets_svamp['test']
        })
    else:
        datasets = dataset_loader.load_from_json()

    if args.llm is None:
        pass
    elif args.llm == 'palm':
        if args.dataset == 'asdiv':
            # training set = SVAMP training + ASDiv training
            train_llm_rationales_svamp, train_llm_labels_svamp = dataset_loader_svamp.load_llm_preds(split='train')
            train_llm_rationales_asdiv, train_llm_labels_asdiv = dataset_loader_asdiv.load_llm_preds(split='train')
            train_llm_rationales = train_llm_rationales_svamp + train_llm_rationales_asdiv
            train_llm_labels = train_llm_labels_svamp + train_llm_labels_asdiv
            # test set = SVAMP test
            test_llm_rationales, test_llm_labels = dataset_loader_svamp.load_llm_preds(split='test')
            if args.model_type == 'task_prefix_v2':
                raise ValueError
        elif args.dataset in ['sbdh_gpt4_v2', 'sbdh_gpt4_v3', 'sbdh_gpt4_msf', 'sbdh_gpt4_msf_v3']:
            pass # Nothing to do
        else:
            train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(split='train')
            test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds(split='test')
            if args.model_type == 'task_prefix_v2': 
                train_llm_rationales_and_labels = dataset_loader.load_llm_preds_v2(split='train')
                test_llm_rationales_and_labels = dataset_loader.load_llm_preds_v2(split='test')
    elif args.llm == 'gpt':
        train_llm_rationales, train_llm_labels = dataset_loader.load_gpt_preds(split='train')
        test_llm_rationales, test_llm_labels = dataset_loader.load_gpt_preds(split='test')
        if args.model_type == 'task_prefix_v2':
                raise ValueError
    else:
        raise ValueError

    if args.llm is not None and args.dataset not in ['sbdh_gpt4_v2', 'sbdh_gpt4_v3','sbdh_gpt4_msf', 'sbdh_gpt4_msf_v3']:
        datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
        datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
        datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
        datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales)
        if args.model_type == 'task_prefix_v2':
            datasets['train'] = datasets['train'].add_column('llm_rationale_label', train_llm_rationales_and_labels)
            datasets['test'] = datasets['test'].add_column('llm_rationale_label', test_llm_rationales_and_labels)

    if args.subsample < 1.0:
        datasets['train'] = datasets['train'].train_test_split(test_size=1.0-args.subsample, seed=args.run)['train']

    if args.dataset not in ['sbdh_gpt4_v2', 'sbdh_gpt4_v3', 'sbdh_gpt4_msf', 'sbdh_gpt4_msf_v3']:
        if dataset_loader.has_valid:
            if args.llm is None:
                pass
            elif args.llm == 'palm':
                valid_llm_rationales, valid_llm_labels = dataset_loader.load_llm_preds(split='valid')
                if args.model_type == 'task_prefix_v2': 
                    valid_llm_rationales_and_labels = dataset_loader.load_llm_preds_v2(split='valid')
            elif args.llm == 'gpt':
                valid_llm_rationales, valid_llm_labels = dataset_loader.load_gpt_preds(split='valid')
                if args.model_type == 'task_prefix_v2': 
                    raise ValueError
            else:
                raise ValueError

            datasets['valid'] = datasets['valid'].add_column('llm_label', valid_llm_labels)
            datasets['valid'] = datasets['valid'].add_column('llm_rationale', valid_llm_rationales)
            datasets['valid'] = datasets['valid'].add_column('llm_rationale_label', valid_llm_rationales_and_labels)
        else:
            train_valid_datasets = datasets['train'].train_test_split(test_size=0.1, seed=0)

            datasets = DatasetDict({
                'train': train_valid_datasets['train'],
                'valid': train_valid_datasets['test'],
                'test': datasets['test'],
            })

    if args.label_type == 'gt':
        pass
    elif args.label_type == 'llm' and args.llm is not None:
        if args.dataset not in ['svamp', 'asdiv']:
            train_label_acc = compute_text_acc(datasets['train']['llm_label'], datasets['train']['label'])
            test_label_acc = compute_text_acc(datasets['test']['llm_label'], datasets['test']['label'])
        elif args.dataset in ['sbdh_gpt4_v2', 'sbdh_gpt4_v3', 'sbdh_gpt4_msf', 'sbdh_gpt4_msf_v3']:
            raise ValueError
        else:
            train_label_acc = compute_equation_acc(datasets['train']['llm_label'], datasets['train']['label'])
            test_label_acc = compute_equation_acc(datasets['test']['llm_label'], datasets['test']['label'])

        print(f'LLM Train Acc: {train_label_acc:.4f}')
        print(f'LLM Test Acc: {test_label_acc:.4f}')

        datasets['train'] = datasets['train'].remove_columns('label')
        datasets['train'] = datasets['train'].add_column('label', datasets['train']['llm_label'])

    else:
        raise ValueError

    if args.llm is not None:
        if 'rationale' in datasets['train'].column_names:
            datasets = datasets.remove_columns('rationale')
        datasets = datasets.rename_column('llm_rationale', 'rationale')


    #### Prepare datasets Prepare data for training
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained.replace('-ptr',''))
    label_list = None
    if args.dataset in ['sbdh_gpt4_v2', 'sbdh_gpt4_v3', 'sbdh_gpt4_msf', 'sbdh_gpt4_msf_v3']:
        label_list = ['barriers_to_care', 'substance_abuse', 'housing_insecurity', 'financial_insecurity', 'psychiatric_symptoms_or_disorders', \
            'isolation_or_loss_of_relationship', 'patient_disability', 'violence', 'legal_problems', 'transitions_of_care', 'pain', 'food_insecurity']
        tokenizer.add_tokens(['<'+l+'>' for l in label_list])
        

    if 'nli' in args.dataset:
        datasets = datasets.map(
            lambda example: {'input': tokenizer.eos_token.join([example['premise'], example['hypothesis']])},
            remove_columns=['premise', 'hypothesis'],
        )


    if args.model_type == 'task_prefix' and args.llm is not None:
        def tokenize_function(examples):
            model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=args.generation_max_length, truncation=True)
                rationale_output_encodings = tokenizer(examples['rationale'], max_length=args.generation_max_length, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

            return model_inputs
        
    elif args.model_type == 'task_prefix_v2' and args.llm is not None:
        def tokenize_function(examples):
            model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            cmbd_model_inputs = tokenizer(['explain and predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
            model_inputs['cmbd_input_ids'] = cmbd_model_inputs['input_ids']
            model_inputs['cmbd_attention_mask'] = cmbd_model_inputs['attention_mask']

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=args.generation_max_length, truncation=True)
                rationale_output_encodings = tokenizer(examples['rationale'], max_length=args.generation_max_length, truncation=True)
                llm_output_encodings = tokenizer(examples['llm_rationale_label'], max_length=args.generation_max_length, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels'] = rationale_output_encodings['input_ids']
            model_inputs['aux_labels_2'] = llm_output_encodings['input_ids']

            return model_inputs

    elif args.model_type == 'standard':
        def tokenize_function(examples):
            model_inputs = tokenizer(
                examples['input'],
                max_length=args.max_input_length,
                truncation=True
            )

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=args.generation_max_length, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']

            return model_inputs

    else:
        raise ValueError

    if args.llm is None:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'label'],
            batched=True
        )
    else:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'rationale', 'label', 'llm_label', 'llm_rationale_label'] if args.model_type == 'task_prefix_v2' else ['input', 'rationale', 'label', 'llm_label', 'ex_no'] if args.dataset =='sbdh_gpt4_msf_v3' else ['input', 'rationale', 'label', 'llm_label'],
            batched=True
        )


    if args.model_type == 'standard':
        if args.dataset in ['sbdh_gpt4_v2', 'sbdh_gpt4_v3', 'sbdh_gpt4_msf', 'sbdh_gpt4_msf_v3']:
            compute_metrics = compute_metrics_ner(tokenizer, label_list, args.result_file)
        elif args.dataset not in ['svamp', 'asdiv']:
            compute_metrics = compute_metrics_text_aux(tokenizer)
        else:
            compute_metrics = compute_metrics_equation_aux(tokenizer)

    else:
        if args.dataset in ['sbdh_gpt4_v2', 'sbdh_gpt4_v3', 'sbdh_gpt4_msf', 'sbdh_gpt4_msf_v3']:
            compute_metrics = compute_metrics_ner(tokenizer, label_list, args.result_file)
        elif args.dataset not in ['svamp', 'asdiv']:
            compute_metrics = compute_metrics_text(tokenizer)
        else:
            compute_metrics = compute_metrics_equation(tokenizer)

    print('### Check processed data ###')
    for k in tokenized_datasets.keys():
        print('***',k,'***')
        print(tokenized_datasets[k])
        print(tokenized_datasets[k][0])
        print(tokenizer.decode(tokenized_datasets[k][0]['input_ids']))
        print(tokenizer.decode(tokenized_datasets[k][0]['labels']))
        if args.model_type != 'standard': print(tokenizer.decode(tokenized_datasets[k][0]['aux_labels']))
        
    label_list = ['<'+l+'>' for l in label_list] if label_list is not None else None
    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics, label_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--save_steps', type=int, default=250)
    parser.add_argument('--evaluation_strategy', type=str, default='steps')
    parser.add_argument('--save_strategy', type=str, default='steps')
    parser.add_argument('--logging_strategy', type=str, default='steps')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--label_type', type=str, default='gt')
    parser.add_argument('--llm', type=str, default='palm')
    parser.add_argument('--result_file', type=str, required=True)
    parser.add_argument('--save_total_limit', type=int, default=1)
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--generation_max_length', type=int, default=256)
    parser.add_argument('--generation_num_beams', type=int, default=1)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--metric_for_best_model', type=str, default='loss')
    parser.add_argument('--load_best_model_at_end', action='store_true')
    parser.add_argument('--greater_is_better', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')

    args = parser.parse_args()

    run(args)