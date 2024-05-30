#!/bin/sh
#
# baseline model 70371

cd /home/zhichaoyang/mimic3/mimic3bench/2018_Clinical_Trial_Cohort_Selection/
source activate mixtral_env

export CUDA_DEVICE_ORDER="PCI_BUS_ID" 

# # nohup bash run01.sh > run01.out 2> run01.err &

WANDB_PROJECT=ClinicalTrial  CUDA_VISIBLE_DEVICES=6 python main_longformer.py \
                --seed 48 --data_seed 48 \
                --data_path ./data \
                --config_name yikuan8/Clinical-Longformer \
                --tokenizer_name yikuan8/Clinical-Longformer \
                --model_name_or_path yikuan8/Clinical-Longformer \
                --do_train --do_eval --max_seq_length 4096 \
                --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --per_device_eval_batch_size 1 \
                --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
                --learning_rate 5e-6 --weight_decay 1e-2 --num_train_epochs 12 \
                --lr_scheduler_type linear --warmup_ratio 0.15 \
                --logging_steps 50 \
                --evaluation_strategy epoch --save_strategy no \
                --logging_first_step \
                --output_dir ./saved_models/clinicallongformer-test02_run02

WANDB_PROJECT=ClinicalTrial  CUDA_VISIBLE_DEVICES=6 python main_longformer.py \
                --seed 3407 --data_seed 3407 \
                --data_path ./data \
                --config_name yikuan8/Clinical-Longformer \
                --tokenizer_name yikuan8/Clinical-Longformer \
                --model_name_or_path yikuan8/Clinical-Longformer \
                --do_train --do_eval --max_seq_length 4096 \
                --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --per_device_eval_batch_size 1 \
                --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
                --learning_rate 5e-6 --weight_decay 1e-2 --num_train_epochs 12 \
                --lr_scheduler_type linear --warmup_ratio 0.15 \
                --logging_steps 50 \
                --evaluation_strategy epoch --save_strategy no \
                --logging_first_step \
                --output_dir ./saved_models/clinicallongformer-test02_run02

WANDB_PROJECT=ClinicalTrial  CUDA_VISIBLE_DEVICES=7 python main_longformer.py \
                --seed 36 --data_seed 36 \
                --data_path ./data \
                --config_name yikuan8/Clinical-Longformer \
                --tokenizer_name yikuan8/Clinical-Longformer \
                --model_name_or_path yikuan8/Clinical-Longformer \
                --do_train --do_eval --max_seq_length 4096 \
                --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --per_device_eval_batch_size 1 \
                --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
                --learning_rate 5e-6 --weight_decay 1e-2 --num_train_epochs 12 \
                --lr_scheduler_type linear --warmup_ratio 0.15 \
                --logging_steps 50 \
                --evaluation_strategy epoch --save_strategy no \
                --logging_first_step \
                --output_dir ./saved_models/clinicallongformer-test02_run03




# Llama fine-tuning starts from here


WANDB_PROJECT=ClinicalTrial  CUDA_VISIBLE_DEVICES=2 python main_llama.py \
                --seed 48 --data_seed 48 --do_rope True \
                --bf16 True --data_path ./data \
                --config_name /home/zhichaoyang/mimic3/clinical-mamba/saved_models/clinicalllama-pretrain03 \
                --tokenizer_name /home/zhichaoyang/mimic3/clinical-mamba/saved_models/clinicalllama-pretrain03 \
                --model_name_or_path /home/zhichaoyang/mimic3/clinical-mamba/saved_models/clinicalllama-pretrain03 \
                --do_train --do_eval --max_seq_length 15004 \
                --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --per_device_eval_batch_size 1 \
                --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
                --learning_rate 1e-4 --weight_decay 1e-2 --num_train_epochs 12 \
                --lr_scheduler_type linear --warmup_ratio 0.15 \
                --logging_steps 50 \
                --evaluation_strategy epoch --save_strategy no \
                --logging_first_step \
                --output_dir ./saved_models/clinicalllama-test02_run01


WANDB_PROJECT=ClinicalTrial  CUDA_VISIBLE_DEVICES=1 python main_llama.py \
                --seed 3407 --data_seed 3407 \
                --bf16 True --data_path ./data --do_rope True \
                --config_name /home/zhichaoyang/mimic3/clinical-mamba/saved_models/clinicalllama-pretrain03 \
                --tokenizer_name /home/zhichaoyang/mimic3/clinical-mamba/saved_models/clinicalllama-pretrain03 \
                --model_name_or_path /home/zhichaoyang/mimic3/clinical-mamba/saved_models/clinicalllama-pretrain03 \
                --do_train --do_eval --max_seq_length 15004 \
                --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --per_device_eval_batch_size 1 \
                --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
                --learning_rate 1e-4 --weight_decay 1e-2 --num_train_epochs 12 \
                --lr_scheduler_type linear --warmup_ratio 0.15 \
                --logging_steps 50 \
                --evaluation_strategy epoch --save_strategy epoch \
                --logging_first_step \
                --output_dir ./saved_models/clinicalllama-test02_run02

WANDB_PROJECT=ClinicalTrial  CUDA_VISIBLE_DEVICES=0 python main_llama.py \
NCCL_P2P_DISABLE=1 WANDB_PROJECT=ClinicalMamba CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 39600 main_llama.py \
                --seed 36 --data_seed 36 \
                --bf16 True --data_path ./data --do_rope True \
                --config_name /home/zhichaoyang/mimic3/clinical-mamba/saved_models/clinicalllama-pretrain03 \
                --tokenizer_name /home/zhichaoyang/mimic3/clinical-mamba/saved_models/clinicalllama-pretrain03 \
                --model_name_or_path /home/zhichaoyang/mimic3/clinical-mamba/saved_models/clinicalllama-pretrain03 \
                --do_train --do_eval --max_seq_length 15004 \
                --per_device_train_batch_size 2 --gradient_accumulation_steps 1 --per_device_eval_batch_size 1 \
                --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
                --learning_rate 1e-4 --weight_decay 1e-2 --num_train_epochs 12 \
                --lr_scheduler_type linear --warmup_ratio 0.15 \
                --logging_steps 50 \
                --evaluation_strategy epoch --save_strategy epoch \
                --logging_first_step \
                --output_dir ./saved_models/clinicalllama-test02_run03



