#!/bin/sh
#
# baseline model 70371

cd ../

export CUDA_DEVICE_ORDER="PCI_BUS_ID" 

# # nohup bash run01.sh > run01.out 2> run01.err &
WANDB_MODE=disabled  CUDA_VISIBLE_DEVICES=4 python main_longformer.py \
                --seed 42 --data_seed 42 \
                --data_path /home/avijit/playground/clinical_mamba/data/Heart_Disease_2014 \
                --config_name yikuan8/Clinical-Longformer \
                --tokenizer_name yikuan8/Clinical-Longformer \
                --model_name_or_path yikuan8/Clinical-Longformer \
                --do_train --do_eval --max_seq_length 4096 \
                --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --per_device_eval_batch_size 1 \
                --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
                --learning_rate 6.304615009e-6 --weight_decay 1e-1 --num_train_epochs 12 \
                --lr_scheduler_type linear --warmup_ratio 0.1 \
                --logging_steps 50 \
                --evaluation_strategy epoch --save_strategy no \
                --logging_first_step \
                --output_dir ./saved_models/clinicallongformer_test01 2>&1 | tee /home/avijit/playground/clinical_mamba/script/run_clilongfrmr_run1.txt

WANDB_MODE=disabled  CUDA_VISIBLE_DEVICES=4 python main_longformer.py \
                --seed 36 --data_seed 36 \
                --data_path /home/avijit/playground/clinical_mamba/data/Heart_Disease_2014 \
                --config_name yikuan8/Clinical-Longformer \
                --tokenizer_name yikuan8/Clinical-Longformer \
                --model_name_or_path yikuan8/Clinical-Longformer \
                --do_train --do_eval --max_seq_length 4096 \
                --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --per_device_eval_batch_size 1 \
                --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
                --learning_rate 6.304615009e-6 --weight_decay 1e-1 --num_train_epochs 12 \
                --lr_scheduler_type linear --warmup_ratio 0.1 \
                --logging_steps 50 \
                --evaluation_strategy epoch --save_strategy no \
                --logging_first_step \
                --output_dir ./saved_models/clinicallongfomer-test02 2>&1 | tee /home/avijit/playground/clinical_mamba/script/run_clilongfrmr_run2.txt

WANDB_MODE=disabled  CUDA_VISIBLE_DEVICES=4 python main_longformer.py \
                --seed 3407 --data_seed 3407 \
                --data_path /home/avijit/playground/clinical_mamba/data/Heart_Disease_2014 \
                --config_name yikuan8/Clinical-Longformer \
                --tokenizer_name yikuan8/Clinical-Longformer \
                --model_name_or_path yikuan8/Clinical-Longformer \
                --do_train --do_eval --max_seq_length 4096 \
                --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --per_device_eval_batch_size 1 \
                --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
                --learning_rate 6.304615009e-6 --weight_decay 1e-1 --num_train_epochs 12 \
                --lr_scheduler_type linear --warmup_ratio 0.1 \
                --logging_steps 50 \
                --evaluation_strategy epoch --save_strategy no \
                --logging_first_step \
                --output_dir ./saved_models/clinicallongfomer-test03 2>&1 | tee /home/avijit/playground/clinical_mamba/script/run_clilongfrmr_run3.txt

# WANDB_MODE=disabled  CUDA_VISIBLE_DEVICES=3 python main_llama.py \
#                 --seed 3407 --data_seed 36 \
#                 --bf16 True --data_path ./data \
#                 --config_name /home/zhichaoyang/mimic3/clinical-mamba/saved_models/clinicalllama-pretrain02 \
#                 --tokenizer_name /home/zhichaoyang/mimic3/clinical-mamba/saved_models/clinicalllama-pretrain02 \
#                 --model_name_or_path /home/zhichaoyang/mimic3/clinical-mamba/saved_models/clinicalllama-pretrain02 \
#                 --do_train --do_eval --max_seq_length 4096 \
#                 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --per_device_eval_batch_size 1 \
#                 --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
#                 --learning_rate 6.304615009e-6 --weight_decay 1e-1 --num_train_epochs 12 \
#                 --lr_scheduler_type linear --warmup_ratio 0.1 \
#                 --logging_steps 50 \
#                 --evaluation_strategy epoch --save_strategy no \
#                 --logging_first_step \
#                 --output_dir ./saved_models/clinicallongfomer-test03 2>&1 | tee /home/avijit/playground/clinical_mamba/script/run_clillama_run3.txt



