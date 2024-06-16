#!/bin/sh
# +-------------------------------------------------+
#  Select dataset 
# +-------------------------------------------------+
DATASET="mimic_sbdh"
DATA_PATH="/home/avijit/playground/sdoh/mimic-sbdh/mimic_sbdh"
MAX_LEN=256
BATCH_SIZE=32
NUM_EPOCH=8

# DATASET="sbdh_gpt4_v2"
# DATA_PATH="/home/avijit/playground/sdoh/synth_data_gpt4/sbdh_gpt4_v2_multilabel"
# MAX_LEN=256
# BATCH_SIZE=32
# NUM_EPOCH=40

# +-------------------------------------------------+
#  Select pretrained model
# +-------------------------------------------------+
MODEL_PATH="./saved_models/climamba_sbdh_gpt4_v2_0/checkpoint-264"

cd ../

export CUDA_DEVICE_ORDER="PCI_BUS_ID" 
declare -a SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"
    do
    echo "######### Seed:" ${SEED} "Dataset:" ${DATASET} "#########"
    # WANDB_MODE=disabled  
    CUDA_VISIBLE_DEVICES=2 python main_mamba.py \
                --seed ${SEED} --data_seed ${SEED} --ddp_find_unused_parameters False\
                --data_path ${DATA_PATH} \
                --config_name /data/data_user_alpha/public_models/state-spaces-mamba/clinicalmamba-130m \
                --tokenizer_name /data/data_user_alpha/public_models/state-spaces-mamba/clinicalmamba-130m \
                --model_name_or_path ${MODEL_PATH} \
                --do_train --do_eval --do_predict --max_seq_length ${MAX_LEN} \
                --per_device_train_batch_size ${BATCH_SIZE} --gradient_accumulation_steps 8 --per_device_eval_batch_size 8 \
                --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
                --learning_rate 3e-4 --weight_decay 1e-2 --num_train_epochs ${NUM_EPOCH} \
                --lr_scheduler_type constant --warmup_ratio 0.15 \
                --evaluation_strategy epoch --save_strategy epoch --logging_strategy epoch  \
                --overwrite_output_dir True \
                --load_best_model_at_end --metric_for_best_model eval_f1_macro --greater_is_better True --save_total_limit 2 \
                --run_name climamba_frm_sbdh_gpt4_v2_ml_${DATASET}_${SEED}\
                --output_dir ./saved_models/climamba_frm_sbdh_gpt4_v2_${DATASET}_${SEED}  2>&1 | tee /home/avijit/playground/sdoh/stdout/stdout_climamba_frm_sbdh_gpt4_v2_${DATASET}_${SEED}.txt
    done

# for SEED in "${SEEDS[@]}"
#     do
#     echo "######### Seed:" ${SEED} "Dataset:" ${DATASET} "#########"
#     WANDB_MODE=disabled  CUDA_VISIBLE_DEVICES=2 python main_mamba.py \
#                 --seed ${SEED} --data_seed ${SEED} --ddp_find_unused_parameters False\
#                 --data_path ${DATA_PATH} \
#                 --config_name /data/data_user_alpha/public_models/state-spaces-mamba/clinicalmamba-130m \
#                 --tokenizer_name /data/data_user_alpha/public_models/state-spaces-mamba/clinicalmamba-130m \
#                 --model_name_or_path /data/data_user_alpha/public_models/state-spaces-mamba/clinicalmamba-130m \
#                 --do_predict --max_seq_length ${MAX_LEN} \
#                 --per_device_train_batch_size ${BATCH_SIZE} --gradient_accumulation_steps 8 --per_device_eval_batch_size 8 \
#                 --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
#                 --learning_rate 3e-4 --weight_decay 1e-2 --num_train_epochs ${NUM_EPOCH} \
#                 --lr_scheduler_type constant --warmup_ratio 0.15 \
#                 --evaluation_strategy epoch --save_strategy epoch --logging_strategy epoch  \
#                 --overwrite_output_dir True \
#                 --load_best_model_at_end --metric_for_best_model eval_f1_macro --greater_is_better True --save_total_limit 2 \
#                 --run_name climamba_ml_${DATASET}_${SEED}\
#                 --output_dir ./saved_models/climamba_${DATASET}_${SEED}  2>&1 | tee /home/avijit/playground/sdoh/stdout/stdout_climamba_${DATASET}_${SEED}_inf.txt
#     done