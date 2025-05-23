#!/bin/sh
# +-------------------------------------------------+
#  Select dataset 
# +-------------------------------------------------+
# DATASET="mimic_sbdh"
# DATA_PATH="./mimic-sbdh/mimic_sbdh"
# MAX_LEN=256
# BATCH_SIZE=32
# NUM_EPOCH=8

# DATASET="mimic_sbdh_50"
# DATA_PATH="./mimic-sbdh/mimic_sbdh"
# MAX_LEN=256
# BATCH_SIZE=32
# NUM_EPOCH=50

# DATASET="sbdh_gpt4_v2"
# DATA_PATH="./synth_data_gpt4/sbdh_gpt4_v2_multilabel"
# MAX_LEN=256
# BATCH_SIZE=32
# NUM_EPOCH=40

# DATASET="sbdh_gpt4_msf"
# DATA_PATH="./synth_data_gpt4/sbdh_gpt4_msf_multilabel"
# MAX_LEN=256
# BATCH_SIZE=32
# NUM_EPOCH=40

# DATASET="sbdh_gpt4_msf_v3"
# DATA_PATH="./synth_data_gpt4/sbdh_gpt4_msf_v3_multilabel"
# MAX_LEN=256
# BATCH_SIZE=32
# NUM_EPOCH=40

DATASET="sbdh_gpt4_hr"
DATA_PATH="./synth_data_gpt4/sbdh_gpt4_v3_multilabel_hr"
MAX_LEN=256
BATCH_SIZE=32
NUM_EPOCH=40

# DATASET="sbdh_gpt4_hr+" # this merges train and test set, uses val set as both val and test sets
# DATA_PATH="./synth_data_gpt4/sbdh_gpt4_v3_multilabel_hr+"
# MAX_LEN=256
# BATCH_SIZE=32
# NUM_EPOCH=40

cd ../

export CUDA_DEVICE_ORDER="PCI_BUS_ID" 
declare -a SEEDS=(0)

for SEED in "${SEEDS[@]}"
    do
    echo "######### Seed:" ${SEED} "Dataset:" ${DATASET} "#########"
    # WANDB_MODE=disabled  
    CUDA_VISIBLE_DEVICES=7 python main_mamba_prompt.py \
                --seed ${SEED} --data_seed ${SEED} --ddp_find_unused_parameters False\
                --data_path ${DATA_PATH} \
                --config_name /data/data_user_alpha/public_models/state-spaces-mamba/clinicalmamba-130m \
                --tokenizer_name /data/data_user_alpha/public_models/state-spaces-mamba/clinicalmamba-130m \
                --model_name_or_path /data/data_user_alpha/public_models/state-spaces-mamba/clinicalmamba-130m \
                --do_train --do_eval --do_predict --max_seq_length ${MAX_LEN} \
                --per_device_train_batch_size ${BATCH_SIZE} --gradient_accumulation_steps 8 --per_device_eval_batch_size 8 \
                --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
                --learning_rate 3e-4 --weight_decay 1e-2 --num_train_epochs ${NUM_EPOCH} \
                --lr_scheduler_type constant --warmup_ratio 0.15 \
                --evaluation_strategy epoch --save_strategy epoch --logging_strategy epoch  \
                --overwrite_output_dir True \
                --load_best_model_at_end --metric_for_best_model eval_f1_macro --greater_is_better True --save_total_limit 2 \
                --run_name climamba_prompt_ml_${DATASET}_${SEED}\
                --output_dir ./saved_models/climamba_prompt_${DATASET}_${SEED}  2>&1 | tee ./stdout/stdout_climamba_prompt_${DATASET}_${SEED}.txt
    done

# for SEED in "${SEEDS[@]}"
#     do
#     echo "######### Seed:" ${SEED} "Dataset:" ${DATASET} "#########"
#     WANDB_MODE=disabled  CUDA_VISIBLE_DEVICES=2 python main_mamba_prompt.py \
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
#                 --run_name climamba_prompt_ml_${DATASET}_${SEED}\
#                 --output_dir ./saved_models/climamba_prompt_${DATASET}_${SEED}  2>&1 | tee ./stdout/stdout_climamba_prompt_${DATASET}_${SEED}_inf.txt
#     done