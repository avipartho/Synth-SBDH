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
# MAX_LEN=128
# BATCH_SIZE=32
# NUM_EPOCH=40

# DATASET="sbdh_gpt4_msf"
# DATA_PATH="/home/avijit/playground/sdoh/synth_data_gpt4/sbdh_gpt4_msf_multilabel"
# MAX_LEN=128
# BATCH_SIZE=32
# NUM_EPOCH=40

# +-------------------------------------------------+
#  Select pretrained model
# +-------------------------------------------------+
# MODEL_PATH="./saved_models/roberta_0/checkpoint-514"

# MODEL_PATH="./saved_models/roberta_sbdh_gpt4_v2_0/checkpoint-792"

MODEL_PATH="./saved_models/roberta_sbdh_gpt4_msf_v3_0/checkpoint-561"

cd ../

export CUDA_DEVICE_ORDER="PCI_BUS_ID" 
declare -a SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"
    do
    echo "######### Seed:" ${SEED} "Dataset:" ${DATASET} "#########"
    # WANDB_MODE=disabled 
    CUDA_VISIBLE_DEVICES=1 python main_roberta.py \
                    --seed ${SEED} --data_seed ${SEED} \
                    --data_path ${DATA_PATH} \
                    --config_name FacebookAI/roberta-base \
                    --tokenizer_name FacebookAI/roberta-base \
                    --model_name_or_path ${MODEL_PATH} \
                    --do_train --do_eval --do_predict --max_seq_length ${MAX_LEN} \
                    --per_device_train_batch_size ${BATCH_SIZE} --gradient_accumulation_steps 8 --per_device_eval_batch_size 8 \
                    --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
                    --learning_rate 1e-5 --weight_decay 1e-1 --num_train_epochs ${NUM_EPOCH} \
                    --lr_scheduler_type linear --warmup_ratio 0.1 \
                    --evaluation_strategy epoch --save_strategy epoch --logging_strategy epoch  \
                    --overwrite_output_dir True \
                    --load_best_model_at_end --metric_for_best_model eval_f1_macro --greater_is_better True --save_total_limit 2 \
                    --run_name roberta_frm_sbdh_gpt4_msf_v3_ml_${DATASET}_${SEED}\
                    --output_dir ./saved_models/roberta_frm_sbdh_gpt4_msf_v3_${DATASET}_${SEED} 2>&1 | tee /home/avijit/playground/sdoh/stdout/stdout_roberta_frm_sbdh_gpt4_msf_v3_${DATASET}_${SEED}.txt
    done

# for SEED in "${SEEDS[@]}"
#     do
#     echo "######### Seed:" ${SEED} "#########"
#     WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=2 python main_roberta.py \
#                     --seed ${SEED} --data_seed ${SEED} \
#                     --data_path ${DATA_PATH} \
#                     --config_name FacebookAI/roberta-base \
#                     --tokenizer_name FacebookAI/roberta-base \
#                     --model_name_or_path FacebookAI/roberta-base \
#                     --do_predict --max_seq_length ${MAX_LEN} \
#                     --per_device_train_batch_size ${BATCH_SIZE} --gradient_accumulation_steps 8 --per_device_eval_batch_size 8 \
#                     --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-5 \
#                     --learning_rate 1e-5 --weight_decay 1e-1 --num_train_epochs ${NUM_EPOCH} \
#                     --lr_scheduler_type linear --warmup_ratio 0.1 \
#                     --evaluation_strategy epoch --save_strategy epoch --logging_strategy epoch  \
#                     --overwrite_output_dir True \
#                     --load_best_model_at_end --metric_for_best_model eval_f1_macro --greater_is_better True --save_total_limit 2 \
#                     --run_name roberta_${DATASET}_ml_${SEED}\
#                     --output_dir ./saved_models/roberta_${DATASET}_${SEED} 2>&1 | tee /home/avijit/playground/sdoh/stdout/stdout_roberta_${SEED}_inf.txt
#     done