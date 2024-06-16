#!/usr/bin/env bash

## verified with transformer v4.5.0 (env lt)
# +-------------------------------------------------+
#  Select model
# +-------------------------------------------------+
MODEL_NAME="roberta-base"
ALIAS="roberta_base"

# MODEL_NAME="/home/avijit/playground/sdoh/RoBERTa-base-PM-M3-Voc-distill-align"
# ALIAS="cliroberta_base"

# +-------------------------------------------------+
#  Dataset specific parameters
# +-------------------------------------------------+
DATASET="sbdh_gpt4_v2"
DATA_ALIAS="sbdh_gpt4_v2"
TRAIN_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_aio_BIO_train_v2.json'
DEV_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_aio_BIO_val_v2.json'
TEST_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_aio_BIO_test_v2.json'
MAX_LEN=256
PER_DEVICE_TRAIN_BATCH_SIZE=64
LEARNING_RATE=1e-5

# +-------------------------------------------------+
#  Other parameters
# +-------------------------------------------------+
EPOCH=40
PER_DEVICE_EVAL_BATCH_SIZE=8 
TASK_NAME="ner"
declare -a SEEDS=(0 1 2)
DATE=$(date +%b%d_%Y)
# N_GPU=3
# GPU_IDS="0,2"
# for i in $(seq 1 $((N_GPU-1))); do GPU_IDS+=",${i}"; done


for SEED in "${SEEDS[@]}"
do
  echo "######### Dataset:" ${DATASET} "Seed:" ${SEED} " #########"
  CUDA_VISIBLE_DEVICES=1 python run_ner.py \
    --model_name_or_path ${MODEL_NAME} \
    --train_file ${TRAIN_DATA} \
    --validation_file ${DEV_DATA} \
    --test_file ${TEST_DATA} \
    --logfile ./logs/${DATE}_${ALIAS}_${DATA_ALIAS}_encOnly_1e5.log \
    --task_name ${TASK_NAME} \
    --seed ${SEED} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCH} \
    --max_length ${MAX_LEN} \
    --lr_scheduler_type linear \
    --best_result_file ./best_result/${ALIAS}_best_result_${DATA_ALIAS}_${SEED}_encOnly_1e5.txt \
    --output_dir ./saved_models/${ALIAS}_${DATA_ALIAS}_${SEED} 2>&1 | tee stdout_rob.txt
    # --use_accelerator \
done