#!/usr/bin/env bash

## verified with transformer v4.5.0 (env lt)
# +-------------------------------------------------+
#  Select model
# +-------------------------------------------------+
# MODEL_NAME="t5-base"
# ALIAS="t5_base"

# MODEL_NAME="t5-large"
# ALIAS="t5_large"

# MODEL_NAME="google/flan-t5-base"
# ALIAS="flan_t5_base"

MODEL_NAME="google/flan-t5-xl"
ALIAS="flan_t5_xl"

# MODEL_NAME="google/t5-v1_1-base"
# # MODEL_NAME="liangtaiwan/t5-v1_1-lm100k-base"
# ALIAS="t5v1_1_base"

# MODEL_NAME="facebook/bart-base"
# ALIAS="bart_base"

# MODEL_NAME="facebook/bart-large"
# ALIAS="bart_large"

# +-------------------------------------------------+
#  Dataset specific parameters
# +-------------------------------------------------+

DATASET="sbdh_gpt4_v2"
DATA_ALIAS="sbdh_gpt4_v2"
TRAIN_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_aio_BIO_train_v2.json'
DEV_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_aio_BIO_val_v2.json'
TEST_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_aio_BIO_test_v2.json'
MAX_LEN=256
PER_DEVICE_TRAIN_BATCH_SIZE=32

# +-------------------------------------------------+
#  Other parameters
# +-------------------------------------------------+
EPOCH=20 # 40 for base models, 20 for t5-xl
LEARNING_RATE=5e-5
SCHEDULER_TYPE="linear"
PER_DEVICE_EVAL_BATCH_SIZE=8 
NUM_BEAMS=1 # set it to 1 for greedy decoding
declare -a SEEDS=(0)
DATE=$(date +%b%d_%Y)

# +-------------------------------------------------+
#  COMMANDS
# +-------------------------------------------------+
N_GPU=4
GPU_IDS="0,1,2,3"
# for i in $(seq 1 $((N_GPU-1))); do GPU_IDS+=",${i}"; done
# COMMAND="accelerate launch"
COMMAND="python"
SCRIPT="run_ner_t5.py"
export CUDA_DEVICE_ORDER="PCI_BUS_ID" 

# +-------------------------------------------------+
#  Training
# +-------------------------------------------------+
if [ ${DATASET} == "bleeding" ] || [ ${DATASET} == "sbdh_gpt4" ] || [ ${DATASET} == "sbdh_gpt4_v2" ] || [ ${DATASET} == "sbdh_gpt4_v2+" ] || [ ${DATASET} == "sbdh_gpt4_hr" ] || [ ${DATASET} == "sbdh_gpt4_msf" ] || [ ${DATASET} == "sbdh_gpt4_msf_v3" ]
then
    for SEED in "${SEEDS[@]}"
    do
        echo "######### Dataset:" ${DATASET} "Seed:" ${SEED} "Model:" ${ALIAS} " #########"
        CUDA_VISIBLE_DEVICES=0 ${COMMAND} ${SCRIPT} \
            --model_name_or_path ${MODEL_NAME} \
            --logfile ./logs/${DATE}_${ALIAS}_${DATA_ALIAS}_noPtr.log \
            --train_file ${TRAIN_DATA} \
            --validation_file ${DEV_DATA} \
            --test_file ${TEST_DATA} \
            --max_source_length ${MAX_LEN} \
            --max_target_length ${MAX_LEN} \
            --num_train_epochs ${EPOCH} \
            --learning_rate ${LEARNING_RATE} \
            --num_beams ${NUM_BEAMS} \
            --seed ${SEED} \
            --lr_scheduler_type ${SCHEDULER_TYPE} \
            --warmup_proportion 0.0 \
            --do_train \
            --do_test \
            --use_accelerate \
            --pred_file ./output/${ALIAS}_preds_${DATA_ALIAS}_${SEED}_noPtr.txt \
            --result_file ./result/${ALIAS}_result_${DATA_ALIAS}_${SEED}_noPtr.txt \
            --best_result_file ./best_result/${ALIAS}_best_result_${DATA_ALIAS}_${SEED}_noPtr.txt \
            --output_dir ./saved_models/${ALIAS}_${DATA_ALIAS}_${SEED}_noPtr \
            --per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
            --per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE} 2>&1 | tee stdout.txt
        break
    done
else
    for SEED in "${SEEDS[@]}"
    do
        echo "######### Dataset:" ${DATASET} "Seed:" ${SEED} "Model:" ${ALIAS} " #########"
        CUDA_VISIBLE_DEVICES=3 ${COMMAND} ${SCRIPT} \
            --model_name_or_path ${MODEL_NAME} \
            --logfile ./logs/${DATE}_${ALIAS}_${DATA_ALIAS}_noPtr.log \
            --dataset_name ${DATASET} \
            --max_source_length ${MAX_LEN} \
            --max_target_length ${MAX_LEN} \
            --num_train_epochs ${EPOCH} \
            --learning_rate ${LEARNING_RATE} \
            --num_beams ${NUM_BEAMS} \
            --seed ${SEED} \
            --lr_scheduler_type ${SCHEDULER_TYPE} \
            --warmup_proportion 0.0 \
            --do_train \
            --do_test \
            --use_accelerate \
            --pred_file ./output/${ALIAS}_preds_${DATA_ALIAS}_${SEED}_noPtr.txt \
            --result_file ./result/${ALIAS}_result_${DATA_ALIAS}_${SEED}_noPtr.txt \
            --best_result_file ./best_result/${ALIAS}_best_result_${DATA_ALIAS}_${SEED}_noPtr.txt \
            --output_dir ./saved_models/${ALIAS}_${DATA_ALIAS}_${SEED}_noPtr \
            --per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
            --per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE} 2>&1 | tee stdout.txt
    done
fi