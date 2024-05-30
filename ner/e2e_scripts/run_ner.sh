#!/usr/bin/env bash

## verified with transformer v4.5.0 (env lt)
# +-------------------------------------------------+
#  Select model
# +-------------------------------------------------+
# MODEL_NAME="bert-base-cased"
# ALIAS="bert_base"

MODEL_NAME="roberta-base"
ALIAS="roberta_base"

# MODEL_NAME="/home/avijit/playground/sdoh/RoBERTa-base-PM-M3-Voc-distill-align"
# ALIAS="cliroberta_base"

# MODEL_NAME="gpt2"
# ALIAS="gpt2"

# MODEL_NAME="michiyasunaga/BioLinkBERT-base"
# ALIAS="biolinkbert_base"
# +-------------------------------------------------+
#  Dataset specific parameters
# +-------------------------------------------------+
# DATASET="conll2003"
# DATA_ALIAS="conll2003"
# TRAIN_DATA='/home/avijit/playground/generatve_ner/EntLM/dataset/conll2003/full/train.json'
# DEV_DATA='/home/avijit/playground/generatve_ner/EntLM/dataset/conll2003/full/validation.json'
# TEST_DATA='/home/avijit/playground/generatve_ner/EntLM/dataset/conll2003/full/test.json'
# MAX_LEN=256
# PER_DEVICE_TRAIN_BATCH_SIZE=64
# LEARNING_RATE=5e-5

# DATASET="wnut_17"
# DATA_ALIAS="wnut_17"
# MAX_LEN=256
# PER_DEVICE_TRAIN_BATCH_SIZE=64
# LEARNING_RATE=1e-5

# DATASET="tner/mit_restaurant"
# DATA_ALIAS="mit_restaurant"
# MAX_LEN=256
# PER_DEVICE_TRAIN_BATCH_SIZE=64
# LEARNING_RATE=1e-5

# DATASET="tner/mit_movie_trivia"
# DATA_ALIAS="mit_movie"
# MAX_LEN=256
# PER_DEVICE_TRAIN_BATCH_SIZE=64
# LEARNING_RATE=5e-5

# DATASET="bleeding"
# DATA_ALIAS="bleeding"
# TRAIN_DATA='/home/avijit/BERT-NER/data/ehr_BIO_train_2021.json'
# DEV_DATA='/home/avijit/BERT-NER/data/ehr_BIO_dev_2021.json'
# TEST_DATA='/home/avijit/BERT-NER/data/ehr_BIO_test_2021.json'
# MAX_LEN=512
# PER_DEVICE_TRAIN_BATCH_SIZE=24
# LEARNING_RATE=1e-5

# DATASET="sbdh_gpt4"
# DATA_ALIAS="sbdh_gpt4"
# TRAIN_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_aio_BIO_train.json'
# DEV_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_aio_BIO_val.json'
# TEST_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_aio_BIO_test.json'
# MAX_LEN=256
# PER_DEVICE_TRAIN_BATCH_SIZE=64
# LEARNING_RATE=1e-5

# DATASET="sbdh_gpt4_v2"
# DATA_ALIAS="sbdh_gpt4_v2"
# TRAIN_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_aio_BIO_train_v2.json'
# DEV_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_aio_BIO_val_v2.json'
# TEST_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_aio_BIO_test_v2.json'
# MAX_LEN=256
# PER_DEVICE_TRAIN_BATCH_SIZE=64
# LEARNING_RATE=1e-5

DATASET="sbdh_gpt4_v2+"
DATA_ALIAS="sbdh_gpt4_v2+"
TRAIN_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_aio_BIO_train&test_v2.json'
DEV_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_aio_BIO_val_v2.json'
TEST_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_aio_BIO_val_v2.json'
MAX_LEN=256
PER_DEVICE_TRAIN_BATCH_SIZE=64
LEARNING_RATE=1e-5

# DATASET="sbdh_gpt4_msf"
# DATA_ALIAS="sbdh_gpt4_msf"
# TRAIN_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_msf_BIO_train_v2.json'
# DEV_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_msf_BIO_val_v2.json'
# TEST_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_msf_BIO_test_v2.json'
# MAX_LEN=256
# PER_DEVICE_TRAIN_BATCH_SIZE=64
# LEARNING_RATE=1e-5

# DATASET="sbdh_gpt4_msf_v3"
# DATA_ALIAS="sbdh_gpt4_msf_v3"
# TRAIN_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_msf_BIO_train_v3.json'
# DEV_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_msf_BIO_val_v3.json'
# TEST_DATA='/home/avijit/playground/sdoh/synth_data_gpt4/synth_data_msf_BIO_test_v3.json'
# MAX_LEN=256
# PER_DEVICE_TRAIN_BATCH_SIZE=64
# LEARNING_RATE=1e-5

# +-------------------------------------------------+
#  Other parameters
# +-------------------------------------------------+
EPOCH=40
PER_DEVICE_EVAL_BATCH_SIZE=8 
TASK_NAME="ner"
declare -a SEEDS=(0 1 2)
DATE=$(date +%b%d_%Y)
N_GPU=3
GPU_IDS="0,2"
# for i in $(seq 1 $((N_GPU-1))); do GPU_IDS+=",${i}"; done

if [ ${DATASET} == "bleeding" ] || [ ${DATASET} == "sbdh_gpt4" ] || [ ${DATASET} == "sbdh_gpt4_v2" ] || [ ${DATASET} == "sbdh_gpt4_v2+" ] || [ ${DATASET} == "sbdh_gpt4_msf" ] || [ ${DATASET} == "sbdh_gpt4_msf_v3" ]
then
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
else
  for SEED in "${SEEDS[@]}"
  do
    echo "######### Dataset:" ${DATASET} "Seed:" ${SEED} " #########"
    CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node 1 --use_env run_ner.py \
      --model_name_or_path ${MODEL_NAME} \
      --dataset_name ${DATASET} \
      --logfile ./logs/${DATE}_${ALIAS}_${DATA_ALIAS}_encOnly_1e5.log \
      --task_name ${TASK_NAME} \
      --seed ${SEED} \
      --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
      --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
      --learning_rate ${LEARNING_RATE} \
      --num_train_epochs ${EPOCH} \
      --max_length ${MAX_LEN} \
      --lr_scheduler_type linear \
      --best_result_file ./best_result/${ALIAS}_best_result_${DATA_ALIAS}_${SEED}_encOnly_1e5.txt \
      --output_dir ./saved_models/${ALIAS}_${DATA_ALIAS}_${SEED} 2>&1 | tee stdout_.txt
      # --use_accelerator \
      # --do_aux_task
  done
fi