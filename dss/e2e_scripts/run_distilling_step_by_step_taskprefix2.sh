#!/usr/bin/env bash

# +-------------------------------------------------+
#  Select model
# +-------------------------------------------------+
# MODEL_NAME="t5-base"
# ALIAS="t5_base"

# MODEL_NAME="t5-large"
# ALIAS="t5_large"

MODEL_NAME="google/flan-t5-base"
ALIAS="flan_t5_base"

# MODEL_NAME="google/flan-t5-base-ptr"
# ALIAS="flan_t5_base_ptr"

# MODEL_NAME="google/t5-v1_1-base"
# ALIAS="t5v1_1_base"

# MODEL_NAME="google/t5-v1_1-base-ptr"
# ALIAS="t5v1_1_base_ptr"

# MODEL_NAME="facebook/bart-base"
# ALIAS="bart_base"

# MODEL_NAME="facebook/bart-large"
# ALIAS="bart_large"

# +-------------------------------------------------+
#  Select dataset 
# +-------------------------------------------------+
# DATASET="svamp"
# MAX_LEN=512
# BATCH_SIZE=16

# DATASET="sbdh_gpt4_v2"
# MAX_LEN=256
# BATCH_SIZE=32

DATASET="sbdh_gpt4_v3"
MAX_LEN=256
BATCH_SIZE=32

# DATASET="sbdh_gpt4_msf"
# MAX_LEN=256
# BATCH_SIZE=32

# DATASET="sbdh_gpt4_msf_v3"
# MAX_LEN=256
# BATCH_SIZE=32

cd ../

declare -a SEEDS=(0 1 2)
TYPE="task_prefix"

for SEED in "${SEEDS[@]}"
    do
        echo "######### Dataset:" ${DATASET} "Seed:" ${SEED} "Model:" ${MODEL_NAME} #########"
        # WANDB_MODE=disabled
        WANDB_PROJECT=distilling_Step_by_step CUDA_VISIBLE_DEVICES=1 python run.py \
        --from_pretrained ${MODEL_NAME} \
        --dataset ${DATASET} \
        --model_type ${TYPE} \
        --label_type gt \
        --llm palm \
        --alpha 0.5 \
        --do_train \
        --do_predict \
        --batch_size ${BATCH_SIZE} \
        --num_train_epochs 100 \
        --grad_steps 2 \
        --generation_num_beams 1 \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --logging_strategy epoch \
        --load_best_model_at_end \
        --metric_for_best_model eval_macro-f \
        --greater_is_better \
        --save_total_limit 2 \
        --result_file ./results/${ALIAS}_result_${DATASET}_${SEED}_${TYPE}.txt \
        --max_input_length ${MAX_LEN} \
        --generation_max_length ${MAX_LEN} \
        --run ${SEED} 2>&1 | tee ./stdout/stdout_${TYPE}_${SEED}_.txt
    done

#### Inference only ####

# for SEED in "${SEEDS[@]}"
#     do
#         echo "######### Dataset:" ${DATASET} "Seed:" ${SEED} "Model:" ${MODEL_NAME} #########"
#         WANDB_MODE=disabled WANDB_PROJECT=distilling_Step_by_step CUDA_VISIBLE_DEVICES=1 python run.py \
#         --from_pretrained ${MODEL_NAME} \
#         --dataset ${DATASET} \
#         --model_type ${TYPE} \
#         --label_type gt \
#         --llm palm \
#         --alpha 0.5 \
#         --do_predict \
#         --batch_size ${BATCH_SIZE} \
#         --num_train_epochs 100 \
#         --grad_steps 2 \
#         --generation_num_beams 1 \
#         --evaluation_strategy epoch \
#         --save_strategy epoch \
#         --logging_strategy epoch \
#         --load_best_model_at_end \
#         --metric_for_best_model eval_macro-f \
#         --greater_is_better \
#         --save_total_limit 2 \
#         --result_file ./results/${ALIAS}_result_${DATASET}_${SEED}_${TYPE}_inf.txt \
#         --max_input_length ${MAX_LEN} \
#         --generation_max_length ${MAX_LEN} \
#         --run ${SEED} 2>&1 | tee ./stdout/stdout_${TYPE}_${SEED}_inf.txt
#     done