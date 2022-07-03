#!/usr/bin/env bash

# +--------------------------------------+
#  Models 
# +--------------------------------------+
# MODEL_NAME="biobert_base"
# MODEL_PATH="/data/python_envs/anaconda3/envs/transformers_cache/biobert_v1.1_pubmed/"

MODEL_NAME="roberta_base"
# MODEL_PATH="/data/python_envs/anaconda3/envs/transformers_cache/roberta-base/"

# +--------------------------------------+
#  Hyper-parameters 
# +--------------------------------------+
NUM_EPOCHS=20
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=8
DATE=$(date +%b%d)
PATIENCE=${NUM_EPOCHS} # set to NUM_EPOCHS to run training for all epochs

# +--------------------------------------+
#  Training & Evaluation(Exact)
# +--------------------------------------+
EVAL_MODE="exact" # exact or relaxed

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_ner.py \
    --train_data_dir datasets/train.txt \
    --dev_data_dir datasets/dev.txt \
    --test_data_dir datasets/test.txt \
    --bert_model ${MODEL_NAME} \
    --output_dir output/out_${MODEL_NAME}_sdoh_${DATE} \
    --max_seq_length 512 \
    --task_name ner \
    --seed 0 \
    --do_train \
    --num_train_epochs ${NUM_EPOCHS} \
	--train_batch_size ${TRAIN_BATCH_SIZE} \
	--eval_batch_size ${EVAL_BATCH_SIZE} \
    --patience ${PATIENCE} \
    --do_eval \
    --logfile logs/output_${MODEL_NAME}_sdoh_${DATE}.log \
    --eval_criteria ${EVAL_MODE}

# +--------------------------------------+
#  Evaluation(Relaxed) 
# +--------------------------------------+
EVAL_MODE="relaxed" # exact or relaxed

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_ner.py \
    --train_data_dir datasets/train.txt \
    --dev_data_dir datasets/dev.txt \
    --test_data_dir datasets/test.txt \
    --bert_model ${MODEL_NAME} \
    --output_dir output/out_${MODEL_NAME}_sdoh_${DATE} \
    --max_seq_length 512 \
    --task_name ner \
	--train_batch_size ${TRAIN_BATCH_SIZE} \
	--eval_batch_size ${EVAL_BATCH_SIZE} \
    --do_eval \
    --logfile logs/output_${MODEL_NAME}_sdoh_${DATE}.log \
    --eval_criteria ${EVAL_MODE}
