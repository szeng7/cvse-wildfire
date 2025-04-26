#!/bin/bash

#conda init
#conda activate cvse

#parameters
MODEL="NDWS_CAE"
DATA_DIR="./data"
BATCH_SIZE=128
CHECKPOINT_DIR="./checkpoints/NDWS_CAE-weighted_BCE/step_0250/ckpt"

python eval.py \
--model ${MODEL} \
--data-dir ${DATA_DIR} \
--checkpoint-dir ${CHECKPOINT_DIR} \
--batch-size ${BATCH_SIZE} \
