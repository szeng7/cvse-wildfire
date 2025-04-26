#!/bin/bash

#conda init
#conda activate cvse

#parameters
MODEL="CAE"
DATA_DIR="./data"
BATCH_SIZE=128
CHECKPOINT_DIR="./checkpoints/CAE-weighted_BCE/step_3000/ckpt"

python eval.py \
--model ${MODEL} \
--data-dir ${DATA_DIR} \
--checkpoint-dir ${CHECKPOINT_DIR} \
--batch-size ${BATCH_SIZE} \
