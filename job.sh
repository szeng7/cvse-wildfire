#!/bin/bash

#conda init
#conda activate cvse

#parameters
MODEL="NDWS_CAE"
DATA_DIR="./data"
STEPS=1500
LOSS="weighted_BCE"
BATCH_SIZE=128
LR=1e-5
SEED=19
CHECKPOINT_DIR="./checkpoints"

python train.py \
--model ${MODEL} \
--data-dir ${DATA_DIR} \
--num-steps ${STEPS} \
--loss ${LOSS} \
--batch-size ${BATCH_SIZE} \
--lr ${LR} \
--shuffle \
--seed ${SEED} \
--checkpoint-dir ${CHECKPOINT_DIR}
