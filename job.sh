#!/bin/bash

#conda init
#conda activate cvse

#parameters
MODEL="NDWS_CAE"
DATA_DIR="./data"
STEPS=400
LOSS="dice"
BATCH_SIZE=128
LR=1e-5
SEED=19
AUGMENT=0
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
--augment ${AUGMENT} \
--checkpoint-dir ${CHECKPOINT_DIR}
