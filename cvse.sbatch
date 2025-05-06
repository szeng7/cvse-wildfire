#!/bin/bash
#SBATCH --array=0-1
#SBATCH --account=csci_ga_3033_109-2025sp
#SBATCH --partition=g2-standard-12
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=cvse_proj
#SBATCH --output=/scratch/xs2682/cvse/%A_%a_%x.out
#SBATCH --error=/scratch/xs2682/cvse/%A_%a_%x.err
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xs2682@nyu.edu

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

# Read line corresponding to current array task ID
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /scratch/xs2682/cvse/cvse-wildfire/configs.txt)
read MODEL DATA_DIR STEPS LOSS BATCH_SIZE LR SEED AUGMENT CHECKPOINT_DIR<<< $CONFIG

singularity exec $nv --overlay /scratch/xs2682/overlay-25GB-500K-tmp.ext3:ro /share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate cvse
cd /scratch/xs2682/cvse/cvse-wildfire
python train.py --model $MODEL --data-dir $DATA_DIR --num-steps $STEPS --loss $LOSS --batch-size $BATCH_SIZE --lr $LR --shuffle --seed $SEED --augment $AUGMENT --checkpoint-dir $CHECKPOINT_DIR
exit"