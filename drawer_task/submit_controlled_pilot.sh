#!/bin/bash
set -euo pipefail
cd /iris/u/ronpo/projects/new_mtql

LAUNCHER=drawer_task/train.sh
PROJECT=offline-drawer-task-v1-controlled
SEEDS="${SEEDS:-1 2 3}"

# A small 300k-step controlled pilot: BC and MTQL see the same 25 successful episodes;
# only MTQL additionally sees 15 deterministically balanced failed episodes. RL uses a
# smaller batch because its actor+critic update does not fit on 24 GB at batch 32.
for seed in $SEEDS; do
  sbatch --partition=iris-hi \
    --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=bc,SETTING_ID="bc_s25_h24_s80_seed${seed}",TRAIN_SEED="$seed",TRAIN_STEPS=300000,NUM_SUCCESS_DEMOS=25,NUM_FAILURE_DEMOS=0,HIST_LENGTH=24,HIST_STRIDE=80,BC_BATCH_SIZE=8 \
    "$LAUNCHER"

  sbatch --partition=iris-hi \
    --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=rl,SETTING_ID="rl_s25_f15_h24_s80_a300_seed${seed}",TRAIN_SEED="$seed",TRAIN_STEPS=300000,NUM_SUCCESS_DEMOS=25,NUM_FAILURE_DEMOS=15,HIST_LENGTH=24,HIST_STRIDE=80,ALPHA=300,DISCOUNT=0.999,NORM_Q=true,RL_BATCH_SIZE=8 \
    "$LAUNCHER"
done
