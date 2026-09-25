#!/bin/bash
set -euo pipefail
cd /iris/u/ronpo/projects/new_mtql

LAUNCHER=drawer_task/train.sh
PROJECT=offline-drawer-task-v1-s150-s200

submit_bc() {
  local tag="$1" successes="$2"
  sbatch --partition=iris-hi \
    --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=bc,SETTING_ID="$tag",TRAIN_SEED=1,TRAIN_STEPS=300000,NUM_SUCCESS_DEMOS="$successes",NUM_FAILURE_DEMOS=0,HIST_LENGTH=20,HIST_STRIDE=100,BC_BATCH_SIZE=8 \
    "$LAUNCHER"
}

submit_rl() {
  local tag="$1" successes="$2" failures="$3" hist_length="$4" hist_stride="$5"
  sbatch --partition=iris-hi \
    --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=rl,MTQL_VARIANT=v2,SETTING_ID="$tag",TRAIN_SEED=1,TRAIN_STEPS=300000,NUM_SUCCESS_DEMOS="$successes",NUM_FAILURE_DEMOS="$failures",HIST_LENGTH="$hist_length",HIST_STRIDE="$hist_stride",ALPHA=100,DISCOUNT=0.999,NORM_Q=true,RL_BATCH_SIZE=8 \
    "$LAUNCHER"
}

# Exactly six Iris-Hi jobs. Keep the BC anchors at H20/S100 and place every
# H32/S64 MTQL run on Iris-Hi because H32 OOMs on regular Iris.
submit_bc bc_s150_h20_s100_seed1                       150
submit_bc bc_s200_h20_s100_seed1                       200
submit_rl rlv2_s150_f30_h32_s64_a100_g999_seed1       150 30 32 64
submit_rl rlv2_s150_f50_h32_s64_a100_g999_seed1       150 50 32 64
submit_rl rlv2_s200_f30_h32_s64_a100_g999_seed1       200 30 32 64
submit_rl rlv2_s200_f50_h32_s64_a100_g999_seed1       200 50 32 64
