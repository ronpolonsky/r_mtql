#!/bin/bash
set -euo pipefail
cd /iris/u/ronpo/projects/new_mtql

LAUNCHER=drawer_task/train.sh
PROJECT=offline-drawer-task-v1-s75-s100

submit_bc() {
  local tag="$1" successes="$2"
  sbatch --partition=iris-hi \
    --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=bc,SETTING_ID="$tag",TRAIN_SEED=1,TRAIN_STEPS=300000,NUM_SUCCESS_DEMOS="$successes",NUM_FAILURE_DEMOS=0,HIST_LENGTH=20,HIST_STRIDE=100,BC_BATCH_SIZE=8 \
    "$LAUNCHER"
}

submit_rl() {
  local tag="$1" successes="$2" failures="$3"
  sbatch --partition=iris-hi \
    --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=rl,MTQL_VARIANT=v2,SETTING_ID="$tag",TRAIN_SEED=1,TRAIN_STEPS=300000,NUM_SUCCESS_DEMOS="$successes",NUM_FAILURE_DEMOS="$failures",HIST_LENGTH=20,HIST_STRIDE=100,ALPHA=100,DISCOUNT=0.999,NORM_Q=true,RL_BATCH_SIZE=8 \
    "$LAUNCHER"
}

# Exactly six Iris-Hi jobs. Put the two matched BC controls and the four most
# promising masked-failure MTQL settings on the faster GPUs.
submit_bc bc_s75_h20_s100_seed1                       75
submit_bc bc_s100_h20_s100_seed1                     100
submit_rl rlv2_s75_f15_h20_s100_a100_g999_seed1      75  15
submit_rl rlv2_s75_f30_h20_s100_a100_g999_seed1      75  30
submit_rl rlv2_s100_f15_h20_s100_a100_g999_seed1     100 15
submit_rl rlv2_s100_f30_h20_s100_a100_g999_seed1     100 30
