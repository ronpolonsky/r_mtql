#!/bin/bash
set -euo pipefail
cd /iris/u/ronpo/projects/new_mtql

LAUNCHER=drawer_task/train.sh
PROJECT=offline-drawer-task-v1-s150-s200

submit_rl() {
  local tag="$1" successes="$2" failures="$3" hist_length="$4"
  local hist_stride="$5" alpha="$6"
  sbatch --partition=iris \
    --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=rl,MTQL_VARIANT=v2,SETTING_ID="$tag",TRAIN_SEED=1,TRAIN_STEPS=300000,NUM_SUCCESS_DEMOS="$successes",NUM_FAILURE_DEMOS="$failures",HIST_LENGTH="$hist_length",HIST_STRIDE="$hist_stride",ALPHA="$alpha",DISCOUNT=0.999,NORM_Q=true,RL_BATCH_SIZE=4 \
    "$LAUNCHER"
}

# Exactly ten regular-Iris jobs. At F30/alpha=100, each success count gets
# H16/S125, H20/S100, and H24/S84; the matching H32/S64 cases are on Iris-Hi.
# This is the main temporal-resolution sweep. H32 is excluded here because it
# previously OOMed on regular Iris at batch size 4.
for successes in 150 200; do
  submit_rl "rlv2_s${successes}_f30_h16_s125_a100_g999_seed1" "$successes" 30 16 125 100
  submit_rl "rlv2_s${successes}_f30_h20_s100_a100_g999_seed1" "$successes" 30 20 100 100
  submit_rl "rlv2_s${successes}_f30_h24_s84_a100_g999_seed1"  "$successes" 30 24 84  100
done

# Failure-count controls at the H20/S100 anchor. Matching F50/H32 runs are on
# Iris-Hi, allowing history and failure count to be interpreted separately.
submit_rl rlv2_s150_f50_h20_s100_a100_g999_seed1 150 50 20 100 100
submit_rl rlv2_s200_f50_h20_s100_a100_g999_seed1 200 50 20 100 100

# One higher-distillation control per success count.
submit_rl rlv2_s150_f30_h20_s100_a300_g999_seed1 150 30 20 100 300
submit_rl rlv2_s200_f30_h20_s100_a300_g999_seed1 200 30 20 100 300
