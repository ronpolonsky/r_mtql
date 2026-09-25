#!/bin/bash
set -euo pipefail
cd /iris/u/ronpo/projects/new_mtql

LAUNCHER=drawer_task/train.sh
PROJECT=offline-drawer-task-v1-s75-s100

submit_rl() {
  local tag="$1" successes="$2" failures="$3" hist_length="$4"
  local hist_stride="$5" alpha="$6"
  sbatch --partition=iris \
    --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=rl,MTQL_VARIANT=v2,SETTING_ID="$tag",TRAIN_SEED=1,TRAIN_STEPS=300000,NUM_SUCCESS_DEMOS="$successes",NUM_FAILURE_DEMOS="$failures",HIST_LENGTH="$hist_length",HIST_STRIDE="$hist_stride",ALPHA="$alpha",DISCOUNT=0.999,NORM_Q=true,RL_BATCH_SIZE=4 \
    "$LAUNCHER"
}

# Exactly ten regular-Iris jobs. The first eight complete the alpha comparison
# around the four high-priority Iris-Hi settings. The final two test alternate
# full-horizon history sampling at the strongest predicted S100/F30 mixture.
for successes in 75 100; do
  for failures in 15 30; do
    for alpha in 30 300; do
      submit_rl \
        "rlv2_s${successes}_f${failures}_h20_s100_a${alpha}_g999_seed1" \
        "$successes" "$failures" 20 100 "$alpha"
    done
  done
done

submit_rl rlv2_s100_f30_h16_s125_a100_g999_seed1 100 30 16 125 100
submit_rl rlv2_s100_f30_h32_s64_a100_g999_seed1  100 30 32 64  100
