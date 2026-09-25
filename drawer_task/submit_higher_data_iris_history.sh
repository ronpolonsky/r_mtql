#!/bin/bash
set -euo pipefail
cd /iris/u/ronpo/projects/new_mtql

LAUNCHER=drawer_task/train.sh
PROJECT=offline-drawer-task-v1-s75-s100

submit_rl() {
  local tag="$1" successes="$2" failures="$3" hist_length="$4" hist_stride="$5"
  sbatch --partition=iris \
    --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=rl,MTQL_VARIANT=v2,SETTING_ID="$tag",TRAIN_SEED=1,TRAIN_STEPS=300000,NUM_SUCCESS_DEMOS="$successes",NUM_FAILURE_DEMOS="$failures",HIST_LENGTH="$hist_length",HIST_STRIDE="$hist_stride",ALPHA=100,DISCOUNT=0.999,NORM_Q=true,RL_BATCH_SIZE=4 \
    "$LAUNCHER"
}

# Exactly ten additional regular-Iris jobs. Together with the H20/S100 anchor
# and the two S100/F30 history runs in submit_higher_data_iris.sh, these finish
# the most useful full-episode history-resolution comparisons at alpha=100.

# H16/S125 and H32/S64 for the three mixtures not already submitted.
for successes_failures in "75 15" "75 30" "100 15"; do
  read -r successes failures <<< "$successes_failures"
  submit_rl \
    "rlv2_s${successes}_f${failures}_h16_s125_a100_g999_seed1" \
    "$successes" "$failures" 16 125
  submit_rl \
    "rlv2_s${successes}_f${failures}_h32_s64_a100_g999_seed1" \
    "$successes" "$failures" 32 64
done

# H24/S84 (2,016-step span) for all four data mixtures.
for successes in 75 100; do
  for failures in 15 30; do
    submit_rl \
      "rlv2_s${successes}_f${failures}_h24_s84_a100_g999_seed1" \
      "$successes" "$failures" 24 84
  done
done
