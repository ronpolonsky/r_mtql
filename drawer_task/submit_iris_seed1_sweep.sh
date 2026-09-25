#!/bin/bash
set -euo pipefail
cd /iris/u/ronpo/projects/new_mtql

LAUNCHER=drawer_task/train.sh
PROJECT=offline-drawer-task-v1-controlled

submit_bc() {
  local tag="$1" successes="$2" hist_length="$3" hist_stride="$4"
  sbatch --partition=iris \
    --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=bc,SETTING_ID="$tag",TRAIN_SEED=1,TRAIN_STEPS=300000,NUM_SUCCESS_DEMOS="$successes",NUM_FAILURE_DEMOS=0,HIST_LENGTH="$hist_length",HIST_STRIDE="$hist_stride",BC_BATCH_SIZE=4 \
    "$LAUNCHER"
}

submit_rl() {
  local tag="$1" successes="$2" failures="$3" hist_length="$4"
  local hist_stride="$5" alpha="$6"
  sbatch --partition=iris \
    --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=rl,SETTING_ID="$tag",TRAIN_SEED=1,TRAIN_STEPS=300000,NUM_SUCCESS_DEMOS="$successes",NUM_FAILURE_DEMOS="$failures",HIST_LENGTH="$hist_length",HIST_STRIDE="$hist_stride",ALPHA="$alpha",DISCOUNT=0.999,NORM_Q=true,RL_BATCH_SIZE=4 \
    "$LAUNCHER"
}

# Exactly ten regular-Iris jobs. H20/S100 spans the full 2,000-step episode,
# so even terminal timeout samples can retain the initial target cue.
# BC and every RL run use the same deterministically selected 25 successes.
submit_bc bc_s25_h20_s100_seed1                    25 20 100

# Focused 3x3 MTQL grid. Five, ten, and fifteen failures are approximately
# 17%, 33%, and 41% of selected transitions. Higher failure counts are avoided
# because the 2,000-step timeout episodes would dominate the replay data.
# Lower alpha values let normalized Q guidance compete with the flow teacher;
# alpha=300 retains the established MTQL reference setting.
for failures in 5 10 15; do
  for alpha in 30 100 300; do
    submit_rl \
      "rl_s25_f${failures}_h20_s100_a${alpha}_g999_seed1" \
      25 "$failures" 20 100 "$alpha"
  done
done
