#!/bin/bash
set -euo pipefail
cd /iris/u/ronpo/projects/new_mtql

LAUNCHER=drawer_task/train.sh
PROJECT=offline-drawer-task-v1

sbatch --partition=iris-hi --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=rl,SETTING_ID=rl_h20_s100_a300,HIST_LENGTH=20,HIST_STRIDE=100,ALPHA=300,DISCOUNT=0.999,NORM_Q=true "$LAUNCHER"
sbatch --partition=iris-hi --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=rl,SETTING_ID=rl_h24_s80_a300,HIST_LENGTH=24,HIST_STRIDE=80,ALPHA=300,DISCOUNT=0.999,NORM_Q=true "$LAUNCHER"
sbatch --partition=iris-hi --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=rl,SETTING_ID=rl_h32_s64_a300,HIST_LENGTH=32,HIST_STRIDE=64,ALPHA=300,DISCOUNT=0.999,NORM_Q=true "$LAUNCHER"
sbatch --partition=iris-hi --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=bc,SETTING_ID=bc_h20_s100,HIST_LENGTH=20,HIST_STRIDE=100 "$LAUNCHER"
sbatch --partition=iris-hi --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=bc,SETTING_ID=bc_h24_s80,HIST_LENGTH=24,HIST_STRIDE=80 "$LAUNCHER"
sbatch --partition=iris-hi --export=ALL,WANDB_PROJECT="$PROJECT",RUN_KIND=bc,SETTING_ID=bc_h32_s64,HIST_LENGTH=32,HIST_STRIDE=64 "$LAUNCHER"
