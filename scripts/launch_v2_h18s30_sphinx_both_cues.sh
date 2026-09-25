#!/usr/bin/env bash
set -euo pipefail

cd /iris/u/ronpo/projects/new_mtql_candy_scooping

DATASET_PATH=/iris/u/ronpo/expo-ft-data/candy_scoop_v2_jitted
DATASET_CACHE=/iris/u/ronpo/mtql-runs/candy_scoop_v2_jitted_cache_v2norm
NORM_STATS_PATH=/iris/u/ronpo/projects/new_mtql_candy_scooping/norm_stats/candy_scoop_v2_jitted
NODELIST='sphinx[4-11]'

launch() {
  local name="$1"
  local project="$2"
  local cue_mode="$3"
  local agent="$4"
  local save_dir="$5"
  local train_script="$6"

  sbatch \
    --account=nlp \
    --partition=sphinx \
    --nodelist="$NODELIST" \
    --nodes=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=128G \
    --time=120:00:00 \
    --job-name="$name" \
    --export="ALL,DATASET_PATH=$DATASET_PATH,DATASET_CACHE=$DATASET_CACHE,NORM_STATS_PATH=$NORM_STATS_PATH,PROJECT=$project,CUE_MODE=$cue_mode,P_AUG=1.0,SEED=0,HIST_LENGTH=18,HIST_STRIDE=30,BATCH_SIZE=16,TRAIN_STEPS=1000000,LOG_INTERVAL=10000,SAVE_INTERVAL=10000,N_SUCC=-1,N_FAILS=-1,AGENT_CONFIG=$agent,SAVE_DIR=$save_dir" \
    "$train_script"
}

launch v2-vis-transformer-h18s30 \
  real_candy_scoop_v2_visual visual \
  agents/mtql_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_visual/transformer_h18s30 \
  run_candy_scoop_real.sh

launch v2-vis-mlp-h18s30 \
  real_candy_scoop_v2_visual visual \
  agents/mtql_mlp_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_visual/mlp_h18s30 \
  run_candy_scoop_real.sh

launch v2-vis-bc-h18s30 \
  real_candy_scoop_v2_visual visual \
  agents/new_bc_flow_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_visual/bc_h18s30 \
  run_candy_scoop_real_bc.sh

launch v2-lang-transformer-h18s30 \
  real_candy_scoop_v2_language language \
  agents/mtql_transformer_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/transformer_h18s30 \
  run_candy_scoop_real.sh

launch v2-lang-mlp-h18s30 \
  real_candy_scoop_v2_language language \
  agents/mtql_mlp_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/mlp_h18s30 \
  run_candy_scoop_real.sh

launch v2-lang-bc-h18s30 \
  real_candy_scoop_v2_language language \
  agents/new_bc_flow_transformer_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/bc_h18s30 \
  run_candy_scoop_real_bc.sh

squeue -u "$USER" -o "%.18i %.40j %.8T %.12M %R"
