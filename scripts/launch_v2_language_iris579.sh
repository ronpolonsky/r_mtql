#!/usr/bin/env bash
set -euo pipefail

cd /iris/u/ronpo/projects/new_mtql_candy_scooping

DATASET_PATH=/iris/u/ronpo/expo-ft-data/candy_scoop_v2_jitted
DATASET_CACHE=/iris/u/ronpo/mtql-runs/candy_scoop_v2_jitted_cache_v2norm
NORM_STATS_PATH=/iris/u/ronpo/projects/new_mtql_candy_scooping/norm_stats/candy_scoop_v2_jitted
PROJECT=real_candy_scoop_v2_language

launch() {
  local name="$1"
  local node="$2"
  local agent="$3"
  local save_dir="$4"
  local train_script="$5"

  sbatch \
    --account=iris \
    --partition=iris-hi \
    --nodelist="$node" \
    --nodes=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=128G \
    --time=120:00:00 \
    --job-name="$name" \
    --export="ALL,DATASET_PATH=$DATASET_PATH,DATASET_CACHE=$DATASET_CACHE,NORM_STATS_PATH=$NORM_STATS_PATH,PROJECT=$PROJECT,CUE_MODE=language,P_AUG=1.0,SEED=0,HIST_LENGTH=20,HIST_STRIDE=25,BATCH_SIZE=16,TRAIN_STEPS=1000000,LOG_INTERVAL=10000,SAVE_INTERVAL=10000,N_SUCC=-1,N_FAILS=-1,AGENT_CONFIG=$agent,SAVE_DIR=$save_dir" \
    "$train_script"
}

launch v2-lang-transformer-h20s25 iris5 \
  agents/mtql_transformer_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/transformer_h20s25 \
  run_candy_scoop_real.sh

launch v2-lang-mlp-h20s25 iris7 \
  agents/mtql_mlp_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/mlp_h20s25 \
  run_candy_scoop_real.sh

launch v2-lang-bc-h20s25 iris9 \
  agents/new_bc_flow_transformer_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/bc_h20s25 \
  run_candy_scoop_real_bc.sh

squeue -u "$USER" -o "%.18i %.35j %.8T %.12M %R"
