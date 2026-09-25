#!/usr/bin/env bash
set -euo pipefail

# Matched v2 comparison for the history/stride recommendation.  This script
# only submits jobs when the user explicitly runs it; it does not modify the
# raw dataset or the derived cache.
cd /iris/u/ronpo/projects/new_mtql_candy_scooping

SBATCH_COMMON=(
  --account=nlp
  --partition=sphinx
  --nodelist='sphinx[9-11]'
  --nodes=1
  --gres=gpu:1
  --cpus-per-task=8
  --mem=128G
  --time=120:00:00
)

COMMON_EXPORT="ALL,DATASET_PATH=/iris/u/ronpo/expo-ft-data/candy_scoop_v2_jitted,DATASET_CACHE=/iris/u/ronpo/mtql-runs/candy_scoop_v2_jitted_cache_v2norm,NORM_STATS_PATH=/iris/u/ronpo/projects/new_mtql_candy_scooping/norm_stats/candy_scoop_v2_jitted,PROJECT=real_candy_scoop_v2_visual,CUE_MODE=visual,P_AUG=1.0,SEED=0,HIST_LENGTH=20,HIST_STRIDE=25,BATCH_SIZE=16,TRAIN_STEPS=1000000,LOG_INTERVAL=10000,SAVE_INTERVAL=10000,N_SUCC=-1,N_FAILS=-1"

launch() {
  local name="$1"
  local agent="$2"
  local save_dir="$3"
  local train_script="$4"

  sbatch "${SBATCH_COMMON[@]}" \
    --job-name="${name}" \
    --export="${COMMON_EXPORT},AGENT_CONFIG=${agent},SAVE_DIR=${save_dir}" \
    "${train_script}"
}

launch v2-transformer-h20s25 \
  agents/mtql_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2/transformer_h20s25 \
  run_candy_scoop_real.sh

launch v2-mlp-h20s25 \
  agents/mtql_mlp_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2/mlp_h20s25 \
  run_candy_scoop_real.sh

launch v2-bc-h20s25 \
  agents/new_bc_flow_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2/bc_h20s25 \
  run_candy_scoop_real_bc.sh

squeue -u ronpo -o "%.18i %.35j %.8T %.12M %R"
