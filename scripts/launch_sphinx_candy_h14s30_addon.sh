#!/usr/bin/env bash
set -euo pipefail

cd /iris/u/ronpo/projects/new_mtql_candy_scooping

SBATCH_COMMON=(
  --account=nlp
  --partition=sphinx
  --nodelist='sphinx[4-11]'
  --nodes=1
  --gres=gpu:1
  --cpus-per-task=8
  --mem=128G
  --time=120:00:00
)

launch() {
  local name="$1"
  local agent_config="$2"
  local save_dir="$3"
  local train_script="$4"

  sbatch "${SBATCH_COMMON[@]}" \
    --job-name="${name}" \
    --export="ALL,DATASET_PATH=/iris/u/ronpo/expo-ft-data/candy_scoop,DATASET_CACHE=/iris/u/ronpo/mtql-runs/candy_scoop_cache_full,PROJECT=real_candy_scoop,CUE_MODE=visual,P_AUG=1.0,SEED=0,HIST_LENGTH=14,HIST_STRIDE=30,BATCH_SIZE=16,TRAIN_STEPS=1000000,LOG_INTERVAL=30000,SAVE_INTERVAL=30000,N_SUCC=-1,N_FAILS=-1,AGENT_CONFIG=${agent_config},SAVE_DIR=${save_dir}" \
    "${train_script}"
}

launch transformer-h14s30 \
  agents/mtql_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/transformer_h14s30_sphinx_full \
  run_candy_scoop_real.sh

launch mlp-h14s30 \
  agents/mtql_mlp_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/mlp_h14s30_sphinx_full \
  run_candy_scoop_real.sh

launch bc-h14s30 \
  agents/new_bc_flow_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/bc_h14s30_sphinx_full \
  run_candy_scoop_real_bc.sh

echo "Submitted the H14/S30 addon runs."
squeue -u ronpo -o "%.18i %.35j %.8T %.12M %R"
