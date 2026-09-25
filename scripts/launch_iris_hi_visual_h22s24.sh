#!/usr/bin/env bash
set -euo pipefail

cd /iris/u/ronpo/projects/new_mtql_candy_scooping

COMMON=(
  --account=iris
  --partition=iris-hi
  --nodelist=iris-hgx-1,iris-hgx-2
  --nodes=1
  --gres=gpu:1
  --cpus-per-task=8
  --mem=128G
  --time=120:00:00
)

launch() {
  local name="$1"
  local agent="$2"
  local save_dir="$3"
  local train_script="$4"

  sbatch "${COMMON[@]}" \
    --job-name="${name}" \
    --export="ALL,DATASET_PATH=/iris/u/ronpo/expo-ft-data/candy_scoop,DATASET_CACHE=/iris/u/ronpo/mtql-runs/candy_scoop_cache_full,PROJECT=real_candy_scoop,CUE_MODE=visual,P_AUG=1.0,SEED=0,HIST_LENGTH=22,HIST_STRIDE=24,BATCH_SIZE=16,TRAIN_STEPS=1000000,LOG_INTERVAL=10000,SAVE_INTERVAL=10000,N_SUCC=-1,N_FAILS=-1,AGENT_CONFIG=${agent},SAVE_DIR=${save_dir}" \
    "${train_script}"
}

launch visual-transformer-h22s24 \
  agents/mtql_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/transformer_h22s24_jit \
  run_candy_scoop_real.sh

launch visual-mlp-h22s24 \
  agents/mtql_mlp_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/mlp_h22s24_jit \
  run_candy_scoop_real.sh

launch visual-bc-h22s24 \
  agents/new_bc_flow_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/bc_h22s24_jit \
  run_candy_scoop_real_bc.sh

echo "Submitted three visual H22/S24 Candy-Scoop runs."
squeue -u ronpo -o "%.18i %.35j %.8T %.12M %R"
