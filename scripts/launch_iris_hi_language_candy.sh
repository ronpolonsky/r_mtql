#!/usr/bin/env bash
set -euo pipefail

cd /iris/u/ronpo/projects/new_mtql_candy_scooping

COMMON=(
  --account=iris
  --partition=iris-hi
  --nodelist=iris5,iris7,iris9
  --nodes=1
  --gres=gpu:1
  --cpus-per-task=8
  --mem=128G
  --time=120:00:00
)

if [[ -n "${EXCLUDE_NODE:-}" ]]; then
  COMMON+=(--exclude="${EXCLUDE_NODE}")
fi

launch() {
  local name="$1"
  local agent="$2"
  local save_dir="$3"
  local train_script="$4"

  sbatch "${COMMON[@]}" \
    --job-name="${name}" \
    --export="ALL,DATASET_PATH=/iris/u/ronpo/expo-ft-data/candy_scoop,DATASET_CACHE=/iris/u/ronpo/mtql-runs/candy_scoop_cache_full,PROJECT=real_candy_scoop_language,CUE_MODE=language,P_AUG=1.0,SEED=0,HIST_LENGTH=18,HIST_STRIDE=30,BATCH_SIZE=16,TRAIN_STEPS=1000000,LOG_INTERVAL=10000,SAVE_INTERVAL=10000,N_SUCC=-1,N_FAILS=-1,AGENT_CONFIG=${agent},SAVE_DIR=${save_dir}" \
    "${train_script}"
}

launch lang-transformer-h18s30 \
  agents/mtql_transformer_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_language/transformer_h18s30_jit \
  run_candy_scoop_real.sh

launch lang-mlp-h18s30 \
  agents/mtql_mlp_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_language/mlp_h18s30_jit \
  run_candy_scoop_real.sh

launch lang-bc-h18s30 \
  agents/new_bc_flow_transformer_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_language/bc_h18s30_jit \
  run_candy_scoop_real_bc.sh

echo "Submitted three language-conditioned Candy-Scoop runs."
squeue -u ronpo -o "%.18i %.35j %.8T %.12M %R"
