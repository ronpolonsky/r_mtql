#!/usr/bin/env bash
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=prep-candy-cache
#SBATCH --output=/iris/u/ronpo/projects/new_mtql_candy_scooping/slurm/%j.out

set -euo pipefail

ROOT="${CACHE_PROJECT_ROOT:-/iris/u/ronpo/projects/new_mtql_candy_scooping}"
PYTHON="${CACHE_PYTHON:-/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python}"
DATASET="${CACHE_DATASET_PATH:-/iris/u/ronpo/expo-ft-data/candy_scoop}"
STATS="${CACHE_NORM_STATS_PATH:-/iris/u/ronpo/expo-ft-data/candy_scoop_norm_stats}"
CACHE="${CACHE_DIR:-/iris/u/ronpo/mtql-runs/candy_scoop_cache_full}"
TASK="${CACHE_TASK_CONFIG:-/iris/u/ronpo/projects/expo-ft/configs/task/candy_scoop.py}"
CACHE_OVERWRITE="${CACHE_OVERWRITE:-0}"
CACHE_N_SUCC="${CACHE_N_SUCC:--1}"
CACHE_N_FAILS="${CACHE_N_FAILS:--1}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:/iris/u/ronpo/projects/expo-ft:${PYTHONPATH:-}"

args=(
  "${PYTHON}" -u scripts/preprocess_droid_cache.py \
  --dataset_path="${DATASET}" \
  --norm_stats_path="${STATS}" \
  --config_task="${TASK}" \
  --cache_dir="${CACHE}" \
  --image_size=224 \
  --n_succ="${CACHE_N_SUCC}" \
  --n_fails="${CACHE_N_FAILS}"
)
if [[ "${CACHE_OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi
exec "${args[@]}"
