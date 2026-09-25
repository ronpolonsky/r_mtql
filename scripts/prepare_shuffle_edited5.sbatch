#!/usr/bin/env bash
# Compute shuffle edited5 normalization stats and build its 224x224 cache.
# Submit with: sbatch scripts/prepare_shuffle_edited5.sbatch
# The raw recordings are read through a symlink-only dataset view.
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodelist=iris[1-10]
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=shuffle-e5-prep
#SBATCH --output=/iris/u/ronpo/projects/new_mtql_candy_scooping/slurm/shuffle-e5-prep-%j.out

set -euo pipefail

ROOT=/iris/u/ronpo/projects/new_mtql_candy_scooping
PYTHON=/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python
SOURCE_ROOT=/iris/u/ronpo/expo-ft-data/shuffle
DATASET_PATH=/iris/u/ronpo/mtql-runs/datasets/shuffle_edited5_view
NORM_STATS_PATH=/iris/u/ronpo/expo-ft-data/shuffle_edited5_norm_stats
DATASET_CACHE=/iris/u/ronpo/mtql-runs/caches/shuffle_edited5_224
TASK_CONFIG=/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py
EXPO_ROOT=/iris/u/ronpo/projects/expo-ft

cd "${ROOT}"
mkdir -p "${ROOT}/slurm"

[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 1; }
[[ -d "${SOURCE_ROOT}" ]] || { echo "Missing shuffle source: ${SOURCE_ROOT}" >&2; exit 1; }
[[ -f "${TASK_CONFIG}" ]] || { echo "Missing task config: ${TASK_CONFIG}" >&2; exit 1; }
[[ ! -e "${DATASET_CACHE}" ]] || { echo "Cache already exists: ${DATASET_CACHE}" >&2; exit 1; }
[[ ! -e "${NORM_STATS_PATH}/norm_stats.json" ]] || {
  echo "Norm stats already exist: ${NORM_STATS_PATH}/norm_stats.json" >&2
  exit 1
}

echo "shuffle edited5 preprocessing job ${SLURM_JOB_ID:-non-slurm}"
echo "node=$(hostname)"
df -h "${DATASET_CACHE%/*}"

"${PYTHON}" scripts/create_shuffle_edited5_view.py \
  --source-root "${SOURCE_ROOT}" \
  --view-dir "${DATASET_PATH}"

export PYTHONPATH="${EXPO_ROOT}:${ROOT}:${PYTHONPATH:-}"
"${PYTHON}" scripts/compute_egg_norm_stats.py \
  --dataset_path="${DATASET_PATH}" \
  --config_task="${TASK_CONFIG}" \
  --output_dir="${NORM_STATS_PATH}"

[[ -s "${NORM_STATS_PATH}/norm_stats.json" ]] || {
  echo "Shuffle normalization did not produce ${NORM_STATS_PATH}/norm_stats.json; refusing to build cache." >&2
  exit 1
}

"${PYTHON}" scripts/preprocess_egg_cache.py \
  --dataset_path="${DATASET_PATH}" \
  --config_task="${TASK_CONFIG}" \
  --norm_stats_path="${NORM_STATS_PATH}" \
  --cache_dir="${DATASET_CACHE}" \
  --image_size=224

echo "Completed shuffle edited5 stats and cache."
cat "${DATASET_CACHE}/metadata.json"
