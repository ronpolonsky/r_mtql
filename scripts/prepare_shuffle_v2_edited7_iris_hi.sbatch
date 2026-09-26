#!/usr/bin/env bash
# Prepare shuffle_v2 edited7 using the verified Egg v4 norm/cache wrapper.
# The only shuffle-specific step is creating the symlink-only dataset view.
#
# Submit from a Slurm login shell with:
#   sbatch scripts/prepare_shuffle_v2_edited7_iris_hi.sbatch
#
# The source HDF5 files are never modified. The dataset view consists only of
# symlinks to the edited7 trajectories.
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodelist=iris[1-10]
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=shuffle-e7-prep
#SBATCH --output=/iris/u/ronpo/projects/new_mtql_candy_scooping/slurm/shuffle-e7-prep-%j.out

set -euo pipefail

ROOT=/iris/u/ronpo/projects/new_mtql_candy_scooping
PYTHON=/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python
SOURCE_ROOT=/iris/u/ronpo/expo-ft-data/shuffle_v2
DATASET_PATH=/iris/u/ronpo/mtql-runs/datasets/shuffle_v2_edited7_view
NORM_STATS_PATH=/iris/u/ronpo/expo-ft-data/shuffle_v2_edited7_norm_stats
DATASET_CACHE=/iris/u/ronpo/mtql-runs/caches/shuffle_v2_edited7_224
TASK_CONFIG=/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py
EXPO_ROOT=/iris/u/ronpo/projects/expo-ft
OVERWRITE=${OVERWRITE:-0}

cd "${ROOT}"
mkdir -p "${ROOT}/slurm"
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 1; }
[[ -d "${SOURCE_ROOT}" ]] || { echo "Missing source: ${SOURCE_ROOT}" >&2; exit 1; }
[[ -f "${TASK_CONFIG}" ]] || { echo "Missing task config: ${TASK_CONFIG}" >&2; exit 1; }

echo "shuffle_v2 edited7 preparation job ${SLURM_JOB_ID:-non-slurm} on $(hostname)"

"${PYTHON}" -u scripts/create_shuffle_edited7_view.py \
  --source-root "${SOURCE_ROOT}" \
  --view-dir "${DATASET_PATH}"

echo "Delegating normalization and cache construction to the Egg v4 wrapper"
EGG_PROJECT_ROOT="${ROOT}" \
EXPO_ROOT="${EXPO_ROOT}" \
PYTHON="${PYTHON}" \
DATASET_PATH="${DATASET_PATH}" \
NORM_STATS_PATH="${NORM_STATS_PATH}" \
DATASET_CACHE="${DATASET_CACHE}" \
TASK_CONFIG="${TASK_CONFIG}" \
IMAGE_SIZE=224 \
OVERWRITE="${OVERWRITE}" \
  ./scripts/prepare_egg_v4.sh

echo "Completed shuffle_v2 edited7 normalization and cache: ${DATASET_CACHE}"
