#!/usr/bin/env bash
# Compute full egg_v4 normalization statistics and build its reusable cache.
# Subset selection (--n_succ/--n_fails) happens later inside the trainer.
set -euo pipefail

ROOT="${EGG_PROJECT_ROOT:-/iris/u/ronpo/projects/new_mtql_candy_scooping}"
EXPO_ROOT="${EXPO_ROOT:-/iris/u/ronpo/projects/expo-ft}"
PYTHON="${PYTHON:-/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python}"
DATASET_PATH="${DATASET_PATH:-/iris/u/ronpo/expo-ft-data/egg_v4}"
NORM_STATS_PATH="${NORM_STATS_PATH:-/iris/u/ronpo/expo-ft-data/egg_v4_norm_stats}"
DATASET_CACHE="${DATASET_CACHE:-/iris/u/ronpo/mtql-runs/caches/egg_v4_224}"
TASK_CONFIG="${TASK_CONFIG:-/afs/cs.stanford.edu/u/ronpo/projects/expo-ft/configs/task/egg_v2.py}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
OVERWRITE="${OVERWRITE:-0}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON}" ]] || die "Python not found or not executable: ${PYTHON}"
[[ -d "${DATASET_PATH}" ]] || die "Dataset not found: ${DATASET_PATH}"
[[ -f "${TASK_CONFIG}" ]] || die "Task config not found: ${TASK_CONFIG}"
[[ "${OVERWRITE}" == 0 || "${OVERWRITE}" == 1 ]] || die "OVERWRITE must be 0 or 1"

if [[ "${OVERWRITE}" == 0 ]]; then
  if [[ -e "${DATASET_CACHE}" ]]; then
    [[ -s "${DATASET_CACHE}/metadata.json" ]] \
      || die "Incomplete cache path exists: ${DATASET_CACHE}"
    die "Completed cache already exists: ${DATASET_CACHE}; use OVERWRITE=1"
  fi
  if [[ -e "${NORM_STATS_PATH}" && ! -s "${NORM_STATS_PATH}/norm_stats.json" ]]; then
    die "Incomplete norm-stats path exists: ${NORM_STATS_PATH}"
  fi
fi

# EXPO-FT must precede this project's partial ``configs`` namespace so
# configs/task/egg_v2.py can import configs.task.egg reliably on compute nodes.
export PYTHONPATH="${EXPO_ROOT}:${ROOT}:${PYTHONPATH:-}"

echo "Preparing Egg v4"
echo "  dataset: ${DATASET_PATH}"
echo "  norm stats: ${NORM_STATS_PATH}"
echo "  cache: ${DATASET_CACHE}"

if [[ "${OVERWRITE}" == 0 && -s "${NORM_STATS_PATH}/norm_stats.json" ]]; then
  echo "Reusing completed norm stats: ${NORM_STATS_PATH}/norm_stats.json"
else
  "${PYTHON}" "${ROOT}/scripts/compute_droid_norm_stats.py" \
    --dataset-path="${DATASET_PATH}" \
    --output-dir="${NORM_STATS_PATH}"
fi

cache_args=(
  "${PYTHON}" "${ROOT}/scripts/preprocess_egg_cache.py"
  --dataset_path="${DATASET_PATH}" \
  --norm_stats_path="${NORM_STATS_PATH}" \
  --cache_dir="${DATASET_CACHE}" \
  --config_task="${TASK_CONFIG}" \
  --image_size="${IMAGE_SIZE}"
)
if [[ "${OVERWRITE}" == 1 ]]; then
  cache_args+=(--overwrite)
fi
"${cache_args[@]}"

[[ -s "${NORM_STATS_PATH}/norm_stats.json" ]] \
  || die "Norm stats were not created: ${NORM_STATS_PATH}/norm_stats.json"
[[ -s "${DATASET_CACHE}/metadata.json" ]] \
  || die "Cache metadata was not created: ${DATASET_CACHE}/metadata.json"

echo "Egg v4 preparation complete."
