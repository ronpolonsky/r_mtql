#!/usr/bin/env bash
# Compute egg_v2 normalization statistics and, only after that succeeds, build
# the full 224x224 training cache. The raw dataset is read-only to this job.
#
# Submit with:
#   sbatch scripts/prepare_egg_v2_iris_hi.sbatch
#
# If normalization completed but cache construction was interrupted, reuse the
# existing statistics with:
#   sbatch --export=ALL,REUSE_EXISTING_STATS=1 \
#     scripts/prepare_egg_v2_iris_hi.sbatch
#
# Existing final outputs are never replaced unless OVERWRITE=1 is exported.
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodelist=iris-hgx-1,iris-hgx-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=egg-v2-prep
#SBATCH --output=/iris/u/ronpo/projects/new_mtql_candy_scooping/slurm/egg-v2-prep-%j.out

set -euo pipefail

ROOT="${EGG_PROJECT_ROOT:-/iris/u/ronpo/projects/new_mtql_candy_scooping}"
PYTHON="${PYTHON:-/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python}"
EXPO="${EXPO_ROOT:-/iris/u/ronpo/projects/expo-ft}"
TASK="${TASK_CONFIG:-/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py}"
DATASET="${DATASET_PATH:-/iris/u/ronpo/expo-ft-data/egg_v2}"
STATS="${NORM_STATS_PATH:-/iris/u/ronpo/expo-ft-data/egg_v2_norm_stats}"
CACHE="${DATASET_CACHE:-/iris/u/ronpo/mtql-runs/caches/egg_v2_224}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
OVERWRITE="${OVERWRITE:-0}"
REUSE_EXISTING_STATS="${REUSE_EXISTING_STATS:-0}"

EXPECTED_EPISODES=133
EXPECTED_TRANSITIONS=15616

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON}" ]] || die "Python is not executable: ${PYTHON}"
[[ -d "${ROOT}" ]] || die "Project directory is missing: ${ROOT}"
[[ -d "${EXPO}" ]] || die "EXPO-FT directory is missing: ${EXPO}"
[[ -f "${TASK}" ]] || die "Egg task config is missing: ${TASK}"
[[ -d "${DATASET}" ]] || die "egg_v2 dataset is missing: ${DATASET}"
[[ "${OVERWRITE}" == 0 || "${OVERWRITE}" == 1 ]] \
  || die "OVERWRITE must be 0 or 1"
[[ "${REUSE_EXISTING_STATS}" == 0 || "${REUSE_EXISTING_STATS}" == 1 ]] \
  || die "REUSE_EXISTING_STATS must be 0 or 1"

mkdir -p "${ROOT}/slurm"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${EXPO}:${PYTHONPATH:-}"

echo "egg_v2 preparation job ${SLURM_JOB_ID:-non-slurm}"
echo "dataset=${DATASET}"
echo "norm_stats=${STATS}"
echo "cache=${CACHE}"
echo "image_size=${IMAGE_SIZE}"
echo "node=$(hostname)"

stats_args=(
  "${PYTHON}" -u scripts/compute_egg_norm_stats.py
  "--dataset_path=${DATASET}"
  "--config_task=${TASK}"
  "--output_dir=${STATS}"
)

if [[ "${REUSE_EXISTING_STATS}" == 1 ]]; then
  [[ -r "${STATS}/norm_stats.json" ]] \
    || die "REUSE_EXISTING_STATS=1 but ${STATS}/norm_stats.json is missing"
  echo "Reusing existing normalization statistics: ${STATS}/norm_stats.json"
else
  if [[ "${OVERWRITE}" == 1 ]]; then
    stats_args+=(--overwrite)
  fi
  echo "Stage 1/2: computing egg_v2 normalization statistics"
  "${stats_args[@]}"
fi

[[ -s "${STATS}/norm_stats.json" ]] \
  || die "Normalization did not produce ${STATS}/norm_stats.json"

cache_args=(
  "${PYTHON}" -u scripts/preprocess_egg_cache.py
  "--dataset_path=${DATASET}"
  "--config_task=${TASK}"
  "--norm_stats_path=${STATS}"
  "--cache_dir=${CACHE}"
  "--image_size=${IMAGE_SIZE}"
)
if [[ "${OVERWRITE}" == 1 ]]; then
  cache_args+=(--overwrite)
fi

echo "Stage 2/2: building the egg_v2 224x224 cache"
"${cache_args[@]}"

echo "Validating finalized cache"
"${PYTHON}" - "${CACHE}" "${DATASET}" "${STATS}" \
  "${EXPECTED_EPISODES}" "${EXPECTED_TRANSITIONS}" "${IMAGE_SIZE}" <<'PY'
import json
import os
import sys
from pathlib import Path

import numpy as np

cache = Path(sys.argv[1])
dataset = os.path.realpath(sys.argv[2])
stats = os.path.realpath(sys.argv[3])
expected_episodes = int(sys.argv[4])
expected_transitions = int(sys.argv[5])
expected_image_size = int(sys.argv[6])

metadata_path = cache / "metadata.json"
if not metadata_path.is_file():
    raise SystemExit(f"Missing cache metadata: {metadata_path}")
metadata = json.loads(metadata_path.read_text())

expected_metadata = {
    "dataset_kind": "egg",
    "full_finalized_dataset": True,
    "dataset_path": dataset,
    "norm_stats_dir": stats,
    "image_size": expected_image_size,
    "num_episodes": expected_episodes,
    "num_transitions": expected_transitions,
}
for key, expected in expected_metadata.items():
    actual = metadata.get(key)
    if actual != expected:
        raise SystemExit(
            f"Cache validation failed for {key}: {actual!r} != {expected!r}"
        )

required_arrays = {
    "observations_image",
    "observations_wrist_image",
    "observations_proprio",
    "actions",
    "raw_actions",
    "rewards",
    "masks",
    "terminals",
}
if set(metadata.get("arrays", ())) != required_arrays:
    raise SystemExit(
        "Cache array manifest mismatch: "
        f"{sorted(metadata.get('arrays', ()))}"
    )

for name in sorted(required_arrays):
    path = cache / f"{name}.npy"
    if not path.is_file():
        raise SystemExit(f"Missing cache array: {path}")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if len(array) != expected_transitions:
        raise SystemExit(
            f"Unexpected length for {name}: {len(array)} != "
            f"{expected_transitions}"
        )

terminals = np.load(
    cache / "terminals.npy", allow_pickle=False, mmap_mode="r"
)
terminal_count = int(np.count_nonzero(terminals))
if terminal_count != expected_episodes or not bool(terminals[-1]):
    raise SystemExit(
        f"Invalid episode terminals: count={terminal_count}, "
        f"expected={expected_episodes}, final={terminals[-1]!r}"
    )

for name in ("observations_image", "observations_wrist_image"):
    array = np.load(cache / f"{name}.npy", allow_pickle=False, mmap_mode="r")
    expected_shape = (expected_image_size, expected_image_size, 3)
    if tuple(array.shape[1:]) != expected_shape or array.dtype != np.uint8:
        raise SystemExit(
            f"Invalid {name}: shape={array.shape}, dtype={array.dtype}"
        )

print(
    "Validated egg_v2 cache: "
    f"{expected_episodes} episodes, {expected_transitions} transitions, "
    f"{expected_image_size}x{expected_image_size} images at {cache}"
)
PY

echo "egg_v2 normalization and cache construction completed successfully"
