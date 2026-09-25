#!/usr/bin/env bash
# Submit the six matched egg_v2 benchmark jobs.
#
#   H18/S4: transformer, MLP, and BC on iris-hi H100/H200 nodes.
#   H14/S5: transformer, MLP, and BC on sphinx10-11.
# MTQL agents train from every outcome; BC samples successful episodes only.
#
# Validate without submitting:
#   DRY_RUN=1 ./scripts/launch_egg_v2_benchmark.sh
#
# Submit:
#   ./scripts/launch_egg_v2_benchmark.sh
set -euo pipefail

ROOT=/iris/u/ronpo/projects/new_mtql_candy_scooping
PYTHON=/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python
DATASET_PATH=/iris/u/ronpo/expo-ft-data/egg_v2
DATASET_CACHE=/iris/u/ronpo/mtql-runs/caches/egg_v2_224
NORM_STATS_PATH=/iris/u/ronpo/expo-ft-data/egg_v2_norm_stats
TASK_CONFIG=/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py
PROJECT=real_egg_v2
RUN_ROOT=/iris/u/ronpo/mtql-runs/egg_v2
TRAIN_SCRIPT=run_egg_real.sh
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"

agents=(
  "transformer:agents/mtql_transformer_real.py"
  "mlp:agents/mtql_mlp_real.py"
  "bc:agents/new_bc_flow_transformer_real.py"
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] \
  || die "DRY_RUN must be 0 or 1"
[[ -x "${PYTHON}" ]] || die "Training Python is not executable: ${PYTHON}"
[[ -d "${DATASET_PATH}" ]] || die "Dataset is missing: ${DATASET_PATH}"
[[ -s "${DATASET_CACHE}/metadata.json" ]] \
  || die "Completed cache metadata is missing: ${DATASET_CACHE}/metadata.json"
[[ -s "${NORM_STATS_PATH}/norm_stats.json" ]] \
  || die "Normalization statistics are missing: ${NORM_STATS_PATH}/norm_stats.json"
[[ -f "${TASK_CONFIG}" ]] || die "Task config is missing: ${TASK_CONFIG}"
[[ -f "${ROOT}/${TRAIN_SCRIPT}" ]] || die "Training wrapper is missing"

cd "${ROOT}"
mkdir -p slurm

# Fail before submitting anything if the cache is stale or belongs to a
# different dataset/statistics directory.
"${PYTHON}" - "${DATASET_CACHE}/metadata.json" \
  "${DATASET_PATH}" "${NORM_STATS_PATH}" <<'PY'
import json
import os
import sys

metadata_path, dataset, stats = sys.argv[1:]
with open(metadata_path) as handle:
    metadata = json.load(handle)

expected = {
    "cache_format_version": 1,
    "dataset_kind": "egg",
    "full_finalized_dataset": True,
    "dataset_path": os.path.realpath(dataset),
    "norm_stats_dir": os.path.realpath(stats),
    "action_space": "cartesian_velocity",
    "gripper_action_space": "velocity",
    "image_size": 224,
    "num_episodes": 133,
    "num_transitions": 15616,
}
mismatches = {
    key: (metadata.get(key), value)
    for key, value in expected.items()
    if metadata.get(key) != value
}
if mismatches:
    raise SystemExit(f"egg_v2 cache validation failed: {mismatches}")
print(
    "Validated egg_v2 cache: "
    f"{metadata['num_episodes']} episodes, "
    f"{metadata['num_transitions']} transitions"
)
PY

job_ids=()

submit() {
  local account="$1"
  local partition="$2"
  local nodelist="$3"
  local setting="$4"
  local hist_length="$5"
  local hist_stride="$6"
  local agent_label="$7"
  local agent_config="$8"

  local name="egg-v2-${agent_label}-${setting}"
  local save_dir="${RUN_ROOT}/${setting}/${agent_label}"
  local group="egg_v2_${setting}"
  local exports
  exports="ALL,DATASET_PATH=${DATASET_PATH},DATASET_CACHE=${DATASET_CACHE}"
  exports+=",NORM_STATS_PATH=${NORM_STATS_PATH},TASK_CONFIG=${TASK_CONFIG}"
  exports+=",PROJECT=${PROJECT},WANDB_RUN_GROUP=${group},P_AUG=1.0"
  exports+=",SEED=${SEED},HIST_LENGTH=${hist_length},HIST_STRIDE=${hist_stride}"
  exports+=",BATCH_SIZE=${BATCH_SIZE},TRAIN_STEPS=${TRAIN_STEPS}"
  exports+=",LOG_INTERVAL=10000,SAVE_INTERVAL=${SAVE_INTERVAL}"
  exports+=",AGENT_CONFIG=${agent_config},SAVE_DIR=${save_dir}"

  local -a command=(
    sbatch
    --parsable
    --account="${account}"
    --partition="${partition}"
    --nodelist="${nodelist}"
    --nodes=1
    --gres=gpu:1
    --cpus-per-task=8
    --mem=128G
    --time=120:00:00
    --job-name="${name}"
    --output="${ROOT}/slurm/%x-%j.out"
    --export="${exports}"
    "${TRAIN_SCRIPT}"
  )

  if [[ "${DRY_RUN}" == 1 ]]; then
    printf 'DRY RUN %s:' "${name}"
    printf ' %q' "${command[@]}"
    printf '\n'
    return
  fi

  local job_id
  job_id=$("${command[@]}")
  job_id="${job_id%%;*}"
  job_ids+=("${job_id}")
  printf 'Submitted %-29s job=%s nodes=%s H=%s S=%s\n' \
    "${name}" "${job_id}" "${nodelist}" "${hist_length}" "${hist_stride}"
}

# Primary setting: all three agents compete for either HGX H100/H200 node.
for entry in "${agents[@]}"; do
  IFS=: read -r label config <<< "${entry}"
  submit iris iris-hi 'iris-hgx-1,iris-hgx-2' \
    h18s4 18 4 "${label}" "${config}"
done

# Secondary setting: all three agents compete for sphinx10 or sphinx11.
for entry in "${agents[@]}"; do
  IFS=: read -r label config <<< "${entry}"
  submit nlp sphinx 'sphinx[10-11]' \
    h14s5 14 5 "${label}" "${config}"
done

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "Validated all six egg_v2 submission commands; no jobs submitted."
  exit 0
fi

echo "Submitted all six egg_v2 jobs: ${job_ids[*]}"
squeue -j "$(IFS=,; echo "${job_ids[*]}")" \
  -o "%.18i %.32j %.9P %.2t %.10M %R" || true
