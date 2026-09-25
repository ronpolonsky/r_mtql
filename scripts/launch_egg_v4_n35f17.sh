#!/usr/bin/env bash
# Launch the matched Egg v4 35-success / 17-failure triplet on sphinx[4-11].
#
# Validate without changing Slurm state:
#   DRY_RUN=1 ./scripts/launch_egg_v4_n35f17.sh
#
# Cancel the old 5S/3F triplet and launch the new triplet:
#   CANCEL_N5F3=1 ./scripts/launch_egg_v4_n35f17.sh
set -euo pipefail

ROOT=/iris/u/ronpo/projects/new_mtql_candy_scooping
PYTHON=/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python
DATASET_PATH=/iris/u/ronpo/expo-ft-data/egg_v4
DATASET_CACHE=/iris/u/ronpo/mtql-runs/caches/egg_v4_224
NORM_STATS_PATH=/iris/u/ronpo/expo-ft-data/egg_v4_norm_stats
TASK_CONFIG=/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py
TRAIN_SCRIPT="${ROOT}/run_egg_real.sh"
RUN_ROOT=/iris/u/ronpo/mtql-runs/egg_v4/n35f17

DRY_RUN="${DRY_RUN:-0}"
CANCEL_N5F3="${CANCEL_N5F3:-0}"
OLD_JOB_IDS=(17523883 17523884 17523885)

TRAIN_STEPS=1000000
LOG_INTERVAL=10000
SAVE_INTERVAL=30000
CHECKPOINT_MAX_TO_KEEP=3
CHECKPOINT_KEEP_PERIOD=150000
HIST_LENGTH=14
HIST_STRIDE=5
BATCH_SIZE=16
N_SUCC=35
N_FAILS=17

agents=(
  'transformer:agents/mtql_transformer_real.py'
  'mlp:agents/mtql_mlp_real.py'
  'bc:agents/new_bc_flow_transformer_real.py'
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

validate_inputs() {
  [[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] \
    || die "DRY_RUN must be 0 or 1"
  [[ "${CANCEL_N5F3}" == 0 || "${CANCEL_N5F3}" == 1 ]] \
    || die "CANCEL_N5F3 must be 0 or 1"
  command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
  command -v squeue >/dev/null 2>&1 || die "squeue is unavailable"
  command -v scancel >/dev/null 2>&1 || die "scancel is unavailable"
  [[ -x "${PYTHON}" ]] || die "Training Python is not executable: ${PYTHON}"
  [[ -x "${TRAIN_SCRIPT}" ]] || die "Training wrapper is missing: ${TRAIN_SCRIPT}"
  [[ -d "${DATASET_PATH}" ]] || die "Egg v4 dataset is missing: ${DATASET_PATH}"
  [[ -s "${DATASET_CACHE}/metadata.json" ]] || die "Egg v4 cache is missing"
  [[ -s "${NORM_STATS_PATH}/norm_stats.json" ]] \
    || die "Egg v4 normalization statistics are missing"
  [[ -f "${TASK_CONFIG}" ]] || die "Egg task configuration is missing"

  "${PYTHON}" - "${DATASET_CACHE}/metadata.json" \
    "${DATASET_PATH}" "${NORM_STATS_PATH}" <<'PY'
import json
import os
import sys

metadata_path, dataset_path, norm_stats_path = sys.argv[1:]
with open(metadata_path) as handle:
    metadata = json.load(handle)

expected = {
    "cache_format_version": 1,
    "dataset_kind": "egg",
    "full_finalized_dataset": True,
    "dataset_path": os.path.realpath(dataset_path),
    "norm_stats_dir": os.path.realpath(norm_stats_path),
    "image_size": 224,
    "num_episodes": 140,
    "num_transitions": 15799,
}
mismatches = {
    key: (metadata.get(key), value)
    for key, value in expected.items()
    if metadata.get(key) != value
}
if mismatches:
    raise SystemExit(f"Egg v4 cache validation failed: {mismatches}")
print("Validated Egg v4 cache: 140 episodes, 15,799 transitions")
PY
}

cancel_old_jobs() {
  local id
  local -a active_ids=()
  for id in "${OLD_JOB_IDS[@]}"; do
    if squeue -h -j "${id}" | grep -q .; then
      active_ids+=("${id}")
    fi
  done

  if [[ "${#active_ids[@]}" -eq 0 ]]; then
    echo "No 5S/3F jobs are active; nothing to cancel."
    return
  fi

  echo "Canceling old 5S/3F jobs: ${active_ids[*]}"
  scancel "${active_ids[@]}"
}

submit_jobs() {
  local entry label agent_config name save_dir exports submission job_id
  local -a command

  for entry in "${agents[@]}"; do
    IFS=: read -r label agent_config <<< "${entry}"
    name="egg-v4-${label}-n35f17"
    save_dir="${RUN_ROOT}/${label}"
    exports="ALL,PYTHON=${PYTHON},DATASET_PATH=${DATASET_PATH}"
    exports+=",DATASET_CACHE=${DATASET_CACHE},NORM_STATS_PATH=${NORM_STATS_PATH}"
    exports+=",TASK_CONFIG=${TASK_CONFIG},PROJECT=real_egg_v4"
    exports+=",WANDB_RUN_GROUP=egg_v4_n35f17,P_AUG=1.0,SEED=0"
    exports+=",HIST_LENGTH=${HIST_LENGTH},HIST_STRIDE=${HIST_STRIDE}"
    exports+=",BATCH_SIZE=${BATCH_SIZE},TRAIN_STEPS=${TRAIN_STEPS}"
    exports+=",LOG_INTERVAL=${LOG_INTERVAL},SAVE_INTERVAL=${SAVE_INTERVAL}"
    exports+=",CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP}"
    exports+=",CHECKPOINT_KEEP_PERIOD=${CHECKPOINT_KEEP_PERIOD}"
    exports+=",N_SUCC=${N_SUCC},N_FAILS=${N_FAILS}"
    exports+=",AGENT_CONFIG=${agent_config},SAVE_DIR=${save_dir}"

    command=(
      sbatch --parsable --account=nlp --partition=sphinx
      --nodelist='sphinx[4-11]' --nodes=1 --gres=gpu:1
      --cpus-per-task=8 --mem=128G --time=168:00:00
      --job-name="${name}" --output="${ROOT}/slurm/%x-%j.out"
      --export="${exports}" "${TRAIN_SCRIPT}"
    )

    if [[ "${DRY_RUN}" == 1 ]]; then
      printf 'DRY RUN %-27s:' "${name}"
      printf ' %q' "${command[@]}"
      printf '\n'
      continue
    fi

    submission=$("${command[@]}")
    job_id=${submission%%;*}
    submitted_jobs+=("${job_id}")
    printf 'Submitted %-27s job=%s data=%sS/%sF\n' \
      "${name}" "${job_id}" "${N_SUCC}" "${N_FAILS}"
  done
}

submitted_jobs=()

cd "${ROOT}"
mkdir -p slurm
validate_inputs

if [[ "${CANCEL_N5F3}" == 1 && "${DRY_RUN}" == 0 ]]; then
  cancel_old_jobs
elif [[ "${CANCEL_N5F3}" == 1 ]]; then
  echo "DRY RUN: would cancel active jobs among: ${OLD_JOB_IDS[*]}"
else
  echo "CANCEL_N5F3=0: old 5S/3F jobs will not be canceled."
fi

submit_jobs

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "Validated all three 35S/17F submissions; no jobs submitted."
  exit 0
fi

echo "Submitted all three Egg v4 35S/17F jobs: ${submitted_jobs[*]}"
job_csv=$(IFS=,; echo "${submitted_jobs[*]}")
squeue -j "${job_csv}" -o '%.18i %.30j %.10P %.2t %.10M %R' || true
