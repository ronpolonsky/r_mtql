#!/usr/bin/env bash
# Cancel the active Egg v4 sweep and launch exactly 21 replacement jobs:
# seven data settings x Transformer/MLP/BC.
#
# Settings:
#   10S/5F, 15S/8F, 20S/10F, 25S/12F, 35S/17F, 50S/25F, 70S/35F
#
# All runs use:
#   1,000,000 train steps; checkpoints every 30,000 steps; H14/S5;
#   MTQL normalize_q_loss=false; attention entropy target
#   ((3.5,3.5),(3.0,3.0)); and effectively unlimited checkpoint retention.
#
# Read-only validation:
#   DRY_RUN=1 CANCEL_EXISTING=1 ./scripts/relaunch_egg_v4_21_keep_all.sh
#
# Actual replacement, after sufficient storage has been made available:
#   CANCEL_EXISTING=1 ./scripts/relaunch_egg_v4_21_keep_all.sh
set -euo pipefail

ROOT=/iris/u/ronpo/projects/new_mtql_candy_scooping
PYTHON=/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python
DATASET_PATH=/iris/u/ronpo/expo-ft-data/egg_v4
DATASET_CACHE=/iris/u/ronpo/mtql-runs/caches/egg_v4_224
NORM_STATS_PATH=/iris/u/ronpo/expo-ft-data/egg_v4_norm_stats
TASK_CONFIG=/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py
TRAIN_SCRIPT="${ROOT}/run_egg_real.sh"
RUN_ROOT=/iris/u/ronpo/mtql-runs/egg_v4

DRY_RUN="${DRY_RUN:-0}"
CANCEL_EXISTING="${CANCEL_EXISTING:-0}"
ALLOW_LOW_SPACE="${ALLOW_LOW_SPACE:-0}"

TRAIN_STEPS=1000000
LOG_INTERVAL=10000
SAVE_INTERVAL=30000
CHECKPOINT_MAX_TO_KEEP=100000
HIST_LENGTH=14
HIST_STRIDE=5
BATCH_SIZE=16
TIME_LIMIT=168:00:00

# Seven complete checkpoint series currently require roughly 609 GiB. Require
# a modest safety margin before canceling or submitting anything.
MIN_FREE_GIB=650

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
  [[ "${CANCEL_EXISTING}" == 0 || "${CANCEL_EXISTING}" == 1 ]] \
    || die "CANCEL_EXISTING must be 0 or 1"
  [[ "${ALLOW_LOW_SPACE}" == 0 || "${ALLOW_LOW_SPACE}" == 1 ]] \
    || die "ALLOW_LOW_SPACE must be 0 or 1"
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
    "action_space": "cartesian_velocity",
    "gripper_action_space": "velocity",
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

  # The shared wrapper supplies these exact MTQL settings. Fail rather than
  # silently launch if they are edited away later.
  grep -q -- '"--agent.normalize_q_loss=false"' "${TRAIN_SCRIPT}" \
    || die "Training wrapper no longer fixes normalize_q_loss=false"
  grep -Fq -- '"--agent.attention_entropy_target=((3.5,3.5),(3.0,3.0))"' \
    "${TRAIN_SCRIPT}" \
    || die "Training wrapper no longer fixes the requested entropy target"
}

validate_storage() {
  local available_kib available_gib
  available_kib=$(df -Pk "${RUN_ROOT}" | awk 'NR == 2 {print $4}')
  [[ "${available_kib}" =~ ^[0-9]+$ ]] \
    || die "Could not determine available storage under ${RUN_ROOT}"
  available_gib=$((available_kib / 1024 / 1024))
  echo "Storage preflight: ${available_gib} GiB available; ${MIN_FREE_GIB} GiB required"

  if (( available_gib < MIN_FREE_GIB )); then
    if [[ "${DRY_RUN}" == 1 ]]; then
      echo "DRY RUN WARNING: insufficient space to retain every checkpoint."
    elif [[ "${ALLOW_LOW_SPACE}" != 1 ]]; then
      die "Insufficient storage. Free space first, or set ALLOW_LOW_SPACE=1 to override at your own risk."
    else
      echo "WARNING: proceeding despite insufficient estimated storage." >&2
    fi
  fi
}

collect_active_v4_jobs() {
  local line id name
  active_v4_jobs=()
  while IFS='|' read -r id name; do
    case "${name}" in
      egg-v4-transformer-*|egg-v4-mlp-*|egg-v4-bc-*)
        active_v4_jobs+=("${id}")
        ;;
    esac
  done < <(squeue -h -u ronpo -o '%i|%j')
}

cancel_active_v4_jobs() {
  collect_active_v4_jobs
  if [[ "${#active_v4_jobs[@]}" -eq 0 ]]; then
    echo "No active Egg v4 jobs found; nothing to cancel."
    return
  fi

  echo "Canceling active Egg v4 jobs only: ${active_v4_jobs[*]}"
  scancel "${active_v4_jobs[@]}"
}

submitted_jobs=()
planned_jobs=0

submit_setting() {
  local account="$1"
  local partition="$2"
  local nodelist="$3"
  local setting="$4"
  local n_succ="$5"
  local n_fails="$6"
  local entry label agent_config name save_dir exports submission job_id
  local -a command

  for entry in "${agents[@]}"; do
    IFS=: read -r label agent_config <<< "${entry}"
    planned_jobs=$((planned_jobs + 1))
    name="egg-v4-${label}-${setting}"
    save_dir="${RUN_ROOT}/${setting}/${label}"
    exports="ALL,PYTHON=${PYTHON},DATASET_PATH=${DATASET_PATH}"
    exports+=",DATASET_CACHE=${DATASET_CACHE},NORM_STATS_PATH=${NORM_STATS_PATH}"
    exports+=",TASK_CONFIG=${TASK_CONFIG},PROJECT=real_egg_v4"
    exports+=",WANDB_RUN_GROUP=egg_v4_${setting},P_AUG=1.0,SEED=0"
    exports+=",HIST_LENGTH=${HIST_LENGTH},HIST_STRIDE=${HIST_STRIDE}"
    exports+=",BATCH_SIZE=${BATCH_SIZE},TRAIN_STEPS=${TRAIN_STEPS}"
    exports+=",LOG_INTERVAL=${LOG_INTERVAL},SAVE_INTERVAL=${SAVE_INTERVAL}"
    exports+=",CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP}"
    exports+=",N_SUCC=${n_succ},N_FAILS=${n_fails}"
    exports+=",AGENT_CONFIG=${agent_config},SAVE_DIR=${save_dir}"

    command=(
      sbatch --parsable --account="${account}" --partition="${partition}"
      --nodelist="${nodelist}" --nodes=1 --gres=gpu:1
      --cpus-per-task=8 --mem=128G --time="${TIME_LIMIT}"
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
    printf 'Submitted %-27s job=%s nodes=%s data=%sS/%sF\n' \
      "${name}" "${job_id}" "${nodelist}" "${n_succ}" "${n_fails}"
  done
}

cd "${ROOT}"
mkdir -p slurm
validate_inputs
validate_storage

if [[ "${CANCEL_EXISTING}" == 1 && "${DRY_RUN}" == 0 ]]; then
  cancel_active_v4_jobs
elif [[ "${CANCEL_EXISTING}" == 1 ]]; then
  collect_active_v4_jobs
  echo "DRY RUN: would cancel these Egg v4 jobs: ${active_v4_jobs[*]:-none}"
else
  echo "CANCEL_EXISTING=0: existing Egg v4 jobs will not be canceled."
fi

# Six jobs on iris-hi: one matched setting on HGX and one on regular Iris.
submit_setting iris iris-hi 'iris-hgx-[1-2]' n25f12 25 12
submit_setting iris iris-hi 'iris[5,7,9]' n50f25 50 25

# Fifteen jobs on sphinx[4-11]: five matched settings.
submit_setting nlp sphinx 'sphinx[4-11]' n10f5 10 5
submit_setting nlp sphinx 'sphinx[4-11]' n15f8 15 8
submit_setting nlp sphinx 'sphinx[4-11]' n20f10 20 10
submit_setting nlp sphinx 'sphinx[4-11]' n35f17 35 17
submit_setting nlp sphinx 'sphinx[4-11]' n70f35 70 35

[[ "${planned_jobs}" -eq 21 ]] \
  || die "Internal error: expected exactly 21 jobs, planned ${planned_jobs}"

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "Validated exactly 21 Egg v4 submissions; no jobs submitted."
  exit 0
fi

[[ "${#submitted_jobs[@]}" -eq 21 ]] \
  || die "Expected 21 submitted jobs, got ${#submitted_jobs[@]}"
echo "Submitted all 21 Egg v4 jobs: ${submitted_jobs[*]}"
job_csv=$(IFS=,; echo "${submitted_jobs[*]}")
squeue -j "${job_csv}" -o '%.18i %.30j %.10P %.2t %.10M %R' || true
