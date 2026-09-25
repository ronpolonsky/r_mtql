#!/usr/bin/env bash
# Submit Egg v4 preprocessing (when needed), then six matched three-agent
# data-scaling settings. Training jobs start only after preprocessing succeeds.
set -euo pipefail

ROOT="${EGG_PROJECT_ROOT:-/iris/u/ronpo/projects/new_mtql_candy_scooping}"
DATASET_PATH="${DATASET_PATH:-/iris/u/ronpo/expo-ft-data/egg_v4}"
DATASET_CACHE="${DATASET_CACHE:-/iris/u/ronpo/mtql-runs/caches/egg_v4_224}"
NORM_STATS_PATH="${NORM_STATS_PATH:-/iris/u/ronpo/expo-ft-data/egg_v4_norm_stats}"
TASK_CONFIG="${TASK_CONFIG:-/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py}"
RUN_ROOT="${RUN_ROOT:-/iris/u/ronpo/mtql-runs/egg_v4}"
PROJECT="${PROJECT:-real_egg_v4}"
TRAIN_SCRIPT="${ROOT}/run_egg_real.sh"
PREP_SCRIPT="${ROOT}/scripts/prepare_egg_v4_iris_hi.sbatch"

DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-30000}"
HIST_LENGTH="${HIST_LENGTH:-14}"
HIST_STRIDE="${HIST_STRIDE:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TIME_LIMIT="${TIME_LIMIT:-168:00:00}"

# Checkpoints are produced every 30k, but bounded retention prevents this
# 18-run sweep from exhausting the shared filesystem. Every 150k checkpoint
# is protected, alongside the three latest unprotected checkpoints.
CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-3}"
CHECKPOINT_KEEP_PERIOD="${CHECKPOINT_KEEP_PERIOD:-150000}"

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
[[ -d "${DATASET_PATH}" ]] || die "Dataset is missing: ${DATASET_PATH}"
[[ -f "${TASK_CONFIG}" ]] || die "Task config is missing: ${TASK_CONFIG}"
[[ -x "${TRAIN_SCRIPT}" ]] || die "Training wrapper is missing: ${TRAIN_SCRIPT}"
[[ -x "${PREP_SCRIPT}" ]] || die "Preparation job is missing: ${PREP_SCRIPT}"

cd "${ROOT}"
mkdir -p "${ROOT}/slurm"

stats_ready=0
cache_ready=0
[[ -s "${NORM_STATS_PATH}/norm_stats.json" ]] && stats_ready=1
[[ -s "${DATASET_CACHE}/metadata.json" ]] && cache_ready=1

dependency_job="${PREP_JOB_ID:-}"
if [[ -n "${dependency_job}" ]]; then
  [[ "${dependency_job}" =~ ^[0-9]+$ ]] \
    || die "PREP_JOB_ID must be a numeric Slurm job ID"
  echo "Using existing preparation dependency: ${dependency_job}"
elif [[ "${stats_ready}" == 1 && "${cache_ready}" == 1 ]]; then
  echo "Egg v4 normalization and cache already exist; no dependency needed."
elif [[ "${cache_ready}" == 1 && "${stats_ready}" == 0 ]]; then
  die "Cache metadata exists without normalization statistics; inspect the inconsistent preparation outputs"
elif [[ "${DRY_RUN}" == 1 ]]; then
  dependency_job="DRY_RUN_PREP"
  printf 'DRY RUN prep:'
  printf ' %q' sbatch --parsable \
    --export="ALL,TASK_CONFIG=${TASK_CONFIG}" "${PREP_SCRIPT}"
  printf '\n'
else
  dependency_job=$(sbatch --parsable \
    --export="ALL,TASK_CONFIG=${TASK_CONFIG}" "${PREP_SCRIPT}")
  dependency_job="${dependency_job%%;*}"
  echo "Submitted Egg v4 preparation job: ${dependency_job}"
fi

job_ids=()

submit_setting() {
  local account="$1"
  local partition="$2"
  local nodelist="$3"
  local setting="$4"
  local n_succ="$5"
  local n_fails="$6"

  local entry label config name save_dir group exports job_id
  for entry in "${agents[@]}"; do
    IFS=: read -r label config <<< "${entry}"
    name="egg-v4-${label}-${setting}"
    save_dir="${RUN_ROOT}/${setting}/${label}"
    group="egg_v4_${setting}"
    exports="ALL,DATASET_PATH=${DATASET_PATH},DATASET_CACHE=${DATASET_CACHE}"
    exports+=",NORM_STATS_PATH=${NORM_STATS_PATH},TASK_CONFIG=${TASK_CONFIG}"
    exports+=",PROJECT=${PROJECT},WANDB_RUN_GROUP=${group},P_AUG=1.0"
    exports+=",SEED=${SEED},HIST_LENGTH=${HIST_LENGTH},HIST_STRIDE=${HIST_STRIDE}"
    exports+=",BATCH_SIZE=${BATCH_SIZE},TRAIN_STEPS=${TRAIN_STEPS}"
    exports+=",LOG_INTERVAL=${LOG_INTERVAL},SAVE_INTERVAL=${SAVE_INTERVAL}"
    exports+=",CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP}"
    exports+=",CHECKPOINT_KEEP_PERIOD=${CHECKPOINT_KEEP_PERIOD}"
    exports+=",N_SUCC=${n_succ},N_FAILS=${n_fails}"
    exports+=",AGENT_CONFIG=${config},SAVE_DIR=${save_dir}"

    command=(
      sbatch
      --parsable
      --account="${account}"
      --partition="${partition}"
      --nodelist="${nodelist}"
      --nodes=1
      --gres=gpu:1
      --cpus-per-task=8
      --mem=128G
      --time="${TIME_LIMIT}"
      --job-name="${name}"
      --output="${ROOT}/slurm/%x-%j.out"
      --export="${exports}"
    )
    if [[ -n "${dependency_job}" ]]; then
      command+=(--dependency="afterok:${dependency_job}" --kill-on-invalid-dep=yes)
    fi
    command+=("${TRAIN_SCRIPT}")

    if [[ "${DRY_RUN}" == 1 ]]; then
      printf 'DRY RUN %-25s:' "${name}"
      printf ' %q' "${command[@]}"
      printf '\n'
      continue
    fi

    job_id=$("${command[@]}")
    job_id="${job_id%%;*}"
    job_ids+=("${job_id}")
    printf 'Submitted %-25s job=%s nodes=%s data=%sS/%sF\n' \
      "${name}" "${job_id}" "${nodelist}" "${n_succ}" "${n_fails}"
  done
}

# Highest-priority limited-data setting on HGX H100/H200.
submit_setting iris iris-hi 'iris-hgx-[1-2]' n25f12 25 12

# Full-data ceiling on the other requested iris-hi pool.
submit_setting iris iris-hi 'iris[5,7,9]' full 100 40

# Remaining four matched settings on sphinx[4-11]. These preserve roughly
# two successful episodes per failed episode and 81-84% successful transitions.
submit_setting nlp sphinx 'sphinx[4-11]' n20f10 20 10
submit_setting nlp sphinx 'sphinx[4-11]' n15f8 15 8
submit_setting nlp sphinx 'sphinx[4-11]' n10f5 10 5
submit_setting nlp sphinx 'sphinx[4-11]' n5f3 5 3

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "Validated preprocessing plus all 18 training submissions; no jobs submitted."
  exit 0
fi

echo "Submitted all 18 Egg v4 training jobs: ${job_ids[*]}"
squeue -j "$(IFS=,; echo "${job_ids[*]}")" \
  -o "%.18i %.30j %.10P %.2t %.10M %R" || true
