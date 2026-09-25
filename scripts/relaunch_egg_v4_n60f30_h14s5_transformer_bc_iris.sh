#!/usr/bin/env bash
# Relaunch only the missing Egg-v4 n60f30 H14/S5 transformer and BC jobs.
# The matched MLP job is intentionally omitted because it is already running.
set -euo pipefail

ROOT="${ROOT:-/iris/u/ronpo/projects/new_mtql_candy_scooping}"
PYTHON="${PYTHON:-/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/run_egg_real.sh}"
DATASET_PATH="${DATASET_PATH:-/iris/u/ronpo/expo-ft-data/egg_v4}"
DATASET_CACHE="${DATASET_CACHE:-/iris/u/ronpo/mtql-runs/caches/egg_v4_224}"
NORM_STATS_PATH="${NORM_STATS_PATH:-/iris/u/ronpo/expo-ft-data/egg_v4_norm_stats}"
TASK_CONFIG="${TASK_CONFIG:-/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py}"

SETTING="n60f30_h14s5"
RUN_ROOT="${RUN_ROOT:-/iris/u/ronpo/mtql-runs/egg_v4/${SETTING}/success_actor_matched}"
PROJECT="${PROJECT:-real_egg_v4_actor_success_only}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-egg_v4_${SETTING}_success_actor_matched}"

ACCOUNT="${ACCOUNT:-iris}"
PARTITION="${PARTITION:-iris}"
NODELIST="${NODELIST:-iris10}"
TIME_LIMIT="${TIME_LIMIT:-168:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-128G}"
DRY_RUN="${DRY_RUN:-0}"

N_SUCC=60
N_FAILS=30
HIST_LENGTH=14
HIST_STRIDE=5
BATCH_SIZE=16
ACTION_CHUNK_SIZE=25
IMAGE_SIZE=224
P_AUG=1.0
SEED=0

TRAIN_STEPS=1000000
LOG_INTERVAL=10000
SAVE_INTERVAL=10000
CHECKPOINT_MAX_TO_KEEP=100
CHECKPOINT_KEEP_PERIOD=10000

ALPHA=300
NORMALIZE_Q_LOSS=false
ACTOR_LR=1e-4
CRITIC_LR=1e-4
CRITIC_GRAD_CLIP=5.0
OPTIMIZER=adamw
ADAMW_WEIGHT_DECAY=0.01
WARMUP_STEPS=10000
HIDDEN_DIM=256
NUM_LAYERS=2
ACTOR_NUM_LAYERS=2
NUM_HEADS=8
MLP_RATIO=4
DROPOUT_RATE=0.0
FLOW_STEPS=10
TOKENIZATION_MODE=per_modality
ENCODER=resnet
ATTENTION_ENTROPY_LAYER0_CLS=3.0
ATTENTION_ENTROPY_LAYER0_OTHER=3.0
ATTENTION_ENTROPY_LAYER1_CLS=2.5
ATTENTION_ENTROPY_LAYER1_OTHER=2.5

agents=(
  "transformer:agents/mtql_transformer_success_actor_real.py"
  "bc:agents/new_bc_flow_transformer_real.py"
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] \
  || die "DRY_RUN must be 0 or 1"
[[ -x "${PYTHON}" ]] || die "Python is not executable: ${PYTHON}"
[[ -x "${TRAIN_SCRIPT}" ]] || die "Training wrapper is missing: ${TRAIN_SCRIPT}"
[[ -d "${DATASET_PATH}" ]] || die "Dataset is missing: ${DATASET_PATH}"
[[ -s "${DATASET_CACHE}/metadata.json" ]] || die "Cache metadata is missing: ${DATASET_CACHE}/metadata.json"
[[ -s "${NORM_STATS_PATH}/norm_stats.json" ]] || die "Normalization stats are missing: ${NORM_STATS_PATH}/norm_stats.json"
[[ -s "${TASK_CONFIG}" ]] || die "Task config is missing: ${TASK_CONFIG}"

cd "${ROOT}"
mkdir -p "${ROOT}/slurm"

job_ids=()
for entry in "${agents[@]}"; do
  IFS=: read -r label agent_config <<< "${entry}"
  job_name="egg-v4-${label}-n60f30-h14s5"
  save_dir="${RUN_ROOT}/${label}"

  exports="ALL,PYTHON=${PYTHON},DATASET_PATH=${DATASET_PATH}"
  exports+=",DATASET_CACHE=${DATASET_CACHE},NORM_STATS_PATH=${NORM_STATS_PATH}"
  exports+=",TASK_CONFIG=${TASK_CONFIG},PROJECT=${PROJECT}"
  exports+=",WANDB_RUN_GROUP=${WANDB_RUN_GROUP},SEED=${SEED}"
  exports+=",N_SUCC=${N_SUCC},N_FAILS=${N_FAILS}"
  exports+=",HIST_LENGTH=${HIST_LENGTH},HIST_STRIDE=${HIST_STRIDE}"
  exports+=",BATCH_SIZE=${BATCH_SIZE},ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE}"
  exports+=",IMAGE_SIZE=${IMAGE_SIZE},P_AUG=${P_AUG}"
  exports+=",ALPHA=${ALPHA},NORMALIZE_Q_LOSS=${NORMALIZE_Q_LOSS}"
  exports+=",ACTOR_LR=${ACTOR_LR},CRITIC_LR=${CRITIC_LR}"
  exports+=",CRITIC_GRAD_CLIP=${CRITIC_GRAD_CLIP},OPTIMIZER=${OPTIMIZER}"
  exports+=",ADAMW_WEIGHT_DECAY=${ADAMW_WEIGHT_DECAY},WARMUP_STEPS=${WARMUP_STEPS}"
  exports+=",HIDDEN_DIM=${HIDDEN_DIM},NUM_LAYERS=${NUM_LAYERS}"
  exports+=",ACTOR_NUM_LAYERS=${ACTOR_NUM_LAYERS},NUM_HEADS=${NUM_HEADS}"
  exports+=",MLP_RATIO=${MLP_RATIO},DROPOUT_RATE=${DROPOUT_RATE}"
  exports+=",FLOW_STEPS=${FLOW_STEPS},TOKENIZATION_MODE=${TOKENIZATION_MODE}"
  exports+=",ENCODER=${ENCODER}"
  exports+=",ATTENTION_ENTROPY_LAYER0_CLS=${ATTENTION_ENTROPY_LAYER0_CLS}"
  exports+=",ATTENTION_ENTROPY_LAYER0_OTHER=${ATTENTION_ENTROPY_LAYER0_OTHER}"
  exports+=",ATTENTION_ENTROPY_LAYER1_CLS=${ATTENTION_ENTROPY_LAYER1_CLS}"
  exports+=",ATTENTION_ENTROPY_LAYER1_OTHER=${ATTENTION_ENTROPY_LAYER1_OTHER}"
  exports+=",TRAIN_STEPS=${TRAIN_STEPS},LOG_INTERVAL=${LOG_INTERVAL}"
  exports+=",SAVE_INTERVAL=${SAVE_INTERVAL}"
  exports+=",CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP}"
  exports+=",CHECKPOINT_KEEP_PERIOD=${CHECKPOINT_KEEP_PERIOD}"
  exports+=",AGENT_CONFIG=${agent_config},SAVE_DIR=${save_dir}"

  command=(
    sbatch --parsable
    --account="${ACCOUNT}" --partition="${PARTITION}"
    --nodelist="${NODELIST}" --nodes=1 --gres=gpu:1
    --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY}"
    --time="${TIME_LIMIT}" --job-name="${job_name}"
    --output="${ROOT}/slurm/%x-%j.out"
    --export="${exports}" "${TRAIN_SCRIPT}"
  )

  printf '%s: agent=%s data=%sS/%sF H=%s S=%s nodes=%s partition=%s\n' \
    "${job_name}" "${agent_config}" "${N_SUCC}" "${N_FAILS}" \
    "${HIST_LENGTH}" "${HIST_STRIDE}" "${NODELIST}" "${PARTITION}"

  if [[ "${DRY_RUN}" == 1 ]]; then
    printf '  DRY RUN:'
    printf ' %q' "${command[@]}"
    printf '\n'
    continue
  fi

  submission=$("${command[@]}")
  job_id=${submission%%;*}
  job_ids+=("${job_id}")
  printf '  submitted job=%s save_dir=%s\n' "${job_id}" "${save_dir}"
done

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "Validated both submissions; no jobs submitted."
  exit 0
fi

job_csv=$(IFS=,; echo "${job_ids[*]}")
echo "Submitted n60f30 H14/S5 transformer+BC jobs: ${job_ids[*]}"
squeue -j "${job_csv}" -o '%.18i %.38j %.10P %.9T %.10M %R' || true
