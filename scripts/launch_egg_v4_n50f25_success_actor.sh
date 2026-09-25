#!/usr/bin/env bash
# Submit the two Egg-v4 50-success / 25-failure separate-batch MTQL agents.
# Edit the experiment settings below, then run this file to submit both jobs.
set -euo pipefail

ROOT="${ROOT:-/iris/u/ronpo/projects/new_mtql_candy_scooping}"
PYTHON="${PYTHON:-/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python}"
DATASET_PATH="${DATASET_PATH:-/iris/u/ronpo/expo-ft-data/egg_v4}"
DATASET_CACHE="${DATASET_CACHE:-/iris/u/ronpo/mtql-runs/caches/egg_v4_224}"
NORM_STATS_PATH="${NORM_STATS_PATH:-/iris/u/ronpo/expo-ft-data/egg_v4_norm_stats}"
TASK_CONFIG="${TASK_CONFIG:-/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/run_egg_real.sh}"

# Experiment settings. Edit these values here before launching if desired.
N_SUCC="${N_SUCC:-50}"
N_FAILS="${N_FAILS:-25}"
SETTING="${SETTING:-n${N_SUCC}f${N_FAILS}}"
RUN_ROOT="${RUN_ROOT:-/iris/u/ronpo/mtql-runs/egg_v4/${SETTING}/success_actor}"
HIST_LENGTH="${HIST_LENGTH:-14}"
HIST_STRIDE="${HIST_STRIDE:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-25}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
P_AUG="${P_AUG:-1.0}"

ALPHA="${ALPHA:-300}"
NORMALIZE_Q_LOSS="${NORMALIZE_Q_LOSS:-false}"
ATTENTION_ENTROPY_LAYER0_CLS="${ATTENTION_ENTROPY_LAYER0_CLS:-3.0}"
ATTENTION_ENTROPY_LAYER0_OTHER="${ATTENTION_ENTROPY_LAYER0_OTHER:-3.0}"
ATTENTION_ENTROPY_LAYER1_CLS="${ATTENTION_ENTROPY_LAYER1_CLS:-2.5}"
ATTENTION_ENTROPY_LAYER1_OTHER="${ATTENTION_ENTROPY_LAYER1_OTHER:-2.5}"

ACTOR_LR="${ACTOR_LR:-1e-4}"
CRITIC_LR="${CRITIC_LR:-1e-4}"
CRITIC_GRAD_CLIP="${CRITIC_GRAD_CLIP:-5.0}"
OPTIMIZER="${OPTIMIZER:-adamw}"
ADAMW_WEIGHT_DECAY="${ADAMW_WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-10000}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-2}"
ACTOR_NUM_LAYERS="${ACTOR_NUM_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-8}"
MLP_RATIO="${MLP_RATIO:-4}"
DROPOUT_RATE="${DROPOUT_RATE:-0.0}"
FLOW_STEPS="${FLOW_STEPS:-10}"
TOKENIZATION_MODE="${TOKENIZATION_MODE:-per_modality}"
ENCODER="${ENCODER:-resnet}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-100}"
# Protect every periodic checkpoint from Orbax pruning by default.
CHECKPOINT_KEEP_PERIOD="${CHECKPOINT_KEEP_PERIOD:-${SAVE_INTERVAL}}"

PROJECT="${PROJECT:-real_egg_v4_actor_success_only}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-egg_v4_${SETTING}_success_actor}"
SEED="${SEED:-0}"

# Slurm settings.
ACCOUNT="${ACCOUNT:-iris}"
PARTITION="${PARTITION:-iris-hi}"
NODELIST="${NODELIST:-iris-hgx-[1-2]}"
TIME_LIMIT="${TIME_LIMIT:-168:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-128G}"

agents=(
  "transformer:agents/mtql_transformer_success_actor_real.py"
  "mlp:agents/mtql_mlp_success_actor_real.py"
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON}" ]] || die "Training Python is not executable: ${PYTHON}"
[[ -x "${TRAIN_SCRIPT}" ]] || die "Training wrapper is missing: ${TRAIN_SCRIPT}"
[[ -d "${DATASET_PATH}" ]] || die "Dataset is missing: ${DATASET_PATH}"
[[ -s "${DATASET_CACHE}/metadata.json" ]] \
  || die "Cache metadata is missing: ${DATASET_CACHE}/metadata.json"
[[ -s "${NORM_STATS_PATH}/norm_stats.json" ]] \
  || die "Normalization statistics are missing: ${NORM_STATS_PATH}/norm_stats.json"
[[ -f "${TASK_CONFIG}" ]] || die "Task config is missing: ${TASK_CONFIG}"

cd "${ROOT}"
mkdir -p "${ROOT}/slurm"

submitted_jobs=()
for entry in "${agents[@]}"; do
  IFS=: read -r label agent_config <<< "${entry}"
  job_name="egg-v4-sa-${label}-${SETTING}"
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

  printf '%-28s agent=%s\n' "${job_name}" "${agent_config}"
  printf '  data=%sS/%sF alpha=%s qnorm=%s entropy=((%s,%s),(%s,%s))\n' \
    "${N_SUCC}" "${N_FAILS}" "${ALPHA}" "${NORMALIZE_Q_LOSS}" \
    "${ATTENTION_ENTROPY_LAYER0_CLS}" \
    "${ATTENTION_ENTROPY_LAYER0_OTHER}" \
    "${ATTENTION_ENTROPY_LAYER1_CLS}" \
    "${ATTENTION_ENTROPY_LAYER1_OTHER}"
  printf '  actor_lr=%s critic_lr=%s batch=%s H=%s S=%s heads=%s layers=%s/%s\n' \
    "${ACTOR_LR}" "${CRITIC_LR}" "${BATCH_SIZE}" \
    "${HIST_LENGTH}" "${HIST_STRIDE}" "${NUM_HEADS}" \
    "${NUM_LAYERS}" "${ACTOR_NUM_LAYERS}"
  printf '  steps=%s checkpoint_every=%s keep_period=%s max_recent=%s nodes=%s\n' \
    "${TRAIN_STEPS}" "${SAVE_INTERVAL}" "${CHECKPOINT_KEEP_PERIOD}" \
    "${CHECKPOINT_MAX_TO_KEEP}" "${NODELIST}"

  submission=$("${command[@]}")
  job_id=${submission%%;*}
  submitted_jobs+=("${job_id}")
  printf '  submitted job=%s save_dir=%s\n' "${job_id}" "${save_dir}"
done

job_csv=$(IFS=,; echo "${submitted_jobs[*]}")
echo "Submitted jobs: ${submitted_jobs[*]}"
squeue -j "${job_csv}" -o '%.18i %.30j %.10P %.2t %.10M %R' || true
