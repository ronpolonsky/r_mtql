#!/usr/bin/env bash
# Run batch sizes sequentially in one allocation and stop at the first
# failure. Submit once per GPU node when comparing devices.
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --nodelist=iris9,iris10
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=candy-batch-smoke
#SBATCH --output=/iris/u/ronpo/projects/new_mtql_candy_scooping/slurm/%j.out

set -u

# Slurm may execute a staged copy of this script under /var/lib/slurm. When
# run directly, prefer the current project so a stale SLURM_SUBMIT_DIR cannot
# select the old /new_mtql checkout.
if [[ -f "${PWD}/m_real.py" ]]; then
  ROOT="${PWD}"
else
  ROOT="${SMOKE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/iris/u/ronpo/projects/new_mtql_candy_scooping}}"
fi
PYTHON="${PYTHON:-/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python}"
DATASET="${DATASET_PATH:-/iris/u/ronpo/expo-ft-data/candy_scoop}"
DATASET_CACHE="${DATASET_CACHE:-}"
TASK="${TASK_CONFIG:-/iris/u/ronpo/projects/expo-ft/configs/task/candy_scoop.py}"
STATS="${NORM_STATS_PATH:-/iris/u/ronpo/expo-ft-data/candy_scoop_norm_stats}"
SMOKE_ROOT="${SMOKE_ROOT:-/iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/batch_smoke}"

# These defaults mirror run_candy_scoop_real.sh. Override them with --export.
AGENT_CONFIG="${AGENT_CONFIG:-agents/mtql_transformer_real.py}"
SEED="${SEED:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-2}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
HIST_LENGTH="${HIST_LENGTH:-20}"
HIST_STRIDE="${HIST_STRIDE:-7}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-25}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
P_AUG="${P_AUG:-1.0}"
CUE_MODE="${CUE_MODE:-visual}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-false}"
N_SUCC="${N_SUCC:--1}"
N_FAILS="${N_FAILS:--1}"
ALPHA="${ALPHA:-300}"
CRITIC_GRAD_CLIP="${CRITIC_GRAD_CLIP:-5.0}"
NORMALIZE_Q_LOSS="${NORMALIZE_Q_LOSS:-false}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
TOKENIZATION_MODE="${TOKENIZATION_MODE:-per_modality}"
ENCODER="${ENCODER:-resnet}"
ATTENTION_ENTROPY_TARGET="${ATTENTION_ENTROPY_TARGET:-((3.5,3.5),(3.0,3.0))}"

# Colon-separated because commas are separators in sbatch --export.
BATCH_SIZES="${BATCH_SIZES:-16:32:64:96:128}"
IFS=':' read -r -a BATCH_SIZE_VALUES <<< "${BATCH_SIZES}"

cd "${ROOT}"
mkdir -p "${SMOKE_ROOT}"

echo "host=$(hostname)"
echo "root=${ROOT}"
echo "entrypoint=${ROOT}/m_real.py"
echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "batch_sizes=${BATCH_SIZES} train_steps=${TRAIN_STEPS}"
echo "hist_length=${HIST_LENGTH} hist_stride=${HIST_STRIDE} action_chunk_size=${ACTION_CHUNK_SIZE}"
echo "image_size=${IMAGE_SIZE} p_aug=${P_AUG} n_succ=${N_SUCC} n_fails=${N_FAILS}"
echo "agent=${AGENT_CONFIG} cue_mode=${CUE_MODE} save_checkpoints=${SAVE_CHECKPOINTS} hidden_dim=${HIDDEN_DIM} encoder=${ENCODER}"
echo "dataset_cache=${DATASET_CACHE:-none}"

MAX_PASS="none"
OVERALL_STATUS=0
CACHE_ARGS=()
if [[ -n "${DATASET_CACHE}" ]]; then
  CACHE_ARGS+=("--dataset_cache=${DATASET_CACHE}")
fi

for BATCH_SIZE in "${BATCH_SIZE_VALUES[@]}"; do
  RUN_DIR="${SMOKE_ROOT}/$(hostname)/h${HIST_LENGTH}s${HIST_STRIDE}_b${BATCH_SIZE}"
  LOG_FILE="${RUN_DIR}/smoke.log"
  mkdir -p "${RUN_DIR}"
  echo "===== START batch_size=${BATCH_SIZE} =====" | tee "${LOG_FILE}"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | tee -a "${LOG_FILE}"

  set +e
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_PLATFORM_NAME=gpu \
  PYTHONPATH="/iris/u/ronpo/projects/expo-ft:${PYTHONPATH:-}" \
  "${PYTHON}" -u "${ROOT}/m_real.py" \
    --seed="${SEED}" \
    --enable_wandb=0 \
    --save_dir="${RUN_DIR}" \
    --dataset_path="${DATASET}" \
    --config_task="${TASK}" \
    --norm_stats_path="${STATS}" \
    --agent="${AGENT_CONFIG}" \
    --train_steps="${TRAIN_STEPS}" \
    --log_interval="${LOG_INTERVAL}" \
    --save_interval="${SAVE_INTERVAL}" \
    --hist_length="${HIST_LENGTH}" \
    --hist_stride="${HIST_STRIDE}" \
    --action_chunk_size="${ACTION_CHUNK_SIZE}" \
    --image_size="${IMAGE_SIZE}" \
    --p_aug="${P_AUG}" \
    --cue_mode="${CUE_MODE}" \
    "${CACHE_ARGS[@]}" \
    --save_checkpoints="${SAVE_CHECKPOINTS}" \
    --n_succ="${N_SUCC}" \
    --n_fails="${N_FAILS}" \
    --save_augmented_video="${RUN_DIR}/augmented_sample.mp4" \
    --agent.batch_size="${BATCH_SIZE}" \
    --agent.alpha="${ALPHA}" \
    --agent.critic_grad_clip="${CRITIC_GRAD_CLIP}" \
    --agent.normalize_q_loss="${NORMALIZE_Q_LOSS}" \
    --agent.hidden_dim="${HIDDEN_DIM}" \
    --agent.tokenization_mode="${TOKENIZATION_MODE}" \
    --agent.encoder="${ENCODER}" \
    --agent.attention_entropy_target="${ATTENTION_ENTROPY_TARGET}" \
    2>&1 | tee -a "${LOG_FILE}"
  STATUS=${PIPESTATUS[0]}
  set -u

  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | tee -a "${LOG_FILE}"
  if [[ ${STATUS} -eq 0 ]]; then
    echo "SMOKE PASS batch_size=${BATCH_SIZE}" | tee -a "${LOG_FILE}"
    MAX_PASS="${BATCH_SIZE}"
  else
    echo "SMOKE FAIL batch_size=${BATCH_SIZE} exit=${STATUS}" | tee -a "${LOG_FILE}"
    OVERALL_STATUS=${STATUS}
    break
  fi
done

echo "SMOKE SUMMARY largest_passing_batch=${MAX_PASS} status=${OVERALL_STATUS}"
exit "${OVERALL_STATUS}"
