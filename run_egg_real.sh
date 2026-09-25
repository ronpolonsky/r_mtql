#!/usr/bin/env bash
# Shared egg trainer for MTQL-transformer, MTQL-MLP, and flow-BC.
# Resource directives are safe direct-submit defaults; launchers may override them.
#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --nodelist=sphinx[4-11]
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=egg-train
#SBATCH --output=/iris/u/ronpo/projects/new_mtql_candy_scooping/slurm/%j.out

set -euo pipefail

ROOT="${EGG_PROJECT_ROOT:-/iris/u/ronpo/projects/new_mtql_candy_scooping}"
PYTHON="${PYTHON:-/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python}"
DATASET_PATH="${DATASET_PATH:-/iris/u/ronpo/expo-ft-data/egg_v1}"
DATASET_CACHE="${DATASET_CACHE:-/iris/u/ronpo/mtql-runs/caches/egg_v1_224}"
TASK_CONFIG="${TASK_CONFIG:-/iris/u/ronpo/projects/expo-ft/configs/task/egg.py}"
NORM_STATS_PATH="${NORM_STATS_PATH:-/iris/u/ronpo/expo-ft-data/egg_v1_norm_stats}"
SAVE_DIR="${SAVE_DIR:-/iris/u/ronpo/mtql-runs/egg_v1/debug}"
AGENT_CONFIG="${AGENT_CONFIG:-agents/mtql_transformer_real.py}"
SEED="${SEED:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-100}"
# Protect every periodic checkpoint from Orbax pruning by default.
CHECKPOINT_KEEP_PERIOD="${CHECKPOINT_KEEP_PERIOD:-${SAVE_INTERVAL}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
RESTORE_PATH="${RESTORE_PATH:-}"
RESTORE_EPOCH="${RESTORE_EPOCH:-}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"
HIST_LENGTH="${HIST_LENGTH:-25}"
HIST_STRIDE="${HIST_STRIDE:-7}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-25}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
P_AUG="${P_AUG:-1.0}"
N_SUCC="${N_SUCC:--1}"
N_FAILS="${N_FAILS:--1}"
ALPHA="${ALPHA:-300}"
NORMALIZE_Q_LOSS="${NORMALIZE_Q_LOSS:-false}"
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
ATTENTION_ENTROPY_LAYER0_CLS="${ATTENTION_ENTROPY_LAYER0_CLS:-3.5}"
ATTENTION_ENTROPY_LAYER0_OTHER="${ATTENTION_ENTROPY_LAYER0_OTHER:-3.5}"
ATTENTION_ENTROPY_LAYER1_CLS="${ATTENTION_ENTROPY_LAYER1_CLS:-3.0}"
ATTENTION_ENTROPY_LAYER1_OTHER="${ATTENTION_ENTROPY_LAYER1_OTHER:-3.0}"
if [[ -z "${ATTENTION_ENTROPY_TARGET:-}" ]]; then
  ATTENTION_ENTROPY_TARGET="((${ATTENTION_ENTROPY_LAYER0_CLS},${ATTENTION_ENTROPY_LAYER0_OTHER}),(${ATTENTION_ENTROPY_LAYER1_CLS},${ATTENTION_ENTROPY_LAYER1_OTHER}))"
fi
PROJECT="${PROJECT:-real_egg_v1}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-egg_v1}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Training Python not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "Egg dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_CACHE}/metadata.json" ]]; then
  echo "Completed egg cache not found: ${DATASET_CACHE}/metadata.json" >&2
  exit 1
fi
if [[ ! -r "${NORM_STATS_PATH}/norm_stats.json" ]]; then
  echo "Egg normalization statistics not found: ${NORM_STATS_PATH}/norm_stats.json" >&2
  exit 1
fi
if [[ ! -f "${TASK_CONFIG}" ]]; then
  echo "Egg task config not found: ${TASK_CONFIG}" >&2
  exit 1
fi
if [[ "${RESUME}" != 0 && "${RESUME}" != 1 ]]; then
  echo "RESUME must be 0 or 1, got: ${RESUME}" >&2
  exit 2
fi
if [[ "${OVERWRITE}" != 0 && "${OVERWRITE}" != 1 ]]; then
  echo "OVERWRITE must be 0 or 1, got: ${OVERWRITE}" >&2
  exit 2
fi
if [[ -n "${RESTORE_PATH}" && -z "${RESTORE_EPOCH}" ]]; then
  echo "RESTORE_PATH and RESTORE_EPOCH must be set together for an exact restore." >&2
  exit 2
fi
if [[ -z "${RESTORE_PATH}" && -n "${RESTORE_EPOCH}" ]]; then
  echo "RESTORE_PATH and RESTORE_EPOCH must be set together for an exact restore." >&2
  exit 2
fi
if [[ "${RESUME}" == 1 && "${OVERWRITE}" == 1 ]]; then
  echo "RESUME=1 and OVERWRITE=1 are mutually exclusive." >&2
  exit 2
fi

export PYTHONPATH="${ROOT}:/iris/u/ronpo/projects/expo-ft:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

"${PYTHON}" -c 'import jax; print("JAX devices:", jax.devices()); print("JAX backend:", jax.default_backend()); assert jax.default_backend() == "gpu", "JAX did not detect a GPU"'

case "${AGENT_CONFIG}" in
  agents/mtql_transformer_success_actor_real.py|agents/mtql_mlp_success_actor_real.py)
    TRAINER="${ROOT}/m_real_egg_success_actor.py"
    ;;
  *)
    TRAINER="${ROOT}/m_real_egg.py"
    ;;
esac

args=(
  "${PYTHON}" -u "${TRAINER}"
  "--seed=${SEED}"
  "--save_dir=${SAVE_DIR}"
  "--dataset_path=${DATASET_PATH}"
  "--dataset_cache=${DATASET_CACHE}"
  "--dataset_kind=egg"
  "--config_task=${TASK_CONFIG}"
  "--norm_stats_path=${NORM_STATS_PATH}"
  "--agent=${AGENT_CONFIG}"
  "--train_steps=${TRAIN_STEPS}"
  "--log_interval=${LOG_INTERVAL}"
  "--save_interval=${SAVE_INTERVAL}"
  "--checkpoint_max_to_keep=${CHECKPOINT_MAX_TO_KEEP}"
  "--hist_length=${HIST_LENGTH}"
  "--hist_stride=${HIST_STRIDE}"
  "--action_chunk_size=${ACTION_CHUNK_SIZE}"
  "--image_size=${IMAGE_SIZE}"
  "--p_aug=${P_AUG}"
  "--cue_mode=none"
  "--n_succ=${N_SUCC}"
  "--n_fails=${N_FAILS}"
  "--agent.batch_size=${BATCH_SIZE}"
  "--agent.hidden_dim=${HIDDEN_DIM}"
  "--agent.encoder=${ENCODER}"
  "--project=${PROJECT}"
  "--wandb_run_group=${WANDB_RUN_GROUP}"
  "--wandb_mode=online"
  "--enable_wandb=1"
)

if [[ -n "${CHECKPOINT_KEEP_PERIOD}" ]]; then
  args+=("--checkpoint_keep_period=${CHECKPOINT_KEEP_PERIOD}")
fi
if [[ -n "${CHECKPOINT_DIR}" ]]; then
  args+=("--checkpoint_dir=${CHECKPOINT_DIR}")
fi
if [[ -n "${RESTORE_PATH}" ]]; then
  args+=("--restore_path=${RESTORE_PATH}" "--restore_epoch=${RESTORE_EPOCH}")
fi
if [[ "${RESUME}" == 1 ]]; then
  args+=("--resume")
fi
if [[ "${OVERWRITE}" == 1 ]]; then
  args+=("--overwrite")
fi

case "${AGENT_CONFIG}" in
  agents/new_bc_flow_transformer_real.py)
    args+=(
      "--agent.actor_num_layers=${ACTOR_NUM_LAYERS}"
      "--agent.actor_lr=${ACTOR_LR}"
      "--agent.actor_grad_clip=5.0"
      "--agent.optimizer=${OPTIMIZER}"
      "--agent.adamw_weight_decay=${ADAMW_WEIGHT_DECAY}"
      "--agent.warmup_steps=${WARMUP_STEPS}"
      "--agent.flow_steps=${FLOW_STEPS}"
      "--agent.num_heads=${NUM_HEADS}"
      "--agent.mlp_ratio=${MLP_RATIO}"
      "--agent.dropout_rate=${DROPOUT_RATE}"
    )
    ;;
  agents/mtql_transformer_real.py|agents/mtql_mlp_real.py|agents/mtql_transformer_success_actor_real.py|agents/mtql_mlp_success_actor_real.py)
    args+=(
      "--agent.alpha=${ALPHA}"
      "--agent.actor_lr=${ACTOR_LR}"
      "--agent.critic_lr=${CRITIC_LR}"
      "--agent.critic_grad_clip=${CRITIC_GRAD_CLIP}"
      "--agent.normalize_q_loss=${NORMALIZE_Q_LOSS}"
      "--agent.optimizer=${OPTIMIZER}"
      "--agent.adamw_weight_decay=${ADAMW_WEIGHT_DECAY}"
      "--agent.warmup_steps=${WARMUP_STEPS}"
      "--agent.num_layers=${NUM_LAYERS}"
      "--agent.actor_num_layers=${ACTOR_NUM_LAYERS}"
      "--agent.num_heads=${NUM_HEADS}"
      "--agent.mlp_ratio=${MLP_RATIO}"
      "--agent.dropout_rate=${DROPOUT_RATE}"
      "--agent.flow_steps=${FLOW_STEPS}"
      "--agent.tokenization_mode=${TOKENIZATION_MODE}"
      "--agent.attention_entropy_target=${ATTENTION_ENTROPY_TARGET}"
    )
    ;;
  *)
    echo "Unsupported egg agent config: ${AGENT_CONFIG}" >&2
    exit 2
    ;;
esac

printf '%s\n' \
  "egg training: agent=${AGENT_CONFIG} H=${HIST_LENGTH} S=${HIST_STRIDE} seed=${SEED}" \
  "trainer=${TRAINER}" \
  "dataset=${DATASET_PATH}" \
  "cache=${DATASET_CACHE}" \
  "norm_stats=${NORM_STATS_PATH}" \
  "steps=${TRAIN_STEPS} batch=${BATCH_SIZE} H=${HIST_LENGTH} S=${HIST_STRIDE} chunk=${ACTION_CHUNK_SIZE} image=${IMAGE_SIZE} augmentation=${P_AUG}" \
  "checkpoint_every=${SAVE_INTERVAL} checkpoint_keep_period=${CHECKPOINT_KEEP_PERIOD} checkpoint_max_recent=${CHECKPOINT_MAX_TO_KEEP}" \
  "actor_lr=${ACTOR_LR} critic_lr=${CRITIC_LR} alpha=${ALPHA} normalize_q_loss=${NORMALIZE_Q_LOSS}" \
  "layers=${NUM_LAYERS} actor_layers=${ACTOR_NUM_LAYERS} heads=${NUM_HEADS} hidden=${HIDDEN_DIM} attention_entropy_target=${ATTENTION_ENTROPY_TARGET}"

cd "${ROOT}"
exec "${args[@]}"
