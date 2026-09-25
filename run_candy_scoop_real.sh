#!/usr/bin/env bash
#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --nodelist=sphinx[10-11]
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=candy-scoop-mtql
#SBATCH --output=/iris/u/ronpo/projects/new_mtql_candy_scooping/slurm/%j.out
cd "${SLURM_SUBMIT_DIR:-/iris/u/ronpo/projects/new_mtql_candy_scooping}"

set -euo pipefail

MTQL_ROOT="${SLURM_SUBMIT_DIR:-/iris/u/ronpo/projects/new_mtql_candy_scooping}"
export PYTHONPATH="/iris/u/ronpo/projects/expo-ft:${PYTHONPATH:-}"
PYTHON="${PYTHON:-/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python}"
DATASET_PATH="${DATASET_PATH:-/iris/u/ronpo/expo-ft-data/candy_scoop}"
DATASET_CACHE="${DATASET_CACHE:-}"
TASK_CONFIG="${TASK_CONFIG:-/iris/u/ronpo/projects/expo-ft/configs/task/candy_scoop.py}"
NORM_STATS_PATH="${NORM_STATS_PATH:-/iris/u/ronpo/expo-ft-data/candy_scoop_norm_stats}"
SAVE_DIR="${SAVE_DIR:-/iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue}"
AGENT_CONFIG="${AGENT_CONFIG:-agents/mtql_transformer_real.py}"
SEED="${SEED:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-30000}"
HIST_LENGTH="${HIST_LENGTH:-20}"
HIST_STRIDE="${HIST_STRIDE:-7}"
BATCH_SIZE="${BATCH_SIZE:-16}"
P_AUG="${P_AUG:-1.0}"
CUE_MODE="${CUE_MODE:-visual}"
PROJECT="${PROJECT:-real_candy_scoop}"
# Set AGENT_CONFIG=agents/mtql_mlp_real.py for the MLP-critic ablation.

# Training values are written explicitly in the command below, matching the
# simulator launchers. Edit those flags directly when defining a new run.


CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
RESTORE_PATH="${RESTORE_PATH:-}"
RESTORE_EPOCH="${RESTORE_EPOCH:-}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"
CHECKPOINT_KEEP_PERIOD="${CHECKPOINT_KEEP_PERIOD:-}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Training Python not found: ${PYTHON}" >&2
  exit 1
fi
"${PYTHON}" -c 'import jax; print("JAX devices:", jax.devices()); print("JAX backend:", jax.default_backend()); assert jax.default_backend() == "gpu", "JAX did not detect a GPU"'
if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "DROID dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi
if [[ -n "${DATASET_CACHE}" && ! -f "${DATASET_CACHE}/metadata.json" ]]; then
  echo "DROID dataset cache metadata not found: ${DATASET_CACHE}/metadata.json" >&2
  exit 1
fi
if [[ "${NORM_STATS_PATH}" == *.json ]]; then
  stats_file="${NORM_STATS_PATH}"
else
  stats_file="${NORM_STATS_PATH}/norm_stats.json"
fi
if [[ ! -r "${stats_file}" ]]; then
  echo "Normalization statistics not found: ${stats_file}" >&2
  echo "Generate them before training." >&2
  exit 1
fi
if [[ ! -f "${TASK_CONFIG}" ]]; then
  echo "Task config not found: ${TASK_CONFIG}" >&2
  exit 1
fi
if [[ -n "${RESTORE_PATH}" && -z "${RESTORE_EPOCH}" ]]; then
  echo "RESTORE_PATH and RESTORE_EPOCH must be set together for an exact restore." >&2
  exit 2
fi
if [[ -z "${RESTORE_PATH}" && -n "${RESTORE_EPOCH}" ]]; then
  echo "RESTORE_PATH and RESTORE_EPOCH must be set together for an exact restore." >&2
  exit 2
fi
if [[ "${RESUME}" == "1" && "${OVERWRITE}" == "1" ]]; then
  echo "RESUME=1 and OVERWRITE=1 are mutually exclusive." >&2
  exit 2
fi

args=(
  "${PYTHON}" -u "${MTQL_ROOT}/m_real.py"
  "--seed=${SEED}"
  "--save_dir=${SAVE_DIR}"
  "--dataset_path=${DATASET_PATH}"
  "--config_task=${TASK_CONFIG}"
  "--norm_stats_path=${NORM_STATS_PATH}"
  "--agent=${AGENT_CONFIG}"
  "--train_steps=${TRAIN_STEPS}"
  "--log_interval=${LOG_INTERVAL}"
  "--save_interval=${SAVE_INTERVAL}"
  "--hist_length=${HIST_LENGTH}"
  "--hist_stride=${HIST_STRIDE}"
  "--action_chunk_size=25"
  # DROID frames are 180x320; the real adapter pads/resizes them to OpenPI
  # square 224x224 input.
  "--image_size=224"
  # EXPO-FT augmentation is enabled by default; override with P_AUG=0.0 to disable.
  "--p_aug=${P_AUG}"
  "--cue_mode=${CUE_MODE}"
  "--n_succ=-1"
  "--n_fails=-1"
  "--agent.batch_size=${BATCH_SIZE}"
  "--agent.alpha=300"
  "--agent.critic_grad_clip=5.0"
  "--agent.normalize_q_loss=false"
  "--agent.hidden_dim=256"
  "--agent.tokenization_mode=per_modality"
  "--agent.encoder=resnet"
  "--agent.attention_entropy_target=((3.5,3.5),(3.0,3.0))"
  "--project=${PROJECT}"
  "--wandb_run_group=candy-scoop-real"
  "--wandb_mode=online"
  "--enable_wandb=1"
)

if [[ -n "${DATASET_CACHE}" ]]; then
  args+=("--dataset_cache=${DATASET_CACHE}")
fi

if [[ -n "${CHECKPOINT_DIR}" ]]; then
  args+=("--checkpoint_dir=${CHECKPOINT_DIR}")
fi
if [[ -n "${RESTORE_PATH}" ]]; then
  args+=("--restore_path=${RESTORE_PATH}" "--restore_epoch=${RESTORE_EPOCH}")
fi
if [[ "${RESUME}" == "1" ]]; then
  args+=("--resume")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  args+=("--overwrite")
fi
if [[ -n "${CHECKPOINT_KEEP_PERIOD}" ]]; then
  args+=("--checkpoint_keep_period=${CHECKPOINT_KEEP_PERIOD}")
fi

printf '%s\n' \
  "candy-scoop real MTQL: project=${PROJECT} agent=${AGENT_CONFIG} cue_mode=${CUE_MODE} cache=${DATASET_CACHE:-none} W&B=online seed=${SEED} train_steps=${TRAIN_STEPS} hist_length=${HIST_LENGTH} hist_stride=${HIST_STRIDE} action_chunk_size=25 p_aug=${P_AUG} log_interval=${LOG_INTERVAL} save_interval=${SAVE_INTERVAL} n_succ=-1 n_fails=-1 batch_size=${BATCH_SIZE} alpha=300 critic_grad_clip=5.0 attention_target=((3.5,3.5),(3.0,3.0))"
cd "${MTQL_ROOT}"
exec "${args[@]}"
