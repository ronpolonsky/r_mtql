#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris8,iris9
#SBATCH --job-name=drawer-v1
#SBATCH --output=/iris/u/ronpo/projects/new_mtql/slurm/%j_drawer_train_v1.out

set -euo pipefail

PROJECT_ROOT=/iris/u/ronpo/projects/new_mtql
RUN_KIND="${RUN_KIND:-rl}"
MTQL_VARIANT="${MTQL_VARIANT:-base}"
SETTING_ID="${SETTING_ID:-h20_s100_a300_g999_n1}"
HIST_LENGTH="${HIST_LENGTH:-20}"
HIST_STRIDE="${HIST_STRIDE:-100}"
ALPHA="${ALPHA:-300}"
DISCOUNT="${DISCOUNT:-0.999}"
NORM_Q="${NORM_Q:-true}"
SEED="${TRAIN_SEED:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-1500000}"
NUM_SUCCESS_DEMOS="${NUM_SUCCESS_DEMOS:-25}"
NUM_FAILURE_DEMOS="${NUM_FAILURE_DEMOS:-15}"
RL_BATCH_SIZE="${RL_BATCH_SIZE:-8}"
BC_BATCH_SIZE="${BC_BATCH_SIZE:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-offline-drawer-task-v1-controlled}"

case "$HIST_LENGTH" in
  16) ENTROPY_TARGET='((3.3,3.3),(2.8,2.8))' ;;
  20) ENTROPY_TARGET='((3.5,3.5),(3.0,3.0))' ;;
  24) ENTROPY_TARGET='((3.7,3.7),(3.2,3.2))' ;;
  32) ENTROPY_TARGET='((4.0,4.0),(3.5,3.5))' ;;
  *) echo "Unsupported HIST_LENGTH=$HIST_LENGTH" >&2; exit 2 ;;
esac

export MS_ASSET_DIR=/iris/u/marcelto/.maniskill
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/cache"
export WANDB_ENTITY=new_mtql

eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"
conda activate cabinet-memory-sim
cd "$PROJECT_ROOT"

# The cabinet environment has a system Vulkan runtime, while this JAX build
# needs the conda CUDA libraries and the known-good local cuDNN 8 runtime.
# Keep both available; omitting these paths makes JAX silently fall back to CPU.
NVIDIA_SITE=/iris/u/marcelto/miniconda3/envs/cabinet-memory-sim/lib/python3.10/site-packages/nvidia
CUDNN8=/iris/u/ronpo/.local/cudnn8_for_jax/nvidia/cudnn/lib
export PATH="$NVIDIA_SITE/cuda_nvcc/bin:$PATH"
export LD_LIBRARY_PATH="$CUDNN8:$NVIDIA_SITE/cublas/lib:$NVIDIA_SITE/cuda_runtime/lib:$NVIDIA_SITE/cusolver/lib:$NVIDIA_SITE/cusparse/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -r "$CUDNN8/libcudnn.so.8" ]]; then
  echo "Required cuDNN 8 library is not readable: $CUDNN8/libcudnn.so.8" >&2
  exit 1
fi

nvidia-smi -L
python -c 'import jax; assert jax.default_backend() == "gpu", jax.devices(); print("JAX devices:", jax.devices())'

COMMON_ARGS=(
  --seed="$SEED"
  --env_name=drawer_task
  --train_steps="$TRAIN_STEPS"
  --hist_length="$HIST_LENGTH"
  --hist_stride="$HIST_STRIDE"
  --action_chunk_size=25
  --action_exec_horizon=25
  --image_obs
  --num_cached_episodes=20
  --chunk_reload_interval=1000
  --log_interval=10000
  --eval_interval=50000
  --eval_episodes=20
  --eval_seed=20260903
  --num_eval_envs=1
  --video_episodes=2
  --video_frame_skip=2
  --save_interval=250000
  --project="$WANDB_PROJECT"
  --wandb_run_group="drawer-v1-${SETTING_ID}"
  --wandb_mode=online
  --enable_wandb=1
)

if [[ "$RUN_KIND" == "bc" ]]; then
  python -u m_main.py \
    "${COMMON_ARGS[@]}" \
    --agent=agents/new_bc_flow_transformer.py \
    --agent.hidden_dim=256 \
    --agent.actor_num_layers=2 \
    --agent.num_heads=4 \
    --agent.actor_lr=1e-4 \
    --agent.actor_grad_clip=5.0 \
    --agent.optimizer=adamw \
    --agent.adamw_weight_decay=0.01 \
    --agent.warmup_steps=10000 \
    --agent.flow_steps=10 \
    --agent.encoder=resnet \
    --agent.batch_size="$BC_BATCH_SIZE" \
    --online_buf_size=0 \
    --num_success_demos="$NUM_SUCCESS_DEMOS" \
    --num_failure_demos=0 \
    --successful_demos_only
elif [[ "$RUN_KIND" == "rl" ]]; then
  case "$MTQL_VARIANT" in
    base) RL_AGENT=agents/mtql_transformer.py ;;
    v2) RL_AGENT=agents/mtql_transformer_v2.py ;;
    *) echo "MTQL_VARIANT must be base or v2, got $MTQL_VARIANT" >&2; exit 2 ;;
  esac
  python -u m_main.py \
    "${COMMON_ARGS[@]}" \
    --agent="$RL_AGENT" \
    --agent.critic_grad_clip=5.0 \
    --agent.alpha="$ALPHA" \
    --agent.discount="$DISCOUNT" \
    --agent.attention_entropy_target="$ENTROPY_TARGET" \
    --agent.hidden_dim=256 \
    --agent.actor_num_layers=2 \
    --agent.num_heads=4 \
    --agent.actor_lr=1e-4 \
    --agent.optimizer=adamw \
    --agent.adamw_weight_decay=0.01 \
    --agent.warmup_steps=10000 \
    --agent.flow_steps=10 \
    --agent.encoder=resnet \
    --agent.batch_size="$RL_BATCH_SIZE" \
    --agent.tokenization_mode=per_modality \
    --agent.normalize_q_loss="$NORM_Q" \
    --num_success_demos="$NUM_SUCCESS_DEMOS" \
    --num_failure_demos="$NUM_FAILURE_DEMOS" \
    --online_buf_size=1
else
  echo "RUN_KIND must be rl or bc, got $RUN_KIND" >&2
  exit 2
fi
