#!/bin/bash
#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=v4count-new-bc
#SBATCH --nodelist=sphinx[4-11]
#SBATCH --output=/iris/u/ronpo/projects/new_mtql/slurm/%A_%a.out
#SBATCH --array=1-3

set -euo pipefail

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export WANDB_ENTITY=new_mtql
export MS_ASSET_DIR=/iris/u/marcelto/.maniskill
export COUNTING_TASK_ENV_ID=TransferCountMemoryPanda-v0

# Pin the untouched original expert dataset. This launcher never reads either
# of the newer mixed success/failure datasets.
export COUNTING_DATASET=/iris/u/ronpo/projects/new_mtql/cabinet-memory-sim/ManiSkill/counting_dataset_scooping_v4

PROJECT_ROOT=/iris/u/ronpo/projects/new_mtql
HIST_LENGTH="${HIST_LENGTH:-20}"
HIST_STRIDE="${HIST_STRIDE:-7}"
SEED="${TRAIN_SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
BC_WANDB_PROJECT="${BC_WANDB_PROJECT:-new_offline-counting_v4}"
BC_WANDB_RUN_GROUP="${BC_WANDB_RUN_GROUP:-counting-scooping-v4-new-bc-flow-expert-only}"

if [[ ! -r "$COUNTING_DATASET/train_episodes.txt" || ! -r "$COUNTING_DATASET/val_counting_dataset.npz" ]]; then
  echo "Original counting-v4 dataset is incomplete under $COUNTING_DATASET" >&2
  exit 1
fi
if [[ ! -r "$COUNTING_DATASET/task_version.txt" ]] || [[ "$(<"$COUNTING_DATASET/task_version.txt")" != "assisted-scooping-easy-done-v4" ]]; then
  echo "Dataset under $COUNTING_DATASET is not counting v4; no training was started." >&2
  exit 1
fi

echo "new-expert-BC-v4 seed=$SEED hist_length=$HIST_LENGTH hist_stride=$HIST_STRIDE project=$BC_WANDB_PROJECT group=$BC_WANDB_RUN_GROUP"

eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"
conda activate cabinet-memory-sim

NVIDIA_SITE=/iris/u/marcelto/miniconda3/envs/cabinet-memory-sim/lib/python3.10/site-packages/nvidia
CUDNN8=/iris/u/ronpo/.local/cudnn8_for_jax/nvidia/cudnn/lib
export PATH="$NVIDIA_SITE/cuda_nvcc/bin:$PATH"
export LD_LIBRARY_PATH="$CUDNN8:$NVIDIA_SITE/cublas/lib:$NVIDIA_SITE/cuda_runtime/lib:$NVIDIA_SITE/cusolver/lib:$NVIDIA_SITE/cusparse/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ ! -r "$CUDNN8/libcudnn.so.8" ]]; then
  echo "Required cuDNN 8 library is not readable: $CUDNN8/libcudnn.so.8" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
python -c 'import jax; print("JAX devices:", jax.devices()); print("JAX backend:", jax.default_backend()); assert jax.default_backend() == "gpu", "JAX failed to initialize the GPU"'

# Purely offline behavior cloning with MTQL's actor_bc_flow architecture and
# flow-matching loss. There is no critic, Q loss, one-step actor, distillation,
# or online data collection.
python -u m_main.py \
  --seed="$SEED" \
  --env_name=counting \
  --train_steps=1500000 \
  --agent=agents/new_bc_flow_transformer.py \
  --agent.hidden_dim=256 \
  --agent.actor_num_layers=2 \
  --agent.actor_lr=1e-4 \
  --agent.actor_grad_clip=5.0 \
  --agent.optimizer=adamw \
  --agent.adamw_weight_decay=0.01 \
  --agent.warmup_steps=10000 \
  --agent.flow_steps=10 \
  --agent.encoder=resnet \
  --agent.batch_size=32 \
  --hist_length="${HIST_LENGTH}" \
  --hist_stride="${HIST_STRIDE}" \
  --action_chunk_size=25 \
  --action_exec_horizon=25 \
  --image_obs \
  --online_buf_size=0 \
  --num_cached_episodes=20 \
  --log_interval=1000000 \
  --eval_interval=50000 \
  --eval_episodes=60 \
  --eval_seed=20260828 \
  --num_eval_envs=1 \
  --video_episodes=3 \
  --video_frame_skip=2 \
  --save_interval=250000 \
  --project="${BC_WANDB_PROJECT}" \
  --wandb_run_group="${BC_WANDB_RUN_GROUP}" \
  --wandb_mode=online \
  --enable_wandb=1
