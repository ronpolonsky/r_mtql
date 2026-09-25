#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=new_bc_flow
#SBATCH --nodelist=iris5,iris6,iris7,iris9,iris10
#SBATCH --output=/iris/u/ronpo/projects/new_mtql/slurm/%A_%a.out
#SBATCH --array=1

set -euo pipefail

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export WANDB_ENTITY=new_mtql
export MS_ASSET_DIR=/iris/u/marcelto/.maniskill
export XLA_PYTHON_CLIENT_PREALLOCATE=false

PROJECT_ROOT=/iris/u/ronpo/projects/new_mtql
SEED="${SLURM_ARRAY_TASK_ID:-0}"
MAX_DEMOS="${MAX_DEMOS:-0}"

eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"
conda activate cabinet-memory-sim

NVIDIA_SITE=/iris/u/marcelto/miniconda3/envs/cabinet-memory-sim/lib/python3.10/site-packages/nvidia
CUDNN8=/iris/u/ronpo/.local/cudnn8_for_jax/nvidia/cudnn/lib
export PATH="$NVIDIA_SITE/cuda_nvcc/bin:$PATH"
export LD_LIBRARY_PATH="$CUDNN8:$NVIDIA_SITE/cublas/lib:$NVIDIA_SITE/cuda_runtime/lib:$NVIDIA_SITE/cusolver/lib:$NVIDIA_SITE/cusparse/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -r "$CUDNN8/libcudnn.so.8" ]]; then
  echo "Required cuDNN 8 library is not readable: $CUDNN8/libcudnn.so.8" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
python -c 'import jax; print("JAX devices:", jax.devices()); print("JAX backend:", jax.default_backend()); assert jax.default_backend() == "gpu", "JAX failed to initialize the GPU"'

# Success-only offline flow-matching BC baseline. Complete successful episodes
# are filtered in memory from the same cabinet dataset; the source files remain
# unchanged. The policy uses MTQL's actor_bc_flow architecture, loss, and flow
# integration without a critic, Q loss, online collection, or one-step actor.
python -u m_main.py \
  --seed="$SEED" \
  --env_name=search_cabinet_two_cams_more_rand \
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
  --hist_length=12 \
  --hist_stride=50 \
  --action_chunk_size=25 \
  --action_exec_horizon=25 \
  --image_obs \
  --nolazy_dataset \
  --successful_demos_only \
  --max_demos="$MAX_DEMOS" \
  --num_cached_episodes=20 \
  --log_interval=10000 \
  --eval_interval=50000 \
  --eval_episodes=20 \
  --num_eval_envs=1 \
  --video_episodes=2 \
  --video_frame_skip=2 \
  --save_interval=1000000 \
  --project=bc-cabinet \
  --wandb_run_group=cabinet-new-bc-flow-success-only-h12-s50 \
  --wandb_mode=online \
  --enable_wandb=1
