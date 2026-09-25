#!/bin/bash
#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=mem_tql_transformer_v2
#SBATCH --nodelist=sphinx9
#SBATCH --output /iris/u/ronpo/projects/new_mtql/slurm/%A_%a.out
#SBATCH --array=2

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
cd /iris/u/ronpo/projects/new_mtql
export WANDB_ENTITY="new_mtql"
export MS_ASSET_DIR=/iris/u/marcelto/.maniskill
SEED="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SUCCESS_DEMOS="${1:-${NUM_SUCCESS_DEMOS:-25}}"
NUM_FAILURE_DEMOS="${2:-${NUM_FAILURE_DEMOS:-5}}"

if [[ $# -gt 2 ]] || ! [[ "$NUM_SUCCESS_DEMOS" =~ ^(-1|[0-9]+)$ ]] || ! [[ "$NUM_FAILURE_DEMOS" =~ ^(-1|[0-9]+)$ ]]; then
  echo "Usage: sbatch $0 [NUM_SUCCESS_DEMOS] [NUM_FAILURE_DEMOS]" >&2
  echo "Counts must be nonnegative integers, or -1 to keep all of that outcome." >&2
  exit 2
fi

eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"
conda activate cabinet-memory-sim
NVIDIA_SITE=/iris/u/marcelto/miniconda3/envs/cabinet-memory-sim/lib/python3.10/site-packages/nvidia
CUDNN8=/iris/u/ronpo/.local/cudnn8_for_jax/nvidia/cudnn/lib
export PATH=$NVIDIA_SITE/cuda_nvcc/bin:$PATH
export LD_LIBRARY_PATH=$CUDNN8:$NVIDIA_SITE/cublas/lib:$NVIDIA_SITE/cuda_runtime/lib:$NVIDIA_SITE/cusolver/lib:$NVIDIA_SITE/cusparse/lib:${LD_LIBRARY_PATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ ! -r "$CUDNN8/libcudnn.so.8" ]]; then
  echo "Required cuDNN 8 library is not readable: $CUDNN8/libcudnn.so.8" >&2
  exit 1
fi

python -c 'import jax; print("JAX devices:", jax.devices()); print("JAX backend:", jax.default_backend()); assert jax.default_backend() == "gpu", "JAX failed to initialize the GPU"'

# Select complete success/failure episodes in memory from the original cabinet
# training data. The source files are never modified.
python -u m_main.py \
  --seed="${SEED}" \
  --env_name=search_cabinet_two_cams_more_rand \
  --train_steps=1500000 \
  --agent=agents/mtql_transformer.py \
  --agent.critic_grad_clip=5.0 \
  --agent.alpha=300 \
  --agent.attention_entropy_target="((3.0, 3.0), (2.5, 2.5))" \
  --agent.hidden_dim=256 \
  --hist_length=20 \
  --hist_stride=50 \
  --log_interval=10000 \
  --agent.tokenization_mode=per_modality \
  --num_eval_envs=1 \
  --eval_interval=50000 \
  --eval_episodes=20 \
  --video_episodes=0 \
  --video_frame_skip=2 \
  --single_step_online \
  --online_warmup_steps=1500000 \
  --image_obs \
  --agent.batch_size=32 \
  --save_interval=1000000 \
  --agent.encoder=resnet \
  --num_cached_episodes 20 \
  --video_episodes 2 \
  --action_chunk_size 25 \
  --action_exec_horizon 25 \
  --online_buf_size 500000 \
  --nolazy_dataset \
  --num_success_demos="${NUM_SUCCESS_DEMOS}" \
  --num_failure_demos="${NUM_FAILURE_DEMOS}" \
  --visualize_online \
  --project=new-offline-cabinet_v2 \
  --wandb_run_group="cabinet-original-s${NUM_SUCCESS_DEMOS}-f${NUM_FAILURE_DEMOS}" \
  --wandb_mode=online \
  --agent.normalize_q_loss=true  \
  --enable_wandb=1
