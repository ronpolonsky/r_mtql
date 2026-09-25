#!/bin/bash
# Matched offline-RL counterpart to W&B run qy36ayog.
# Fixed H24/S40 sweep: 3 entropy targets x 2 Q-normalization choices x 3 seeds.
#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=cabinet_hard_s25_f15_mtql
#SBATCH --nodelist=sphinx[4-11]
#SBATCH --output=/iris/u/ronpo/projects/new_mtql/slurm/%A_%a.out
#SBATCH --array=0-17

set -euo pipefail

cd /iris/u/ronpo/projects/new_mtql

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export WANDB_ENTITY="${WANDB_ENTITY:-new_mtql}"
export MS_ASSET_DIR=/iris/u/marcelto/.maniskill
export XLA_PYTHON_CLIENT_PREALLOCATE=false

DATASET_DIR=/iris/u/ronpo/projects/new_mtql/cabinet-memory-sim/ManiSkill/cabinet_dataset_two_cams_more_rand_hard_bc_s25_f15
if [[ ! -r "${DATASET_DIR}/train_cabinet_dataset.npz" || \
      ! -r "${DATASET_DIR}/val_cabinet_dataset.npz" || \
      ! -r "${DATASET_DIR}/manifest.json" ]]; then
  echo "The exact cabinet hard-s25-f15 dataset is incomplete: ${DATASET_DIR}" >&2
  exit 1
fi

eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"
conda activate cabinet-memory-sim

NVIDIA_SITE=/iris/u/marcelto/miniconda3/envs/cabinet-memory-sim/lib/python3.10/site-packages/nvidia
CUDNN8=/iris/u/ronpo/.local/cudnn8_for_jax/nvidia/cudnn/lib
export PATH="${NVIDIA_SITE}/cuda_nvcc/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDNN8}:${NVIDIA_SITE}/cublas/lib:${NVIDIA_SITE}/cuda_runtime/lib:${NVIDIA_SITE}/cusolver/lib:${NVIDIA_SITE}/cusparse/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -r "${CUDNN8}/libcudnn.so.8" ]]; then
  echo "Required cuDNN 8 library is not readable: ${CUDNN8}/libcudnn.so.8" >&2
  exit 1
fi

python -c 'import jax; print("JAX devices:", jax.devices()); print("JAX backend:", jax.default_backend()); assert jax.default_backend() == "gpu", "JAX failed to initialize the GPU"'

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID > 17 )); then
  echo "Array task ID must be in [0, 17], got ${TASK_ID}" >&2
  exit 2
fi

# Three adjacent task IDs are the three training seeds for one setting.
SEED=$((TASK_ID % 3))
SETTING_ID=$((TASK_ID / 3))
ENTROPY_ID=$((SETTING_ID / 2))
NORM_ID=$((SETTING_ID % 2))

case "${ENTROPY_ID}" in
  0)
    ENTROPY_LABEL=low_2p5_2p0
    ATTENTION_ENTROPY_TARGET='((2.5, 2.5), (2.0, 2.0))'
    ;;
  1)
    ENTROPY_LABEL=base_3p0_2p5
    ATTENTION_ENTROPY_TARGET='((3.0, 3.0), (2.5, 2.5))'
    ;;
  2)
    ENTROPY_LABEL=high_3p5_3p0
    ATTENTION_ENTROPY_TARGET='((3.5, 3.5), (3.0, 3.0))'
    ;;
esac

if (( NORM_ID == 0 )); then
  NORM_Q_LOSS=true
else
  NORM_Q_LOSS=false
fi

WANDB_PROJECT="${WANDB_PROJECT:-final-cabinet}"
WANDB_GROUP="cabinet-hard-s25-f15-h24-s40-ent-${ENTROPY_LABEL}-normq-${NORM_Q_LOSS}"
echo "Sweep task ${TASK_ID}: seed=${SEED}, entropy=${ATTENTION_ENTROPY_TARGET}, normalize_q_loss=${NORM_Q_LOSS}"

python -u m_main.py \
  --seed="${SEED}" \
  --env_name=search_cabinet_two_cams_more_rand_hard_bc_s25_f15_outcome_time \
  --train_steps=1500000 \
  --agent=agents/mtql_transformer.py \
  --agent.critic_grad_clip=5.0 \
  --agent.alpha=300 \
  --agent.discount=0.9995 \
  --agent.attention_entropy_target="${ATTENTION_ENTROPY_TARGET}" \
  --agent.hidden_dim=256 \
  --agent.actor_num_layers=2 \
  --agent.num_heads=8 \
  --agent.flow_steps=10 \
  --agent.actor_lr=0.0001 \
  --agent.adamw_weight_decay=0.01 \
  --hist_length=24 \
  --hist_stride=40 \
  --log_interval=10000 \
  --agent.tokenization_mode=per_modality \
  --num_eval_envs=1 \
  --eval_interval=50000 \
  --eval_episodes=20 \
  --eval_seed=10000 \
  --video_episodes=2 \
  --video_frame_skip=2 \
  --image_obs \
  --agent.batch_size=32 \
  --save_interval=250000 \
  --agent.encoder=resnet \
  --num_cached_episodes=40 \
  --action_chunk_size=25 \
  --action_exec_horizon=25 \
  --online_buf_size=1 \
  --nolazy_dataset \
  --num_success_demos=25 \
  --num_failure_demos=15 \
  --project="${WANDB_PROJECT}" \
  --wandb_run_group="${WANDB_GROUP}" \
  --wandb_mode=online \
  --agent.normalize_q_loss="${NORM_Q_LOSS}" \
  --enable_wandb=1
