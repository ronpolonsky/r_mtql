#!/bin/bash
#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=v4counting-scoop
#SBATCH --nodelist=sphinx[10-11]
#SBATCH --output=/iris/u/ronpo/projects/new_mtql/slurm/%A_%a.out
#SBATCH --array=2

set -euo pipefail

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
cd /iris/u/ronpo/projects/new_mtql
export WANDB_ENTITY="new_mtql"
WANDB_PROJECT="${WANDB_PROJECT:-new_offline-counting_v4}"
export MS_ASSET_DIR=/iris/u/marcelto/.maniskill
export COUNTING_TASK_ENV_ID=TransferCountMemoryPanda-v0
export COUNTING_DATASET=/iris/u/ronpo/projects/new_mtql/cabinet-memory-sim/ManiSkill/counting_dataset_scooping_v4

if [[ "${COUNTING_SWEEP:-0}" == "1" ]]; then
  SWEEP_INDEX="${SLURM_ARRAY_TASK_ID:?COUNTING_SWEEP requires a job array}"
  if (( SWEEP_INDEX < 0 || SWEEP_INDEX > 15 )); then
    echo "Sweep array index must be in [0, 15], got $SWEEP_INDEX" >&2
    exit 1
  fi
  HIST_LENGTHS=(0)
  HIST_STRIDES=(1)
  NORMALIZE_Q_VALUES=(true false)
  HIST_LENGTH="${HIST_LENGTHS[$((SWEEP_INDEX / 8))]}"
  HIST_STRIDE="${HIST_STRIDES[$(((SWEEP_INDEX / 2) % 4))]}"
  NORMALIZE_Q_LOSS="${NORMALIZE_Q_VALUES[$((SWEEP_INDEX % 2))]}"
else
  HIST_LENGTH="${HIST_LENGTH:-20}"
  HIST_STRIDE="${HIST_STRIDE:-7}"
  NORMALIZE_Q_LOSS="${NORMALIZE_Q_LOSS:-false}"
fi
SEED="${TRAIN_SEED:-${SLURM_ARRAY_TASK_ID:-0}}"

MTQL_VARIANT="${MTQL_VARIANT:-base}"
case "$MTQL_VARIANT" in
  base) AGENT_CONFIG=agents/mtql_transformer.py ;;
  mlp) AGENT_CONFIG=agents/mtql_mlp.py ;;
  v2) AGENT_CONFIG=agents/mtql_transformer_v2.py ;;
  *)
    echo "MTQL_VARIANT must be base, mlp, or v2, got: $MTQL_VARIANT" >&2
    exit 2
    ;;
esac

RESTORE_PATH="${RESTORE_PATH:-}"
RESTORE_EPOCH="${RESTORE_EPOCH:-}"
RESTORE_ARGS=()
if [[ -n "$RESTORE_PATH" || -n "$RESTORE_EPOCH" ]]; then
  if [[ -z "$RESTORE_PATH" || -z "$RESTORE_EPOCH" ]]; then
    echo "RESTORE_PATH and RESTORE_EPOCH must be set together" >&2
    exit 2
  fi
  if ! [[ "$RESTORE_EPOCH" =~ ^[0-9]+$ ]]; then
    echo "RESTORE_EPOCH must be a nonnegative integer, got: $RESTORE_EPOCH" >&2
    exit 2
  fi
  if [[ ! -r "$RESTORE_PATH/params_${RESTORE_EPOCH}.pkl" ]]; then
    echo "Checkpoint is not readable: $RESTORE_PATH/params_${RESTORE_EPOCH}.pkl" >&2
    exit 1
  fi
  RESTORE_ARGS=(
    "--restore_path=${RESTORE_PATH}"
    "--restore_epoch=${RESTORE_EPOCH}"
  )
fi

echo "v4 variant=$MTQL_VARIANT seed=$SEED hist_length=$HIST_LENGTH hist_stride=$HIST_STRIDE normalize_q_loss=$NORMALIZE_Q_LOSS restore_epoch=${RESTORE_EPOCH:-none}"

if [[ ! -r "$COUNTING_DATASET/train_counting_dataset.npz" || ! -r "$COUNTING_DATASET/val_counting_dataset.npz" ]]; then
  echo "Counting v4 train/validation datasets are missing under $COUNTING_DATASET" >&2
  echo "Collect them with collect_counting_scooping_v4.sh; no training was started." >&2
  exit 1
fi
if [[ ! -r "$COUNTING_DATASET/task_version.txt" ]] || [[ "$(<"$COUNTING_DATASET/task_version.txt")" != "assisted-scooping-easy-done-v4" ]]; then
  echo "Dataset under $COUNTING_DATASET is not counting v4; no training was started." >&2
  exit 1
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

python -u m_main.py \
  --seed="${SEED}" \
  --env_name=counting \
  --train_steps=1500000 \
  --agent="${AGENT_CONFIG}" \
  --agent.critic_grad_clip=5.0 \
  --agent.alpha=300 \
  --agent.attention_entropy_target="((3.5,3.5),(3.0,3.0))" \
  --agent.hidden_dim=256 \
  --hist_length="${HIST_LENGTH}" \
  --hist_stride="${HIST_STRIDE}" \
  --log_interval=1000000 \
  --agent.tokenization_mode=per_modality \
  --num_eval_envs=1 \
  --eval_interval=50000 \
  --eval_episodes=20 \
  --video_frame_skip=2 \
  --single_step_online \
  --online_warmup_steps=1500000 \
  --image_obs \
  --agent.batch_size=32 \
  --save_interval=25000 \
  --agent.encoder=resnet \
  --num_cached_episodes=20 \
  --video_episodes=2 \
  --action_chunk_size=25 \
  --action_exec_horizon=25 \
  --online_buf_size=500000 \
  --visualize_online \
  --use_env_reward \
  --project="${WANDB_PROJECT}" \
  --wandb_run_group=counting-scooping-v4 \
  --wandb_mode=online \
  --agent.normalize_q_loss="${NORMALIZE_Q_LOSS}" \
  --enable_wandb=1 \
  "${RESTORE_ARGS[@]}"
