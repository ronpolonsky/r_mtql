#!/usr/bin/env bash
# Isolated Egg-v4 MLP H19/S7 speed test with compact observations resident on GPU.
# Submit directly with: sbatch scripts/run_egg_v4_n50f25_h19s7_mlp_device_cache.sh
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --nodelist=iris[5,7,9]
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --job-name=egg-v4-gpu-mlp-n50f25-h19s7
#SBATCH --output=/iris/u/ronpo/projects/new_mtql_candy_scooping/slurm/%x-%j.out

set -euo pipefail

ROOT="${ROOT:-/iris/u/ronpo/projects/new_mtql_candy_scooping}"
PYTHON="${PYTHON:-/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python}"
DATASET_PATH="${DATASET_PATH:-/iris/u/ronpo/expo-ft-data/egg_v4}"
DATASET_CACHE="${DATASET_CACHE:-/iris/u/ronpo/mtql-runs/caches/egg_v4_224}"
NORM_STATS_PATH="${NORM_STATS_PATH:-/iris/u/ronpo/expo-ft-data/egg_v4_norm_stats}"
TASK_CONFIG="${TASK_CONFIG:-/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py}"
SAVE_ROOT="${SAVE_ROOT:-/iris/u/ronpo/mtql-runs/egg_v4/n50f25_h19s7/success_actor_device_cache/mlp}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-100}"
CHECKPOINT_KEEP_PERIOD="${CHECKPOINT_KEEP_PERIOD:-10000}"

[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 1; }
[[ -d "${DATASET_PATH}" ]] || { echo "Missing data: ${DATASET_PATH}" >&2; exit 1; }
[[ -s "${DATASET_CACHE}/metadata.json" ]] || { echo "Missing cache metadata" >&2; exit 1; }
[[ -s "${NORM_STATS_PATH}/norm_stats.json" ]] || { echo "Missing norm stats" >&2; exit 1; }
[[ -s "${TASK_CONFIG}" ]] || { echo "Missing task config" >&2; exit 1; }

export PYTHONPATH="${ROOT}:/iris/u/ronpo/projects/expo-ft:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${ROOT}"

printf '%s\n' \
  "trainer=m_real_egg_success_actor_device_cache.py" \
  "device_cache=compact observations + precomputed H19/S7 indices" \
  "data=egg_v4 n_success=50 n_failure=25" \
  "agent=mtql_mlp_success_actor_real batch=16 H=19 S=7 chunk=25" \
  "partition=${SLURM_JOB_PARTITION:-unknown} node=${SLURMD_NODENAME:-unknown}"

exec "${PYTHON}" -u "${ROOT}/m_real_egg_success_actor_device_cache.py" \
  --device_frame_cache=true \
  --seed=0 \
  --save_dir="${SAVE_ROOT}" \
  --dataset_path="${DATASET_PATH}" \
  --dataset_cache="${DATASET_CACHE}" \
  --dataset_kind=egg \
  --config_task="${TASK_CONFIG}" \
  --norm_stats_path="${NORM_STATS_PATH}" \
  --agent=agents/mtql_mlp_success_actor_real.py \
  --train_steps="${TRAIN_STEPS}" \
  --log_interval="${LOG_INTERVAL}" \
  --save_interval="${SAVE_INTERVAL}" \
  --checkpoint_max_to_keep="${CHECKPOINT_MAX_TO_KEEP}" \
  --checkpoint_keep_period="${CHECKPOINT_KEEP_PERIOD}" \
  --hist_length=19 \
  --hist_stride=7 \
  --action_chunk_size=25 \
  --image_size=224 \
  --p_aug=1.0 \
  --cue_mode=none \
  --n_succ=50 \
  --n_fails=25 \
  --agent.batch_size=16 \
  --agent.hidden_dim=256 \
  --agent.encoder=resnet \
  --agent.alpha=300 \
  --agent.actor_lr=1e-4 \
  --agent.critic_lr=1e-4 \
  --agent.critic_grad_clip=5.0 \
  --agent.normalize_q_loss=false \
  --agent.optimizer=adamw \
  --agent.adamw_weight_decay=0.01 \
  --agent.warmup_steps=10000 \
  --agent.num_layers=2 \
  --agent.actor_num_layers=2 \
  --agent.num_heads=8 \
  --agent.mlp_ratio=4 \
  --agent.dropout_rate=0.0 \
  --agent.flow_steps=10 \
  --agent.tokenization_mode=per_modality \
  '--agent.attention_entropy_target=((3.0,3.0),(2.5,2.5))' \
  --project=real_egg_v4_actor_success_only_device_cache \
  --wandb_run_group=egg_v4_n50f25_h19s7_success_actor_device_cache \
  --wandb_mode=online \
  --enable_wandb=1
