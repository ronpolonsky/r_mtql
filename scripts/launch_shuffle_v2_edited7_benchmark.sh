#!/usr/bin/env bash
# Launch one matched Shuffle edited7 benchmark member.
#
# Examples:
#   METHOD=transformer H=14 S=6 SEED=0 sbatch scripts/launch_shuffle_v2_edited7_benchmark.sh
#   METHOD=mlp         H=14 S=6 SEED=0 sbatch scripts/launch_shuffle_v2_edited7_benchmark.sh
#   METHOD=bc          H=14 S=6 SEED=0 sbatch scripts/launch_shuffle_v2_edited7_benchmark.sh
#
# The default history H14/S6 is the matched Shuffle scaling-benchmark setting.
#SBATCH --account=iris
#SBATCH --partition=iris-hi
# iris1 has intermittently exposed no CUDA device despite a GPU allocation;
# leave it out of this benchmark launcher.
#SBATCH --nodelist=iris[2-10]
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=21-00:00:00
#SBATCH --output=/iris/u/ronpo/projects/new_mtql_candy_scooping/slurm/shuffle-e7-%x-%j.out

set -euo pipefail

ROOT=/iris/u/ronpo/projects/new_mtql_candy_scooping
PYTHON=/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python
METHOD=${METHOD:-transformer}
H=${H:-14}
S=${S:-6}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-16}
TRAIN_STEPS=${TRAIN_STEPS:-100000}
LOG_INTERVAL=${LOG_INTERVAL:-10000}
SAVE_INTERVAL=${SAVE_INTERVAL:-30000}
CHECKPOINT_MAX_TO_KEEP=${CHECKPOINT_MAX_TO_KEEP:-100}
CHECKPOINT_KEEP_PERIOD=${CHECKPOINT_KEEP_PERIOD:-}
RESUME=${RESUME:-1}
N_SUCC=${N_SUCC:--1}
N_FAILS=${N_FAILS:--1}
NORMALIZE_Q_LOSS=${NORMALIZE_Q_LOSS:-false}
# This is a two-layer target: (layer 0 cls/other, layer 1 cls/other).
ATTENTION_ENTROPY_TARGET=${ATTENTION_ENTROPY_TARGET:-"((3.0,3.0),(2.5,2.5))"}
ENABLE_WANDB=${ENABLE_WANDB:-1}
WANDB_MODE=${WANDB_MODE:-online}
PROJECT=${PROJECT:-shuffle_v2_edited7_benchmark}
DRY_RUN=${DRY_RUN:-0}

case "${METHOD}" in
  transformer)
    ENTRY=m_real_shuffle_edited7_success_actor_device_cache.py
    AGENT=agents/mtql_transformer_success_actor_real.py
    TAG=transformer
    RL_AGENT=1
    ;;
  mlp)
    ENTRY=m_real_shuffle_edited7_success_actor_device_cache.py
    AGENT=agents/mtql_mlp_success_actor_real.py
    TAG=mlp
    RL_AGENT=1
    ;;
  bc)
    ENTRY=m_real_shuffle_edited7_bc_device_cache.py
    AGENT=agents/new_bc_flow_transformer_real.py
    TAG=bc
    RL_AGENT=0
    ;;
  *) echo "METHOD must be transformer, mlp, or bc" >&2; exit 2 ;;
esac

WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-shuffle_v2_edited7_${TAG}_h${H}s${S}}

DATASET=/iris/u/ronpo/mtql-runs/datasets/shuffle_v2_edited7_view
CACHE=/iris/u/ronpo/mtql-runs/caches/shuffle_v2_edited7_224
NORM=/iris/u/ronpo/expo-ft-data/shuffle_v2_edited7_norm_stats
TASK=/iris/u/ronpo/projects/expo-ft/configs/task/egg_v2.py
RUN=${RUN:-/iris/u/ronpo/mtql-runs/shuffle_v2_edited7/benchmark/${TAG}_s${N_SUCC}f${N_FAILS}_h${H}s${S}_seed${SEED}}

cd "${ROOT}"
mkdir -p "${ROOT}/slurm"
# wandb/log_utils uses tempfile.mkdtemp() during import.  The cluster
# environment may point TMPDIR at a per-user directory that is not created
# yet, so make it explicit before Python starts.
mkdir -p "${TMPDIR:-/tmp}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}

AGENT_OVERRIDES=()
if [[ "${RL_AGENT}" == 1 ]]; then
  AGENT_OVERRIDES+=(
    "--agent.attention_entropy_target=${ATTENTION_ENTROPY_TARGET}"
    "--agent.normalize_q_loss=${NORMALIZE_Q_LOSS}"
  )
fi

command=(
  "${PYTHON}" -u "${ENTRY}"
  --agent="${AGENT}" \
  --dataset_kind=egg \
  --dataset_path="${DATASET}" \
  --dataset_cache="${CACHE}" \
  --norm_stats_path="${NORM}" \
  --config_task="${TASK}" \
  --cue_mode=none \
  --cue_frames=7 \
  --hist_length="${H}" \
  --hist_stride="${S}" \
  --action_chunk_size=25 \
  --image_size=224 \
  --agent.batch_size="${BATCH_SIZE}" \
  --seed="${SEED}" \
  --train_steps="${TRAIN_STEPS}" \
  --log_interval="${LOG_INTERVAL}" \
  --save_interval="${SAVE_INTERVAL}" \
  --resume="${RESUME}" \
  --checkpoint_max_to_keep="${CHECKPOINT_MAX_TO_KEEP}" \
  --save_dir="${RUN}" \
  --project="${PROJECT}" \
  --wandb_run_group="${WANDB_RUN_GROUP}" \
  --n_succ="${N_SUCC}" \
  --n_fails="${N_FAILS}" \
  --p_aug=1.0 \
  --enable_wandb="${ENABLE_WANDB}" \
  --wandb_mode="${WANDB_MODE}"
)
command+=("${AGENT_OVERRIDES[@]}")

if [[ "${DRY_RUN}" == 1 ]]; then
  printf 'DRY RUN:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}"
