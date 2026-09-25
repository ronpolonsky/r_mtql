#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --array=0-9%6
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris8,iris9
#SBATCH --job-name=drawer-data-v1
#SBATCH --output=/iris/u/ronpo/projects/new_mtql/slurm/%A_%a_drawer_data_v1.out

set -euo pipefail

PROJECT_ROOT=/iris/u/ronpo/projects/new_mtql
OUTPUT_DIR="$PROJECT_ROOT/drawer_task/four_cube_memory_dataset_v1"

export MS_ASSET_DIR=/iris/u/marcelto/.maniskill
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/cache"

cd "$PROJECT_ROOT"
/iris/u/marcelto/miniconda3/envs/cabinet-memory-sim/bin/python -u \
  drawer_task/collect.py \
  --output-dir="$OUTPUT_DIR" \
  --shard-index="${SLURM_ARRAY_TASK_ID}" \
  --num-shards=10 \
  --successes=300 \
  --failures=50 \
  --cue-steps=10 \
  --max-steps=2000 \
  --step-penalty=-0.0001
