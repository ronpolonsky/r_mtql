#!/bin/bash
#SBATCH --account=iris
#SBATCH --job-name=counting-v4-preview
#SBATCH --partition=iris
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/iris/u/ronpo/projects/new_mtql/slurm/counting_v4_preview_%j.out

set -euo pipefail

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MPLCONFIGDIR="/tmp/counting-v4-preview-${SLURM_JOB_ID}"
export PYTHONPATH="/iris/u/ronpo/projects/new_mtql/cabinet-memory-sim/ManiSkill:${PYTHONPATH:-}"

cd /iris/u/ronpo/projects/new_mtql/cabinet-memory-sim/ManiSkill

/iris/u/marcelto/miniconda3/envs/cabinet-memory-sim/bin/python -u \
  -m counting_task.render_previews \
  --env-id TransferCountMemoryPanda-v0 \
  --output-dir /iris/u/ronpo/projects/new_mtql/counting_task_previews \
  --success-only
