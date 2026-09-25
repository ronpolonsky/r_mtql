#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=collect_counting_scoop_v4
#SBATCH --nodelist=iris5,iris6,iris7,iris9,iris10
#SBATCH --output=/iris/u/ronpo/projects/new_mtql/slurm/counting_scooping_v4_%j.out

set -euo pipefail

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MS_ASSET_DIR=/iris/u/marcelto/.maniskill

eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"
conda activate cabinet-memory-sim

cd /iris/u/ronpo/projects/new_mtql/cabinet-memory-sim/ManiSkill

python -u -m counting_task.collect \
  --env-id=TransferCountMemoryPanda-v0 \
  --output-dir=counting_dataset_scooping_v4 \
  --seed=42 \
  --num-success=300 \
  --num-recovery=0 \
  --num-early-done=0 \
  --num-over-transfer=0 \
  --num-timeout=0 \
  --validation-fraction=0.1 \
  --image-size=128 \
  --max-control-steps=1200 \
  --sim-backend=cpu \
  --render-backend=gpu \
  --max-retries=5 \
  --resume
