#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=reconstruct_cabinet
#SBATCH --nodelist=iris9,iris10
#SBATCH --output=slurm/%j_reconstruct_cabinet.out

set -euo pipefail

cd /iris/u/ronpo/projects/new_mtql

eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"
conda activate cabinet-memory-sim

python scripts/reconstruct_cabinet_first_success.py \
  --input-dir=/iris/u/marcelto/mtql/cabinet-memory-sim/ManiSkill/cabinet_dataset_two_cams_more_rand \
  --output-dir=/iris/u/ronpo/projects/new_mtql/cabinet-memory-sim/ManiSkill/cabinet_dataset_two_cams_more_rand \
  --seed=42 \
  --val-fraction=0.1
