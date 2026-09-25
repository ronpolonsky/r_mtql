#!/bin/bash
# H20/S7 MLP-critic ablation matched to the counting-v4 transformer runs.
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=v4count-mlp-h20s7
#SBATCH --nodelist=iris5,iris6,iris7,iris9,iris10
#SBATCH --output=/iris/u/ronpo/projects/new_mtql/slurm/%A_%a.out
#SBATCH --array=1-3

set -euo pipefail

cd /iris/u/ronpo/projects/new_mtql

export MTQL_VARIANT=mlp
export HIST_LENGTH=20
export HIST_STRIDE=7
export NORMALIZE_Q_LOSS=false
export WANDB_PROJECT=final_offline-counting_v4

exec bash run_counting_scooping_v4.sh
