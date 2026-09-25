#!/usr/bin/env bash
# Build the v2 cache from the separate read-only-linked dataset view.
# The original /iris/u/ronpo/expo-ft-data/candy_scoop_v2 tree is never written.
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=prep-candy-v2-jit
#SBATCH --output=/iris/u/ronpo/projects/new_mtql_candy_scooping/slurm/%j.out

set -euo pipefail

ROOT=/iris/u/ronpo/projects/new_mtql_candy_scooping
PYTHON=/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python
DATASET=/iris/u/ronpo/expo-ft-data/candy_scoop_v2_jitted
STATS=/iris/u/ronpo/expo-ft-data/candy_scoop_norm_stats
CACHE=/iris/u/ronpo/mtql-runs/candy_scoop_v2_jitted_cache
TASK=/iris/u/ronpo/projects/expo-ft/configs/task/candy_scoop.py

cd "$ROOT"
export PYTHONPATH="$ROOT:/iris/u/ronpo/projects/expo-ft:${PYTHONPATH:-}"

exec "$PYTHON" -u scripts/preprocess_droid_cache.py \
  --dataset_path="$DATASET" \
  --norm_stats_path="$STATS" \
  --config_task="$TASK" \
  --cache_dir="$CACHE" \
  --image_size=224 \
  --n_succ=-1 \
  --n_fails=-1
