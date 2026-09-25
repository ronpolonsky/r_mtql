#!/usr/bin/env bash
# Prepare egg_v4 on an iris-hi node: norm stats first, then the full cache.
# Submit from an Iris login shell with:
#   sbatch scripts/prepare_egg_v4_iris_hi.sbatch
#
# Existing outputs are protected by default. To intentionally rebuild them:
#   sbatch --export=ALL,OVERWRITE=1 scripts/prepare_egg_v4_iris_hi.sbatch
#SBATCH --account=iris
#SBATCH --partition=iris-hi
# Cache construction is CPU/I/O-heavy and does not require an HGX GPU.
# Keep the request broad so an ordinary iris-hi node can start it promptly.
#SBATCH --nodelist=iris[1-10]
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=egg-v4-prep
#SBATCH --output=/iris/u/ronpo/projects/new_mtql_candy_scooping/slurm/egg-v4-prep-%j.out

set -euo pipefail

ROOT="${EGG_PROJECT_ROOT:-/iris/u/ronpo/projects/new_mtql_candy_scooping}"
PYTHON="${PYTHON:-/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python}"

cd "${ROOT}"
mkdir -p "${ROOT}/slurm"

echo "egg_v4 preparation job ${SLURM_JOB_ID:-non-slurm}"
echo "node=$(hostname)"

PYTHON="${PYTHON}" \
  OVERWRITE="${OVERWRITE:-0}" \
  ./scripts/prepare_egg_v4.sh
