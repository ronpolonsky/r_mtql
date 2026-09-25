#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --job-name=check_jax_gpu
#SBATCH --nodelist=iris9,iris10
#SBATCH --output=slurm/check_jax_gpu_%j.out

set -euo pipefail

eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"
conda activate cabinet-memory-sim

NVIDIA_SITE=/iris/u/marcelto/miniconda3/envs/cabinet-memory-sim/lib/python3.10/site-packages/nvidia
CUDNN8=/iris/u/ronpo/.local/cudnn8_for_jax/nvidia/cudnn/lib

export PATH="$NVIDIA_SITE/cuda_nvcc/bin:$PATH"
export LD_LIBRARY_PATH="$CUDNN8:$NVIDIA_SITE/cublas/lib:$NVIDIA_SITE/cuda_runtime/lib:$NVIDIA_SITE/cusolver/lib:$NVIDIA_SITE/cusparse/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

test -r "$CUDNN8/libcudnn.so.8"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c 'import jax, jaxlib; print("jax:", jax.__version__); print("jaxlib:", jaxlib.__version__); print("JAX devices:", jax.devices()); print("JAX backend:", jax.default_backend()); assert jax.default_backend() == "gpu", "JAX failed to initialize the GPU"'
