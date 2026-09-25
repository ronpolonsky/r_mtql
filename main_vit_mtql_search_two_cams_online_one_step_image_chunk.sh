#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=4 # Request 8 CPUs for this task
#SBATCH --mem=64G # Request 8GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name=mem_tql_transformer # Name the job (for easier monitoring)
#SBATCH --nodelist=iris8,iris9,iris10# Don't run on iris1
#SBATCH --output slurm/%j.out # MAKE SURE slurm/ ALREADY EXISTS, OR ELSE YOUR JOB WILL FAIL SILENTLY!

# Now your Python or general experiment/job runner code
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
cd /iris/u/marcelto/mtql
export HOME=/iris/u/marcelto
eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"
conda activate cabinet-memory-sim
NVIDIA_SITE=/iris/u/marcelto/miniconda3/envs/cabinet-memory-sim/lib/python3.10/site-packages/nvidia
CUDNN8=/iris/u/marcelto/.local/cudnn8_for_jax/nvidia/cudnn/lib
export PATH=$NVIDIA_SITE/cuda_nvcc/bin:$PATH
export LD_LIBRARY_PATH=$CUDNN8:$NVIDIA_SITE/cublas/lib:$NVIDIA_SITE/cuda_runtime/lib:$NVIDIA_SITE/cusolver/lib:$NVIDIA_SITE/cusparse/lib:${LD_LIBRARY_PATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false


# python collect_cabinet_dataset.py --num-episodes 200 --output-dir ./cabinet_dataset_2 --visualize --seed 42 --image-size 64 --num-expert 0 --num-noisy 250 

# python collect_cabinet_dataset.py --num-expert 0 --num-noisy 250 --output-dir ./cabinet_dataset_two_cams_more_rand --visualize --seed 42 --image-size 128 --randomize-cabinet-pose

python m_main.py --env_name=search_cabinet_two_cams_more_rand --train_steps=2000000 --agent=agents/mtql_transformer.py --agent.critic_grad_clip=5.0 --agent.alpha=300 --agent.attention_entropy_target="((4.5, 4.5), (4.0, 4.0))" --agent.hidden_dim=256 --hist_length=12 --log_interval=10 --agent.tokenization_mode=per_modality  --num_eval_envs=1 --eval_interval=10000 --eval_episodes=20 --video_episodes=0 --video_frame_skip=2 --hist_stride=50 --single_step_online --online_warmup_steps=100000 --image_obs --agent.batch_size=32 --save_interval=10000 --agent.encoder=mem_vit_small --num_cached_episodes 20 --video_episodes 2 --action_chunk_size 25 --action_exec_horizon 25 --online_buf_size 500000 --visualize_online --nolazy_dataset
