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


# python m_main.py --env_name=house-n3-v0 --train_steps=2000000 --agent=agents/mtql_transformer.py --agent.critic_grad_clip=5.0 --agent.alpha=300 --agent.attention_entropy_target="((3.0, 3.0), (2.5, 2.5))" --agent.hidden_dim=128 --hist_length=12 --log_interval=10 --agent.tokenization_mode=per_modality  --num_eval_envs=10 --eval_interval=20000 --eval_episodes=40 --video_episodes=6 --video_frame_skip=2 --hist_stride=5 --num_expert_episodes=0 --num_suboptimal_episodes=500 --single_step_online --online_warmup_steps=100000 --count_reward --small_obs --image_obs --agent.batch_size=128 --save_interval=50000 --env_randomization=medium --agent.encoder=resnet
# python m_main.py --env_name=search_cabinet --train_steps=2000000 --agent=agents/mtql_transformer.py --agent.critic_grad_clip=5.0 --agent.alpha=300 --agent.attention_entropy_target="((3.0, 3.0), (2.5, 2.5))" --agent.hidden_dim=128 --hist_length=12 --log_interval=10 --agent.tokenization_mode=per_modality  --num_eval_envs=1 --eval_interval=20000 --eval_episodes=40 --video_episodes=0 --video_frame_skip=2 --hist_stride=5 --num_expert_episodes=0 --num_suboptimal_episodes=500 --single_step_online --online_warmup_steps=100000 --count_reward --small_obs --image_obs --agent.batch_size=128 --save_interval=50000 --env_randomization=medium --agent.encoder=resnet --num_cached_episodes 20
# python collect_cabinet_dataset.py --num-episodes 200 --output-dir ./cabinet_dataset_noisy --visualize --seed 42 --num-expert 0 --num-noisy 10
# python collect_cabinet_dataset.py --num-episodes 200 --output-dir ./cabinet_dataset_two_cams --visualize --seed 42 --num-expert 0 --num-noisy 300 
# python collect_cabinet_dataset.py --num-episodes 200 --output-dir ./cabinet_dataset_low_res --visualize --seed 42 --image-size 64 --num-expert 0 --num-noisy 250 

python m_main.py --env_name=search_cabinet_two_cams_more_rand --train_steps=2000000 --agent=agents/bc_flow_transformer.py --agent.critic_grad_clip=5.0 --agent.alpha=300 --agent.attention_entropy_target="((3.8, 3.8), (3.3, 3.3))" --agent.hidden_dim=128 --hist_length=12 --log_interval=10 --agent.tokenization_mode=per_modality  --num_eval_envs=1 --eval_interval=20000 --eval_episodes=50 --video_episodes=0 --video_frame_skip=2 --hist_stride=50 --single_step_online --online_warmup_steps=1000000 --image_obs --agent.batch_size=64 --save_interval=10000 --agent.encoder=mem_vit_tokens_small --num_cached_episodes 20 --video_episodes 2 --action_chunk_size 25 --action_exec_horizon 25 --online_buf_size 50000 --visualize_online  --nolazy_dataset
