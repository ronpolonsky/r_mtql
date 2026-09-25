#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=120:00:00 
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4 
#SBATCH --mem=64G 
#SBATCH --gres=gpu:1 
#SBATCH --job-name=mem_tql_transformer 
#SBATCH --nodelist=iris5,iris6,iris7,iris9,iris10
#SBATCH --output /iris/u/ronpo/projects/new_mtql/slurm/%A_%a.out 
#SBATCH --array=1-3


# Now your Python or general experiment/job runner code
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
cd /iris/u/ronpo/projects/new_mtql
export WANDB_ENTITY="new_mtql"
export MS_ASSET_DIR=/iris/u/marcelto/.maniskill
SEED="${SLURM_ARRAY_TASK_ID:-0}"


eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"
conda activate cabinet-memory-sim
NVIDIA_SITE=/iris/u/marcelto/miniconda3/envs/cabinet-memory-sim/lib/python3.10/site-packages/nvidia
CUDNN8=/iris/u/ronpo/.local/cudnn8_for_jax/nvidia/cudnn/lib
export PATH=$NVIDIA_SITE/cuda_nvcc/bin:$PATH
export LD_LIBRARY_PATH=$CUDNN8:$NVIDIA_SITE/cublas/lib:$NVIDIA_SITE/cuda_runtime/lib:$NVIDIA_SITE/cusolver/lib:$NVIDIA_SITE/cusparse/lib:${LD_LIBRARY_PATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ ! -r "$CUDNN8/libcudnn.so.8" ]]; then
  echo "Required cuDNN 8 library is not readable: $CUDNN8/libcudnn.so.8" >&2
  exit 1
fi

python -c 'import jax; print("JAX devices:", jax.devices()); print("JAX backend:", jax.default_backend()); assert jax.default_backend() == "gpu", "JAX failed to initialize the GPU"'


# python m_main.py --env_name=house-n3-v0 --train_steps=2000000 --agent=agents/mtql_transformer.py --agent.critic_grad_clip=5.0 --agent.alpha=300 --agent.attention_entropy_target="((3.0, 3.0), (2.5, 2.5))" --agent.hidden_dim=128 --hist_length=12 --log_interval=10 --agent.tokenization_mode=per_modality  --num_eval_envs=10 --eval_interval=20000 --eval_episodes=40 --video_episodes=6 --video_frame_skip=2 --hist_stride=5 --num_expert_episodes=0 --num_suboptimal_episodes=500 --single_step_online --online_warmup_steps=100000 --count_reward --small_obs --image_obs --agent.batch_size=128 --save_interval=50000 --env_randomization=medium --agent.encoder=resnet
# python m_main.py --env_name=search_cabinet --train_steps=2000000 --agent=agents/mtql_transformer.py --agent.critic_grad_clip=5.0 --agent.alpha=300 --agent.attention_entropy_target="((3.0, 3.0), (2.5, 2.5))" --agent.hidden_dim=128 --hist_length=12 --log_interval=10 --agent.tokenization_mode=per_modality  --num_eval_envs=1 --eval_interval=20000 --eval_episodes=40 --video_episodes=0 --video_frame_skip=2 --hist_stride=5 --num_expert_episodes=0 --num_suboptimal_episodes=500 --single_step_online --online_warmup_steps=100000 --count_reward --small_obs --image_obs --agent.batch_size=128 --save_interval=50000 --env_randomization=medium --agent.encoder=resnet --num_cached_episodes 20
# python collect_cabinet_dataset.py --num-episodes 200 --output-dir ./cabinet_dataset_noisy --visualize --seed 42 --num-expert 0 --num-noisy 10
# python collect_cabinet_dataset.py --num-episodes 200 --output-dir ./cabinet_dataset_two_cams --visualize --seed 42 --num-expert 0 --num-noisy 300 
# python collect_cabinet_dataset.py --num-episodes 200 --output-dir ./cabinet_dataset_low_res --visualize --seed 42 --image-size 64 --num-expert 0 --num-noisy 250 


python -u m_main.py \
  --seed="${SEED}" \
  --env_name=search_cabinet_two_cams_more_rand \
  --train_steps=1500000 \
  --agent=agents/mtql_transformer.py \
  --agent.critic_grad_clip=5.0 \
  --agent.alpha=300 \
  --agent.attention_entropy_target="((3.0, 3.0), (2.5, 2.5))" \
  --agent.hidden_dim=256 \
  --hist_length=0 \
  --hist_stride=1 \
  --log_interval=10000 \
  --agent.tokenization_mode=per_modality \
  --num_eval_envs=1 \
  --eval_interval=50000 \
  --eval_episodes=20 \
  --video_episodes=0 \
  --video_frame_skip=2 \
  --single_step_online \
  --online_warmup_steps=1500000 \
  --image_obs \
  --agent.batch_size=32 \
  --save_interval=1000000 \
  --agent.encoder=resnet \
  --num_cached_episodes 20 \
  --video_episodes 2 \
  --action_chunk_size 25 \
  --action_exec_horizon 25 \
  --online_buf_size 500000 \
  --nolazy_dataset \
  --visualize_online \
  --project=new-offline-cabinet \
  --wandb_run_group=cabinet-success-fix \
  --wandb_mode=online \
  --agent.normalize_q_loss=true \
  --enable_wandb=1

# python m_main.py --env_name=search_cabinet_two_cams_more_rand --train_steps=2000000 --agent=agents/mtql_transformer.py --agent.critic_grad_clip=5.0 --agent.alpha=300 --agent.attention_entropy_target="((4.5, 4.5), (4.0, 4.0))" --agent.hidden_dim=256 --hist_length=12 --log_interval=10 --agent.tokenization_mode=per_modality  --num_eval_envs=1 --eval_interval=10000 --eval_episodes=10 --video_episodes=0 --video_frame_skip=2 --hist_stride=50 --single_step_online --online_warmup_steps=0 --image_obs --agent.batch_size=32 --save_interval=10000 --agent.encoder=resnet --num_cached_episodes 20 --video_episodes 2 --action_chunk_size 25 --action_exec_horizon 25 --online_buf_size 2000000 --visualize_online
