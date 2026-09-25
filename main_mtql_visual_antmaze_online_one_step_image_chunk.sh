#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=4 # Request 8 CPUs for this task
#SBATCH --mem=128G # Request 8GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name=mem_tql_transformer # Name the job (for easier monitoring)
#SBATCH --nodelist=iris6,iris8# Don't run on iris1
#SBATCH --output slurm/%j.out # MAKE SURE slurm/ ALREADY EXISTS, OR ELSE YOUR JOB WILL FAIL SILENTLY!

# Now your Python or general experiment/job runner code
cd /iris/u/marcelto/mtql
source .venv/bin/activate
source env_setup.sh
export MUJOCO_GL=egl

python m_main.py --env_name=visual-antmaze-medium-navigate-singletask-v0 --train_steps=2000000 --agent=agents/mtql_transformer.py --agent.critic_grad_clip=5.0 --agent.alpha=300 --agent.attention_entropy_target="((3.8, 3.8), (3.3, 3.3))" --agent.hidden_dim=128 --hist_length=12 --log_interval=10 --agent.tokenization_mode=per_modality  --num_eval_envs=1 --eval_interval=10000 --eval_episodes=10 --video_episodes=0 --video_frame_skip=2 --hist_stride=50 --single_step_online --online_warmup_steps=1000000 --image_obs --agent.batch_size=32 --save_interval=10000 --agent.encoder=mem_vit_tokens_small --num_cached_episodes 20 --video_episodes 2 --action_chunk_size 8 --action_exec_horizon 8 --online_buf_size 500000 --nolazy_dataset
