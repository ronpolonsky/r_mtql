#!/usr/bin/env bash
set -euo pipefail

cd /iris/u/ronpo/projects/new_mtql_candy_scooping

SBATCH_COMMON=(
  --account=nlp
  --partition=sphinx
  --nodelist='sphinx[4-11]'
  --nodes=1
  --gres=gpu:1
  --cpus-per-task=8
  --mem=128G
  --time=120:00:00
)

launch() {
  local name="$1"
  local hist_length="$2"
  local hist_stride="$3"
  local agent_config="$4"
  local save_dir="$5"
  local train_script="$6"

  sbatch "${SBATCH_COMMON[@]}" \
    --job-name="${name}" \
    --export="ALL,DATASET_PATH=/iris/u/ronpo/expo-ft-data/candy_scoop,DATASET_CACHE=/iris/u/ronpo/mtql-runs/candy_scoop_cache_full,PROJECT=real_candy_scoop,CUE_MODE=visual,P_AUG=1.0,SEED=0,HIST_LENGTH=${hist_length},HIST_STRIDE=${hist_stride},BATCH_SIZE=16,TRAIN_STEPS=1000000,LOG_INTERVAL=10000,SAVE_INTERVAL=10000,N_SUCC=-1,N_FAILS=-1,AGENT_CONFIG=${agent_config},SAVE_DIR=${save_dir}" \
    "${train_script}"
}

launch transformer-h18s30 18 30 \
  agents/mtql_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/transformer_h18s30_sphinx_full \
  run_candy_scoop_real.sh

launch mlp-h18s30 18 30 \
  agents/mtql_mlp_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/mlp_h18s30_sphinx_full \
  run_candy_scoop_real.sh

launch bc-h18s30 18 30 \
  agents/new_bc_flow_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/bc_h18s30_sphinx_full \
  run_candy_scoop_real_bc.sh

launch transformer-h23s24 23 24 \
  agents/mtql_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/transformer_h23s24_sphinx_full \
  run_candy_scoop_real.sh

launch mlp-h23s24 23 24 \
  agents/mtql_mlp_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/mlp_h23s24_sphinx_full \
  run_candy_scoop_real.sh

launch bc-h23s24 23 24 \
  agents/new_bc_flow_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/bc_h23s24_sphinx_full \
  run_candy_scoop_real_bc.sh

launch transformer-h14s30 14 30 \
  agents/mtql_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/transformer_h14s30_sphinx_full \
  run_candy_scoop_real.sh

launch mlp-h14s30 14 30 \
  agents/mtql_mlp_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/mlp_h14s30_sphinx_full \
  run_candy_scoop_real.sh

launch bc-h14s30 14 30 \
  agents/new_bc_flow_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_pixel_cue/bc_h14s30_sphinx_full \
  run_candy_scoop_real_bc.sh

echo "Submitted nine Candy-Scoop runs."
squeue -u ronpo -o "%.18i %.35j %.8T %.12M %R"
