#!/usr/bin/env bash
set -euo pipefail

# Matched v2 runs for both conditioning modes.  This script only submits jobs
# when explicitly executed; it does not modify the raw dataset or cache.
cd /iris/u/ronpo/projects/new_mtql_candy_scooping

launch() {
  local name="$1"
  local account="$2"
  local partition="$3"
  local nodelist="$4"
  local project="$5"
  local cue_mode="$6"
  local agent="$7"
  local save_dir="$8"
  local train_script="$9"

  sbatch \
    --account="${account}" \
    --partition="${partition}" \
    --nodelist="${nodelist}" \
    --nodes=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=128G \
    --time=120:00:00 \
    --job-name="${name}" \
    --export="ALL,DATASET_PATH=/iris/u/ronpo/expo-ft-data/candy_scoop_v2_jitted,DATASET_CACHE=/iris/u/ronpo/mtql-runs/candy_scoop_v2_jitted_cache_v2norm,NORM_STATS_PATH=/iris/u/ronpo/projects/new_mtql_candy_scooping/norm_stats/candy_scoop_v2_jitted,PROJECT=${project},CUE_MODE=${cue_mode},P_AUG=1.0,SEED=0,HIST_LENGTH=20,HIST_STRIDE=25,BATCH_SIZE=16,TRAIN_STEPS=1000000,LOG_INTERVAL=10000,SAVE_INTERVAL=10000,N_SUCC=-1,N_FAILS=-1,AGENT_CONFIG=${agent},SAVE_DIR=${save_dir}" \
    "${train_script}"
}

launch v2-visual-transformer-h20s25 \
  iris iris-hi iris-hgx-1 \
  real_candy_scoop_v2_visual visual \
  agents/mtql_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2/transformer_h20s25 \
  run_candy_scoop_real.sh

launch v2-visual-mlp-h20s25 \
  iris iris-hi iris-hgx-1 \
  real_candy_scoop_v2_visual visual \
  agents/mtql_mlp_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2/mlp_h20s25 \
  run_candy_scoop_real.sh

launch v2-visual-bc-h20s25 \
  iris iris-hi iris-hgx-1 \
  real_candy_scoop_v2_visual visual \
  agents/new_bc_flow_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2/bc_h20s25 \
  run_candy_scoop_real_bc.sh

launch v2-language-transformer-h20s25 \
  nlp sphinx 'sphinx[5,7,9]' \
  real_candy_scoop_v2_language language \
  agents/mtql_transformer_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/transformer_h20s25 \
  run_candy_scoop_real.sh

launch v2-language-mlp-h20s25 \
  nlp sphinx 'sphinx[5,7,9]' \
  real_candy_scoop_v2_language language \
  agents/mtql_mlp_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/mlp_h20s25 \
  run_candy_scoop_real.sh

launch v2-language-bc-h20s25 \
  nlp sphinx 'sphinx[5,7,9]' \
  real_candy_scoop_v2_language language \
  agents/new_bc_flow_transformer_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/bc_h20s25 \
  run_candy_scoop_real_bc.sh

squeue -u ronpo -o "%.18i %.40j %.8T %.12M %R"
