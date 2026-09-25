#!/usr/bin/env bash
set -euo pipefail

cd /iris/u/ronpo/projects/new_mtql_candy_scooping

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available; run this script from a Slurm login shell." >&2
  exit 1
fi

resume() {
  local name="$1"
  local account="$2"
  local partition="$3"
  local nodelist="$4"
  local project="$5"
  local cue_mode="$6"
  local hist_length="$7"
  local hist_stride="$8"
  local agent="$9"
  local save_dir="${10}"
  local run_dir="${11}"
  local script="${12}"

  if [[ ! -d "${run_dir}/checkpoints" ]]; then
    echo "Checkpoint directory not found: ${run_dir}/checkpoints" >&2
    return 1
  fi

  # These are exported separately so long paths cannot be split while pasting.
  export PYTHON=/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python
  export PYTHONPATH=/iris/u/ronpo/projects/expo-ft
  export TASK_CONFIG=/iris/u/ronpo/projects/expo-ft/configs/task/candy_scoop.py
  export DATASET_PATH=/iris/u/ronpo/expo-ft-data/candy_scoop_v2_jitted
  export DATASET_CACHE=/iris/u/ronpo/mtql-runs/candy_scoop_v2_jitted_cache_v2norm
  export NORM_STATS_PATH=/iris/u/ronpo/projects/new_mtql_candy_scooping/norm_stats/candy_scoop_v2_jitted
  export PROJECT="$project"
  export CUE_MODE="$cue_mode"
  export P_AUG=1.0
  export SEED=0
  export HIST_LENGTH="$hist_length"
  export HIST_STRIDE="$hist_stride"
  export BATCH_SIZE=16
  export TRAIN_STEPS=1000000
  export LOG_INTERVAL=10000
  export SAVE_INTERVAL=10000
  export N_SUCC=-1
  export N_FAILS=-1
  export AGENT_CONFIG="$agent"
  export SAVE_DIR="$save_dir"
  export CHECKPOINT_DIR="${run_dir}/checkpoints"
  export RESUME=1
  unset RESTORE_PATH RESTORE_EPOCH OVERWRITE

  echo "Submitting ${name} from ${CHECKPOINT_DIR} on ${nodelist}"
  sbatch \
    --account="$account" \
    --partition="$partition" \
    --nodelist="$nodelist" \
    --nodes=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=128G \
    --time=120:00:00 \
    --job-name="$name" \
    --export=ALL \
    "$script"
}

# h18/s30: six jobs on the Sphinx pool.
resume resume-v2-vis-transformer-h18s30 nlp sphinx 'sphinx[4-11]' real_candy_scoop_v2_visual visual 18 30 agents/mtql_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_visual/transformer_h18s30 \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_visual/transformer_h18s30/candy-scoop-real/mtql_transformer_real_h18_img_sd000_s_17400147.0.20260912_031946 \
  run_candy_scoop_real.sh

resume resume-v2-vis-mlp-h18s30 nlp sphinx 'sphinx[4-11]' real_candy_scoop_v2_visual visual 18 30 agents/mtql_mlp_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_visual/mlp_h18s30 \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_visual/mlp_h18s30/candy-scoop-real/mtql_mlp_real_h18_img_sd000_s_17400148.0.20260912_032455 \
  run_candy_scoop_real.sh

resume resume-v2-vis-bc-h18s30 nlp sphinx 'sphinx[4-11]' real_candy_scoop_v2_visual visual 18 30 agents/new_bc_flow_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_visual/bc_h18s30 \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_visual/bc_h18s30/candy-scoop-real-bc/new_bc_flow_transformer_real_h18_img_sd000_s_17400149.0.20260912_032955 \
  run_candy_scoop_real_bc.sh

resume resume-v2-lang-transformer-h18s30 nlp sphinx 'sphinx[4-11]' real_candy_scoop_v2_language language 18 30 agents/mtql_transformer_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/transformer_h18s30 \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/transformer_h18s30/candy-scoop-real/mtql_transformer_language_real_h18_img_sd000_s_17400150.0.20260912_032956 \
  run_candy_scoop_real.sh

resume resume-v2-lang-mlp-h18s30 nlp sphinx 'sphinx[4-11]' real_candy_scoop_v2_language language 18 30 agents/mtql_mlp_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/mlp_h18s30 \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/mlp_h18s30/candy-scoop-real/mtql_mlp_language_real_h18_img_sd000_s_17400151.0.20260912_033454 \
  run_candy_scoop_real.sh

resume resume-v2-lang-bc-h18s30 nlp sphinx 'sphinx[4-11]' real_candy_scoop_v2_language language 18 30 agents/new_bc_flow_transformer_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/bc_h18s30 \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/bc_h18s30/candy-scoop-real-bc/new_bc_flow_transformer_language_real_h18_img_sd000_s_17400152.0.20260912_033455 \
  run_candy_scoop_real_bc.sh

# h20/s25: six jobs on the Iris H100/H200 pool.
resume resume-v2-vis-transformer-h20s25 iris iris-hi 'iris-hgx-1,iris-hgx-2' real_candy_scoop_v2_visual visual 20 25 agents/mtql_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2/transformer_h20s25 \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2/transformer_h20s25/candy-scoop-real/mtql_transformer_real_h20_img_sd000_s_17400096.0.20260912_030843 \
  run_candy_scoop_real.sh

resume resume-v2-vis-mlp-h20s25 iris iris-hi 'iris-hgx-1,iris-hgx-2' real_candy_scoop_v2_visual visual 20 25 agents/mtql_mlp_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2/mlp_h20s25 \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2/mlp_h20s25/candy-scoop-real/mtql_mlp_real_h20_img_sd000_s_17400097.0.20260912_030843 \
  run_candy_scoop_real.sh

resume resume-v2-vis-bc-h20s25 iris iris-hi 'iris-hgx-1,iris-hgx-2' real_candy_scoop_v2_visual visual 20 25 agents/new_bc_flow_transformer_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2/bc_h20s25 \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2/bc_h20s25/candy-scoop-real-bc/new_bc_flow_transformer_real_h20_img_sd000_s_17400098.0.20260912_030843 \
  run_candy_scoop_real_bc.sh

resume resume-v2-lang-transformer-h20s25 iris iris-hi 'iris-hgx-1,iris-hgx-2' real_candy_scoop_v2_language language 20 25 agents/mtql_transformer_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/transformer_h20s25 \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/transformer_h20s25/candy-scoop-real/mtql_transformer_language_real_h20_img_sd000_s_17400133.0.20260912_031412 \
  run_candy_scoop_real.sh

resume resume-v2-lang-mlp-h20s25 iris iris-hi 'iris-hgx-1,iris-hgx-2' real_candy_scoop_v2_language language 20 25 agents/mtql_mlp_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/mlp_h20s25 \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/mlp_h20s25/candy-scoop-real/mtql_mlp_language_real_h20_img_sd000_s_17400134.0.20260912_031342 \
  run_candy_scoop_real.sh

resume resume-v2-lang-bc-h20s25 iris iris-hi 'iris-hgx-1,iris-hgx-2' real_candy_scoop_v2_language language 20 25 agents/new_bc_flow_transformer_language_real.py \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/bc_h20s25 \
  /iris/u/ronpo/mtql-runs/candy_scoop_v2_language/bc_h20s25/candy-scoop-real-bc/new_bc_flow_transformer_language_real_h20_img_sd000_s_17400135.0.20260912_031342 \
  run_candy_scoop_real_bc.sh

echo "Submitted all 12 v2 resume jobs."
squeue -u ronpo -o "%.18i %.35j %.8T %.12M %R"
