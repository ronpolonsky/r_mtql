# Candy Scoop MTQL

This is the command reference for the real Franka/Robotiq candy-scoop task.
Use the **right arm only**.

## NUC: start the right-arm services

After reboot, use two NUC terminals. Leave both running.

Terminal 1, the Polymetis robot controller:

```bash
source /home/iliad/Utilities/miniconda3/etc/profile.d/conda.sh
conda activate droid
cd /tmp
mkdir -p /tmp/iliad-polymetis-right
launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=172.16.0.2 port=50053 hydra.run.dir=/tmp/iliad-polymetis-right hydra.output_subdir=null
```

Terminal 2, the DROID bridge:

```bash
source /home/iliad/Utilities/miniconda3/etc/profile.d/conda.sh
conda activate droid
cd /home/iliad/ronpo/droid
python scripts/server/run_server.py --zerorpc-port=4242 --robot-ip=172.16.0.2 --robot-port=50053 --gripper-comport=/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DA6UJOT5-if00-port0 --gripper-port=50054
```

Do not run `run_servers.sh`; it starts both arms.

## Iris: rollout service

```bash
cd /afs/cs.stanford.edu/u/ronpo/projects/expo-ft
export PYTHONPATH="$PWD"
/dev/shm/expo-ft-client-venv/bin/python -m client.run_client \
  --server_host=0.0.0.0 \
  --server_port=8102 \
  --config_task_path=configs/task/candy_scoop.py
```

The rollout environment resets exactly like collection: it first retracts
vertically to `z=0.30`, then returns in joint space to the configured base
joints. This reset uses arm-only commands, preserves the physical gripper
width, and is not included in the evaluation trajectory. Restart this rollout
service after changing the EXPO-FT environment code.

```bash
/dev/shm/expo-ft-client-venv/bin/python -c 'import os, zerorpc;c=zerorpc.Client(heartbeat=20); c.connect("tcp://"+os.environ["NUC_IP"]+":4242"); c.launch_robot(); state,_=c.get_robot_state();print(state["joint_positions"]); print(state["gripper_position"])'
```

## Iris: inspect the robot

This does not launch a controller or command the gripper.

```bash
cd /afs/cs.stanford.edu/u/ronpo/projects/expo-ft
export PYTHONPATH="$PWD"
/dev/shm/expo-ft-client-venv/bin/python -m client.inspect_robot_state \
  --task_config=configs/task/candy_scoop.py \
  --task_config.side_camera_id=38651013_left \
  --task_config.wrist_camera_id=15577469_left \
  --move=true \
  --launch_controller=false \
  --allow_gripper=true
```

## Iris: view ZED cameras

```bash
newgrp zed
cd ~/projects/expo-ft
/usr/local/bin/ZED_Explorer
```

Camera IDs: side `38651013_left`, wrist `15577469_left`.

## View  eval recordings
```bash
  cd /tmp/iliad-eval-bc-language-h18s30-150k
  python3 -m http.server 8765 --bind 127.0.0.1 >/tmp/candy-video-http.log 2>&1 &
  xdg-open http://127.0.0.1:8765/raw_0914_161433_train_ep000006.mp4
```
## Iris: evaluate v2 visual MTQL at step 40000

The rollout service above must already be running. From a fresh Iris terminal,
run the matched v2 visual transformer checkpoint:

```bash
cd /iris/u/ronpo/projects/new_mtql_candy_scooping
export PYTHON=/dev/shm/ronpo-newmtql-venv/bin/python
export PYTHONPATH="/afs/cs.stanford.edu/u/ronpo/projects/expo-ft:$PWD"
RUN=/iris/u/ronpo/mtql-runs/candy_scoop_v2/transformer_h20s25/candy-scoop-real/mtql_transformer_real_h20_img_sd000_s_17400096.0.20260912_030843
RESULTS="$RUN/evaluation/step_40000"

"$PYTHON" -u eval_mtql_droid.py \
  --agent=agents/mtql_transformer_real.py \
  --config_task=/iris/u/ronpo/projects/expo-ft/configs/task/candy_scoop.py \
  --norm_stats_path=/iris/u/ronpo/projects/new_mtql_candy_scooping/norm_stats/candy_scoop_v2_jitted \
  --restore_path="$RUN/checkpoints" \
  --restore_epoch=40000 \
  --checkpoint_format=orbax \
  --cue_mode=visual \
  --hist_length=20 \
  --hist_stride=25 \
  --image_size=224 \
  --action_chunk_size=25 \
  --action_exec_horizon=0 \
  --gripper_mode=hold \
  --inject_training_gripper_state=true \
  --num_episodes=20 \
  --client_host=localhost \
  --client_port=8102 \
  --results_dir="$RESULTS" \
  --video_dir="$RESULTS/videos"
```

The evaluator starts each episode automatically after a one-second delay.
`gripper_mode=hold` ignores the seventh policy action and leaves the physical
gripper untouched. Results are written to the next available
`$RESULTS/trial_N/` directory.

This command matches the v2 trainer metadata: v2 jitted dataset, v2 norm stats,
visual conditioning, history length 20, stride 25, 224px inputs, and 25-step
action chunks. The generic `run_candy_scoop_real.sh` defaults to the older v1
dataset and stats, so do not use its defaults for this checkpoint.

## Iris: general v2 evaluation template

Use this template for any v2 checkpoint. Set `RUN`, `STEP`, and `AGENT` to the
run and checkpoint being evaluated. Use the agent and cue mode that were used
for training: `mtql_transformer_real.py` with `visual`,
`mtql_transformer_language_real.py` with `language`, or the corresponding
`new_bc_*_real.py` agent for a BC checkpoint.

```bash
cd /iris/u/ronpo/projects/new_mtql_candy_scooping
export PYTHON=/dev/shm/ronpo-newmtql-venv/bin/python
export PYTHONPATH="/afs/cs.stanford.edu/u/ronpo/projects/expo-ft:$PWD"

RUN=/iris/u/ronpo/mtql-runs/candy_scoop_v2/<run-directory>
STEP=<checkpoint-step>
AGENT=agents/<matching-agent>.py
CUE_MODE=<visual-or-language>
RESULTS="$RUN/evaluation/step_${STEP}"

"$PYTHON" -u eval_mtql_droid.py \
  --agent="$AGENT" \
  --config_task=/iris/u/ronpo/projects/expo-ft/configs/task/candy_scoop.py \
  --norm_stats_path=/iris/u/ronpo/projects/new_mtql_candy_scooping/norm_stats/candy_scoop_v2_jitted \
  --restore_path="$RUN/checkpoints" \
  --restore_epoch="$STEP" \
  --checkpoint_format=orbax \
  --cue_mode="$CUE_MODE" \
  --hist_length=20 \
  --hist_stride=25 \
  --image_size=224 \
  --action_chunk_size=25 \
  --action_exec_horizon=0 \
  --gripper_mode=hold \
  --inject_training_gripper_state=true \
  --num_episodes=20 \
  --client_host=localhost \
  --client_port=8102 \
  --results_dir="$RESULTS" \
  --video_dir="$RESULTS/videos"
```

The v2 injection flag supplies the constant training gripper observation
(`0.17621146`) while `gripper_mode=hold` leaves the physical gripper untouched.
The rollout service must be running first. Results are written under the run's
`evaluation/step_<step>/trial_N/` directory.


## Iris: evaluate legacy v1 language MTQL at step 110000

The rollout service above must already be running. From a fresh terminal, copy
this complete block; it defines all required paths and uses the Iris-stored
evaluator environment directly:

```bash
cd /iris/u/ronpo/projects/new_mtql_candy_scooping
export PYTHON=/iris/u/ronpo/venvs/ronpo-newmtql-venv/bin/python
export PYTHONPATH="/afs/cs.stanford.edu/u/ronpo/projects/expo-ft:$PWD"
RUN=/iris/u/ronpo/mtql-runs/candy_scoop_language/transformer_h18s30_jit/candy-scoop-real/mtql_transformer_language_real_h18_img_sd000_s_17359466.0.20260910_040725
RESULTS="$RUN/evaluation/step_110000"

"$PYTHON" -u eval_mtql_droid.py \
  --agent=agents/mtql_transformer_language_real.py \
  --config_task=/iris/u/ronpo/projects/expo-ft/configs/task/candy_scoop.py \
  --norm_stats_path=/iris/u/ronpo/expo-ft-data/candy_scoop_norm_stats \
  --restore_path="$RUN/checkpoints" \
  --restore_epoch=110000 \
  --checkpoint_format=orbax \
  --cue_mode=language \
  --hist_length=18 \
  --hist_stride=30 \
  --image_size=224 \
  --action_chunk_size=25 \
  --action_exec_horizon=0 \
  --gripper_mode=hold \
  --num_episodes=20 \
  --client_host=localhost \
  --client_port=8102 \
  --results_dir="$RESULTS"
```

For this legacy v1 checkpoint, adding `--inject_training_gripper_state=true`
would inject the v1 training constant `gripper_position=0.1850220263004303`
into the policy observation while leaving the physical gripper unchanged.

The evaluator always uses a 7D checkpoint action, with two physical gripper modes:

- `--gripper_mode=hold` (default): ignore the seventh action and move only the arm, leaving the physical gripper completely untouched.
- `--gripper_mode=policy`: send all seven action dimensions so the policy controls the gripper.

Add `--inject_training_gripper_state=true` to replace the live gripper observation with the constant training value while leaving the physical gripper unchanged. The exact constants are dataset-specific:

- v1 (`/iris/u/ronpo/expo-ft-data/candy_scoop_norm_stats`): `q01=q99=0.1850220263004303`.
- v2 (`/iris/u/ronpo/projects/new_mtql_candy_scooping/norm_stats/candy_scoop_v2_jitted`): raw trajectories store `0.17621145562692822`; norm stats store `q01=q99=0.17621146142482758` (printed as `0.17621146`).

The evaluator selects the value from the norm-stats path passed to that run and injects it into the initial observation, every reset observation, every per-step observation, and the history buffer. When injection is enabled, it verifies q01 and q99 agree and records the reference and policy-injected value in `results.json`.

Results go to `$RUN/evaluation/step_110000/trial_N/`. Enter `4` at the manual
result prompt to discard an attempt; it is not counted and the same evaluation
number is retried. Restart the evaluator after evaluator-code updates; restart
the rollout service after rollout-server code updates.

If the evaluator environment ever needs rebuilding, keep it off AFS:

```bash
cd /iris/u/ronpo/projects/new_mtql_candy_scooping
mkdir -p /iris/u/ronpo/venvs /iris/u/ronpo/uv-tmp
UV_PROJECT_ENVIRONMENT=/iris/u/ronpo/venvs/ronpo-newmtql-venv \
UV_CACHE_DIR=/iris/u/ronpo/.cache/uv \
TMPDIR=/iris/u/ronpo/uv-tmp uv sync
```

Keep `/iris/u/ronpo/mtql-runs` and `/iris/u/ronpo/expo-ft-data`; they contain
checkpoints and trajectories.
