# MTQL Real-Robot and EXPO-FT Compatibility

Last updated: 2026-09-06

## Scope

This document records design decisions for adding DROID real-robot and
EXPO-FT compatibility to MTQL.

- Baseline commit: `e859587`, the last commit before real-robot experiments.
- Real-robot training/evaluation uses the new `MTQLTransformerRealAgent`, a
  thin interface adapter around the unchanged base MTQL transformer.
- Do not modify `agents/mtql_transformer.py`.
- Preserve MTQL's native agent behavior unless the EXPO training loop is
  explicitly adopted.
- Do not commit changes until explicitly requested.

## Agent interface

MTQL remains the primary agent implementation.

EXPO-FT's learner is a PI0.5 residual learner, with a frozen PI0.5 actor,
separate ResNetV2 critic encoder, `N=8` action candidates, `n_edit_samples=8`,
and `utd_ratio=20`. Those settings do not exist in the MTQL transformer and
are intentionally not copied into `m_real.py`. The real MTQL run copies the
EXPO-compatible DROID representation, OpenPI normalization, PIL resize, full
augmentation, W&B/checkpoint conventions, and dataset handling while keeping
MTQL's own optimizer, critic, attention-entropy, history, and action-chunk
configuration.

The launcher key `encoder=resnet` is an MTQL registry name for its lightweight
IMPALA image encoder; it is not EXPO's ResNetV2. This is intentional because
changing it would change the MTQL checkpoint schema.

The real-data MLP ablation is `agents/mtql_mlp_real.py`. It is a thin adapter
around `agents/mtql_mlp.py`: the actor, encoder, history, action chunks, data
normalization, and training loop stay the same, while the transformer Q critic
is replaced by the parameter-matched flattened-token MLP (`67` hidden units
for `4` layers). Because this critic has no self-attention, attention-entropy
metrics and targets are ignored. Run it with
`AGENT_CONFIG=agents/mtql_mlp_real.py bash run_candy_scoop_real.sh`; use a
separate save/checkpoint directory from the transformer baseline.

The pure BC baseline is `agents/new_bc_flow_transformer_real.py`, built directly
on `agents/new_bc_flow_transformer.py`. It keeps only MTQL's
`actor_bc_flow` and flow-matching loss; it has no critic, Q loss, target
network, or attention-entropy loss. It still uses the same DROID preprocessing,
history, action chunks, augmentation, W&B, and Orbax checkpoint machinery.
Launch it with `bash run_candy_scoop_real_bc.sh`. The launcher defaults to
all successes and zero failures (`N_FAILS=0`), since BC has no reward signal
and should not imitate failed demonstrations; set `N_SUCC`/`N_FAILS` explicitly
if a different data mixture is intended.

- Use MTQL's native `create`, `sample_actions`, and `update` behavior in the
  standalone robot evaluator.
- Do not add an EXPO `AgentLearner` wrapper solely for checkpoint support.
- Add the EXPO `sample_actions`, `update`, `update_actor`, and
  `cache_infer_params` contract only if MTQL must run directly inside EXPO's
  training loop.

## Execution boundary

Only data collection and policy evaluation execute against the real robot.

- Real-robot processes collect raw DROID transitions and execute evaluated
  policy actions.
- Gradient updates and checkpoint saving run off-robot on a GPU machine.
- Training continues to use MTQL's native agent, losses, and `update` method.
- `m_main.py` remains the existing simulator trainer and is not modified for
  DROID.
- `m_real.py` is a separate offline trainer for saved DROID trajectories.
  It never creates a simulator or connects to a live robot.
- The DROID dataset adapter is input preprocessing for native MTQL training;
  it is not a new training algorithm.
- Reuse EXPO-FT's policy-independent `client/collect_data.py` to collect raw
  DROID trajectories; do not duplicate that collector inside MTQL.

## DROID observation mapping

Use the same two real camera views selected by EXPO-FT:

| Raw DROID field | MTQL field | Use |
| --- | --- | --- |
| `exterior_image_1_left` | `image` | Exterior/base camera |
| `wrist_image_left` | `wrist_image` | Wrist camera |
| `exterior_image_2_left` | — | Ignored |
| `cartesian_position` | `proprio` | First part of robot state |
| `gripper_position` | `proprio` | Final part of robot state |

`exterior_image_2_left` is not used because EXPO does not use it and the
current live wrapper duplicates `exterior_image_1_left` into that field.

OpenPI creates a zero-filled right-wrist image for its own expected input
schema. MTQL does not need this artificial image.

## MTQL observation contract

Current observations use separate camera keys:

```text
observations:
  image:       uint8 [B, H, W, 3]
  wrist_image: uint8 [B, H, W, 3]
  proprio:     float32 [B, 7]

history_observations:
  image:       uint8 [B, T, H, W, 3]
  wrist_image: uint8 [B, T, H, W, 3]
  proprio:     float32 [B, T, 7]

actions:
  float32 [B, action_chunk_size, 7]
```

MTQL does not concatenate the two images into a six-channel tensor. Its
observation encoder applies the shared IMPALA encoder to each camera,
concatenates the resulting embeddings with proprioception, and passes the
combined representation to the temporal transformer.

## Image preprocessing

- Default image size: 224 by 224.
- This matches OpenPI's `ModelTransformFactory` default and EXPO's replay
  buffer default.
- Use OpenPI's actual PIL-bilinear `resize_with_pad` implementation.
- Preserve aspect ratio and add black padding; do not stretch the image.
- Keep images as uint8 in `[0, 255]`.
- MTQL's IMPALA encoder performs its own division by 255.
- The image size remains configurable because training, evaluation, and
  checkpoint initialization must use the same resolution.

Resizing is deterministic preprocessing and applies during both training and
evaluation. It is not data augmentation.

## Normalization

- Load the same OpenPI `norm_stats.json` used by EXPO/OpenPI.
- Use OpenPI's `NormStats`, `Normalize`, and `Unnormalize` behavior.
- Use quantile normalization.

Training:

```text
raw 7D proprio ──Normalize(state stats)──> MTQL observation
raw 7D action  ──Normalize(action stats)─> MTQL training target
```

Live evaluation:

```text
raw 7D proprio ──Normalize(state stats)──> MTQL
                                                │
                                                ▼
                                      normalized action
                                                │
                                 Unnormalize(action stats)
                                                ▼
                                      raw 7D robot command
```

- Proprio observations are normalized before MTQL during both training and
  evaluation. They are not unnormalized because observations are inputs.
- MTQL learns and predicts actions in normalized action space. Predictions are
  unnormalized before they are sent to the robot-side server.
- Images remain `uint8` in `[0, 255]`; OpenPI normalization is not applied to
  them. MTQL's IMPALA encoder divides image values by 255 internally.
- Rewards and masks are not normalized.
- Do not compute dataset min/max normalization for DROID.

## Training augmentation

Match EXPO's augmentation strengths:

- Random crop to 95 percent of the original width and height.
- Resize back to the configured image size.
- Random rotation from -5 to +5 degrees.
- Color jitter with brightness, contrast, and saturation strengths of 0.1.

For each batch item and camera, share one transform across the historical
frames and current observation so augmentation does not invent camera motion.
Apply a separate transform to the next history and next observation, matching
EXPO-FT's independent current/next draws. Proprioception and actions are never
augmented.

Never apply random augmentation during evaluation.

## Temporal history

- Use the existing `HistoryDataset` history semantics.
- Respect `hist_length` and `hist_stride`.
- Clamp missing pre-episode history to the first observation of that episode.
- Never cross episode boundaries.
- Live evaluation keeps the same convention: history contains observations
  preceding previously executed actions.
- Present raw single-step DROID actions to the real MTQL agent with an explicit action-chunk
  dimension.
- For multi-step action chunks, compute discounted cumulative rewards and
  disable bootstrapping when the chunk reaches an episode terminal.
- Clamp next observations and next history to the current episode.

## Checkpoints

Support both checkpoint formats:

1. Native MTQL `params_<epoch>.pkl` checkpoints.
2. MTQL checkpoints using EXPO's Orbax manager layout.

Manager-layout compatibility does not imply agent-schema compatibility.
EXPO/OpenPI agent checkpoints cannot be restored into MTQL, and MTQL
checkpoints cannot be restored as EXPO/OpenPI agents.

The Orbax manager matches EXPO's `initialize_checkpoint_dir` configuration:

- `agent` and `params` PyTree item handlers.
- `max_to_keep=100`.
- Configurable `keep_period`.
- `create=False`.
- Two-hour asynchronous checkpoint timeout.
- EXPO-compatible overwrite, resume, and empty-directory behavior.

MTQL Orbax payloads are split as follows:

- `agent`: MTQL agent state with `network.params` replaced by an empty tree.
- `params`: a mapping containing `network_params`.
- `metadata`: Python and NumPy RNG state used for exact host-side resumption.

After restoration, `network_params` is merged back into the MTQL agent.

The local initializer is behaviorally identical to EXPO's implementation but
imports Orbax lazily. This avoids importing EXPO's complete agent and OpenPI
stack during evaluator startup.

### Native-to-Orbax conversion

The existing simulator trainer continues to save native
`params_<epoch>.pkl` checkpoints.

The legacy `convert_mtql_checkpoint_to_orbax.py` utility remains V2-specific
and is not used by `m_real.py`:

- Initializes a matching legacy V2 checkpoint skeleton from DROID data and the exact
  training configuration.
- Restores the native MTQL checkpoint.
- Writes the unchanged MTQL state into EXPO's Orbax manager layout.
- Does not retrain the model or change its parameters.

Conversion is optional when using the evaluator's native checkpoint mode. It
is required when an existing native checkpoint must be opened through
`initialize_checkpoint_dir`.

### Real-data training

`m_real.py` uses the native MTQL training loop with the Orbax checkpoint
manager described above.

- New real-data runs write Orbax checkpoints under `checkpoints/`.
- `--resume` restores the latest checkpoint in that directory.
- `--restore_path` and `--restore_epoch` restore an exact Orbax checkpoint.
- The trainer and evaluator must use identical agent architecture, encoder,
  history length, history stride, action chunk size, and image size.
- `--n_succ` and `--n_fails` optionally select deterministic subsets of the
  DROID dataset. Each total is balanced across target counts 1, 2, and 3;
  fixed-target and `arbitrary/target_num_N` failures are interleaved within
  their matching target count. `-1` loads all episodes.

For example, `--n_succ=30 --n_fails=30` selects 10 successes and 10
failures for each target count, with the same episode paths chosen on every
run. The old global `--num_data` cap is no longer part of `m_real.py`.

### Reproducible real-data launcher

Use `run_candy_scoop_real.sh` for the saved DROID dataset. It exposes the
real-data inputs and training choices explicitly: dataset path, task config,
OpenPI normalization statistics, deterministic success/failure counts, image
size, history length/stride, action chunk size, augmentation probability,
MTQL encoder and attention target, W&B settings, and Orbax checkpoint/resume
settings. Its defaults use all available episodes and the production DROID
settings (`hist_length=20`, `hist_stride=7`, `action_chunk_size=25`, and
`image_size=224`, `alpha=300`, and `critic_grad_clip=5.0`).

The pixel-cue experiment trains from the raw dataset
`/iris/u/ronpo/expo-ft-data/candy_scoop`. The loader derives the target from
each trajectory path, applies EXPO-style augmentation to the raw images, and
then draws the target-only cue into the transformed exterior image. This keeps
the cue fixed in the corner instead of rotating or color-jittering it. The
older `candy_scoop_pixel_cue` copy remains available for rendered previews, but
is not the default training source.

```bash
cd /afs/cs.stanford.edu/u/ronpo/projects/new_mtql
bash run_candy_scoop_real.sh
```

To define a different experiment, edit the literal values in the `args=(...)`
block of the launcher. For example, change `--n_succ=-1` to `--n_succ=30`,
`--n_fails=-1` to `--n_fails=30`, or change the history and
attention-target flags there. Dataset, normalization, save, and restore paths
remain environment-variable overrides.

Camera serial numbers, robot IPs, gripper ports, and NUC controller commands
are collection/evaluation settings; this offline trainer reads the recorded
HDF5 files and does not connect to the robot.

### Checkpoints and resumption

`m_real.py` uses an EXPO-FT-style Orbax checkpoint manager. It writes a
checkpoint directory under `<save_dir>/checkpoints` at every `--save_interval`
(default in the real launcher: 25,000 steps), retaining up to 100 checkpoints by default. W&B is
enabled by default by the launcher. At every `--log_interval`, it records all
agent update losses and diagnostics under `training/`, plus sampling time,
update time, interval time, total time, and throughput under `time/`. Each
checkpoint save also records its duration. The run configuration and derived
dataset/preprocessing metadata are stored in the W&B run config. Each
checkpoint stores the complete MTQL agent state, including model and
target-network parameters, optimizer state, optimizer steps, agent RNG, and
training metadata for the Python and NumPy RNGs.

Start a new run with an explicit checkpoint interval:

```bash
--save_interval=25000
```

Resume the latest checkpoint in the same directory:

```bash
/dev/shm/expo-ft-server-venv/bin/python m_real.py \
  ... \
  --save_dir=exp \
  --resume
```

Resume an exact step, optionally saving continued training to another
checkpoint directory:

```bash
/dev/shm/expo-ft-server-venv/bin/python m_real.py \
  ... \
  --restore_path=/path/to/checkpoints \
  --restore_epoch=200000 \
  --checkpoint_dir=/path/to/continued_checkpoints \
  --train_steps=1500000
```

The resumed loop starts at step 200001. Keep the agent architecture,
optimizer configuration, history settings, action chunk size, image size, and
normalization statistics unchanged when resuming.

- `eval_mtql_droid.py` can restore these checkpoints with
  `--checkpoint_format=orbax`.

## Standalone robot evaluation

`eval_mtql_droid.py`:

- Connects through EXPO's `EnvClientWrapper`.
- Accepts raw live DROID observations.
- Applies deterministic camera conversion and OpenPI normalization internally.
- Constructs MTQL temporal history.
- Calls the native `MTQLTransformerRealAgent`.
- Advances its sampling RNG explicitly.
- Supports native and Orbax checkpoints.
- Unnormalizes predicted actions.
- Sends raw 7D commands to the robot.
- Never applies training augmentation.

## Verified behavior

- Native and Orbax evaluator checkpoint routing.
- Orbax save, close, resume, and restore.
- Exact restoration of real-agent parameters, RNG, and configuration.
- Real image-shaped real-agent round-trip on an NVIDIA L40S.
- Two 224 by 224 uint8 cameras, temporal history, 7D proprioception, and 7D
  actions.
- Byte-identical output from OpenPI's `resize_with_pad`.
- Configurable image-size propagation through live history.
- DROID action chunks of size one and greater.
- Discounted action-chunk rewards and terminal masks.
- Episode-safe current and next temporal history.
- End-to-end native-pickle to Orbax conversion using EXPO-processed DROID
  input shapes and OpenPI normalization metadata.
- Exact real-agent restoration after native-to-Orbax conversion.
- Offline DROID trainer execution on an NVIDIA L40S, including EXPO dataset
  loading, OpenPI normalization, per-image EXPO augmentation, and a genuine
  real-agent gradient update.
- Direct Orbax save at step 1, restore/resume from step 1, a resumed gradient
  update, and direct Orbax save at step 2.
- MTQL logging utilities import successfully in EXPO's environment when
  `wandb_osh` is absent.
- Online or disabled W&B operation does not require `wandb_osh`; offline W&B
  synchronization reports a clear dependency error when it is unavailable.

## Remaining work

- Run `m_real.py` against the production DROID dataset and its production
  OpenPI `norm_stats.json`.
- Provide a runtime environment containing MTQL, EXPO/OpenPI, current Orbax,
  and W&B.
- Install `wandb_osh` only when offline W&B synchronization is required.
- Validate that training and evaluation use identical history length, history
  stride, action chunk size, image size, and agent configuration.
- Perform a guarded live-robot smoke test before executing unrestricted
  policy actions.
