# Egg v2 finger-instruction memory analysis

## Recommendation

Use these two settings for the matched `mtql_transformer_real`,
`mtql_mlp_real`, and `new_bc_flow_transformer_real` comparison:

1. **Primary: H18/S4** — 18 history frames, sampled every 0.4 seconds, for a
   7.2-second span.
2. **Secondary: H14/S5** — 14 history frames, sampled every 0.5 seconds, for
   a 7.0-second span.

Both are below 20 history frames. H18/S4 is the stronger transformer-focused
setting: it retains a denser sequence and stays close to the H20/action-chunk-25
length at which the MLP critic was parameter matched. H14/S5 is the efficient
confirmation setting: it has nearly the same physical horizon with 22% fewer
history image tokens.

These settings maximize cue availability without turning the history almost
entirely into copies of the initial frame. They cannot guarantee that the
transformer will be much better. The current data contains target and route
shortcuts, and BC has the same history-aware transformer actor as MTQL.

## Dataset

Only finalized numeric episode directories are included. The four
`tmp/session_*` recordings are excluded.

| Intended target | Success | Failure | Total |
| --- | ---: | ---: | ---: |
| Pepper (`card_black`) | 50 | 17 | 67 |
| Salt (`card_white`) | 50 | 16 | 66 |
| **Total** | **100** | **33** | **133** |

The dataset contains 15,616 transitions: 13,395 successful transitions and
2,221 failure transitions. Uniform transition sampling therefore draws only
14.2% of its samples from failures even though failures are 24.8% of episodes.

The EXPO task configuration records a 10 Hz control loop. Successful episodes
range from 99 to 186 steps, with median 131.5 steps (13.15 seconds). Failure
episodes range from 38 to 153 steps, with median 60 steps (6.0 seconds).

## Actual finger-cue timing

The previous `analyze_egg_history.py` card/Y-position heuristic is not valid
for egg v2. In v2 the cue is the person's hand and finger in the side-camera
image. The v2 analysis detects skin-colored pixels in the relevant side-camera
region, requires three consecutive low-skin frames to mark departure, and was
visually checked on short- and long-cue outliers.

For successful episodes:

- last finger-cue frame: median step 10 (1.0 seconds), 95th percentile step
  18, range step 2--23;
- first gripper-close action: median step 47 (4.7 seconds), 95th percentile
  step 68, range step 28--78;
- last cue to gripper close: median 37 steps (3.7 seconds), 95th percentile
  54.05 steps, maximum 66 steps (6.6 seconds).

Requiring the cue to remain in sampled history through gripper close is
conservative: target-route commitment occurs earlier than the grasp.

There is an important limitation. The robot begins moving at median step 3,
while the finger remains visible until median step 10. Thus action and cue
overlap. The demonstrations do not contain a clean interval in which the hand
has left but the robot has not yet committed to a route.

## History-grid results

Production history indices are exactly `t-H*S, ..., t-S`, with indices before
episode start clamped to frame 0. Metrics below use all 100 successful v2
episodes and test retention at gripper-close onset.

| Setting | Span | Interval | Any cue | >=2 unique cue frames | Mean unique cue frames | Frame-0 padding | Actor / critic tokens | Batch-16 current+next history images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H10/S7 | 7.0 s | 0.7 s | 100% | 94% | 2.46 | 37.5% | 12 / 37 | 91.9 MiB |
| H12/S6 | 7.2 s | 0.6 s | 100% | 96% | 2.68 | 38.3% | 14 / 39 | 110.3 MiB |
| **H14/S5** | **7.0 s** | **0.5 s** | **100%** | **97%** | **2.99** | **36.1%** | **16 / 41** | **128.6 MiB** |
| H16/S4 | 6.4 s | 0.4 s | 99% | 97% | 3.49 | 29.4% | 18 / 43 | 147.0 MiB |
| **H18/S4** | **7.2 s** | **0.4 s** | **100%** | **98%** | **3.61** | **36.8%** | **20 / 45** | **165.4 MiB** |
| H18/S5 | 9.0 s | 0.5 s | 100% | 98% | 3.03 | 50.3% | 20 / 45 | 165.4 MiB |

Both recommendations retain at least one cue at every post-cue step through
grasp in every successful episode. H18/S4 has the best useful density among
the full-coverage settings. H18/S5 adds no observed coverage and replaces more
real temporal context with repeated frame 0. H16/S4 is efficient but misses
the cue at grasp in one observed success.

Do not use stride 25 for this task. At grasp, H12/S25, H14/S25, and H18/S25
have 88.9%, 90.5%, and 92.6% frame-0 padding, respectively. They technically
retain a cue, but mostly by repeating the same initial image. That is not a
meaningful test of long temporal reasoning.

## Why MTQL has a real opportunity to beat BC

Successful grasp positions cleanly separate the two target routes: a nearest
centroid classifier on grasp XYZ separates all 100 successes. Of 33 failures:

- 32 reach a gripper-close action;
- 14 take the opposite target route: 8 intended-pepper episodes go to salt,
  and 6 intended-salt episodes go to pepper;
- those wrong-route failures contribute 1,099 transitions, 7.0% of the full
  dataset;
- 18 failures take the intended route but fail later, contributing 1,058
  transitions;
- one failure never closes the gripper.

BC treats all of those actions as demonstrations. MTQL receives terminal
reward 1 for successful episodes and 0 for failures, so its critic can in
principle suppress the cue-inconsistent branch and poor executions. This
reward-labeled counterfactual data—not history length by itself—is the main
reason to expect MTQL to beat BC.

The architecture comparison must be described accurately:

- MTQL-transformer and MTQL-MLP use the same transformer flow actor. Only the
  critic differs.
- The MLP critic is not order-blind. It receives positional embeddings, then
  flattens all ordered tokens into an MLP.
- BC also uses the same history-aware transformer flow actor, but has no
  reward critic.
- The transformer's plausible advantage is selective token interaction in
  the reward critic over a moderately long, noisy sequence. It is not
  privileged access to the cue.

## Serious benchmark confounds in v2

### Target is predictable without looking at the finger

The salt and pepper data were collected in separate time blocks: white/salt
from roughly 18:49--19:29 and black/pepper from 19:32--20:05. Camera exposure
drift is large. In the bottom 30 pixels of the initial side frame, which does
not contain the pointing hand, mean RGB is approximately:

- pepper block: `(137.8, 140.5, 127.0)`;
- salt block: `(153.4, 157.9, 146.2)`.

A small logistic model with five paired episode-index folds predicts target
with 100% accuracy from a 2-by-4 RGB grid of only that bottom strip. The same
diagnostic gets 89% from the bottom of the initial wrist frame. This is a
collection-session shortcut, not memory.

The current augmentation uses crop, rotation, and brightness/contrast/
saturation jitter of 0.1. Although `p_aug=1.0`, AugMax `ColorJitter` itself has
default probability 0.5. This may reduce the shortcut but does not establish
that it has been removed.

### The robot route quickly reveals the target

A cross-validated linear classifier using only current proprioception predicts
the intended target at:

| Time from start | Accuracy |
| ---: | ---: |
| 0.0 s | 63% |
| 1.0 s | 66% |
| 1.5 s | 72% |
| 2.0 s | 78% |
| 2.5 s | 89% |
| 3.0 s | 97% |
| 4.0 s | 99% |

Once route motion is established, neither the actor nor critic needs the
finger cue to infer which object the demonstrator chose. This is unavoidable
in most embodied tasks after commitment, but here commitment begins while the
cue is still visible.

### Outcome also has some initial-frame leakage

Within the pepper block, the initial side-camera bottom-strip diagnostic
predicts success versus failure at 70.4% balanced accuracy. Within the salt
block, the analogous wrist diagnostic reaches 69.4%. A critic may therefore
learn some collection chronology rather than action quality.

These confounds do not make training useless, but they prevent a strong claim
that a performance gap proves long-memory reasoning.

## Benchmark protocol

1. Precommit to H18/S4 as primary and H14/S5 as secondary. Do not choose the
   winner after evaluating many history settings.
2. Give all three agents exactly the same v2 cache, v2 normalization, action
   chunk 25, augmentation, batch size, number of updates, seed set, and
   history setting. Keep `cue_mode=none`; path labels and HDF5 `card_color`
   metadata must never enter model observations.
3. Use at least three training seeds. Evaluate checkpoints at the same steps,
   initially every 30k through 300k, because one seed/checkpoint can easily
   reverse a small ranking on 100 successes and 33 failures.
4. During robot evaluation, randomize salt versus pepper trial-by-trial under
   one unchanged camera, lighting, object, and reset setup. Never evaluate all
   salt and then all pepper. This breaks the strongest collection shortcut.
5. Report target-selection accuracy separately from final task success. Also
   report a 2-by-2 intended-versus-picked shaker confusion matrix and late
   execution failures after the correct shaker was selected.
6. For the cleanest memory evaluation, enforce a fixed pause after the finger
   leaves before the robot may move. Existing v2 demonstrations do not have
   that pause, so this would ideally be included in a future interleaved data
   collection rather than introduced only at evaluation.

The existing `m_real_egg.py` correctly uses `cue_mode=none`, the v2 cache, and
v2 norm statistics. It currently defaults to `configs/task/egg.py`; offline
training uses the same action keys either way, but v2 robot evaluation must use
`configs/task/egg_v2.py` because its reset joints and workspace bounds differ.

## Reproducibility

The read-only analysis is implemented in
`scripts/analyze_egg_v2_memory.py`. It scans all 133 finalized episodes,
reads the two camera streams plus proprio/action arrays, uses the production
history-index convention, and can write a machine-readable JSON report. It
does not modify raw data.
