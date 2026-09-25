# Egg Task History/Stride Analysis

## Recommendation

Use **history length 25 and stride 7 (`H25/S7`)** for the primary MTQL,
MTQL-MLP, and BC comparison.

- The data and control loop are 10 Hz, so this is a **17.5-second** history
  with one observation every **0.7 seconds**.
- It is the smallest tested dense-history setting that retains at least two
  conservative card-view observations at gripper-close onset in **all 102
  successful episodes**.
- Across three nested geometric definitions of the card-view interval, it
  retains at least one cue at every post-card step through grasp in **100% of
  successful episodes**.
- Under the strictest card-view definition it retains a mean of **6.30 cue
  samples** at grasp. This redundancy matters because image crop/rotation can
  weaken an edge-of-view card frame.

For a secondary setting constrained to fewer than 20 history frames, use
**H19/S10**. It spans 19 seconds with one observation per second, retains a
strict cue at grasp in all 102 successful episodes, and retains at least two
strict cue frames in 99% of them. Its mean strict cue count is 4.51. This is a
slightly stronger temporal-margin choice than H18/S10 for the cost of one
additional history frame.

Use **H30/S7** only as an optional long-margin ablation. It spans 21 seconds
and raises the strict cue count only slightly, from 6.30 to 6.49, while
increasing image traffic by 20% and transformer attention cost. It is not the
recommended secondary setting under the fewer-than-20-frame constraint.

Do **not** use H20/S7 for the primary experiment. Its 14-second span misses
the strict cue at grasp in 12 of 102 successful episodes.

## Data and timing facts

The finalized dataset contains 131 episodes and 33,629 transitions:

| Card | Success | Failure | Total |
| --- | ---: | ---: | ---: |
| Black | 51 | 15 | 66 |
| White | 51 | 14 | 65 |
| Total | 102 | 29 | 131 |

The collection loop is 10 Hz. The MP4 files are incorrectly tagged as 30 fps,
so they play three times faster than real time; HDF5 transition indices and
history calculations use the true 10 Hz rate.

Successful episodes have median duration 27.65 seconds, 90th percentile 35.0
seconds, and range 20.7--39.3 seconds. Failures are shorter: median 14.3
seconds and range 9.5--19.2 seconds.

The card-view interval was localized by the first positive-Y wrist-camera
excursion and checked visually against wrist videos. Three thresholds were
used (`y > 0.20`, `0.25`, and `0.28`) so the recommendation does not depend on
one boundary. For the strict `y > 0.28` core:

- card-view duration: median 4.6 seconds, 5th percentile 2.21 seconds,
  minimum 1.3 seconds;
- last core card frame to gripper-close onset: median 9.3 seconds,
  90th percentile 14.29 seconds, 95th percentile 14.79 seconds, maximum
  16.7 seconds.

Gripper-close onset is deliberately conservative. The trajectory commits to
the salt/pepper route before grasp, so requiring the history to retain the card
through grasp is stricter than requiring it only at initial route selection.

## Candidate comparison

Metrics below are the minimum over all three card-window thresholds. “Two cue
frames” is the fraction of successful episodes with at least two sampled card
observations at grasp. Actor/critic token counts assume one observation token
per history frame and action chunk 25.

| Setting | Span | Interval | Cue at grasp | Two cue frames | Mean strict cues | Every post-card step | Actor / critic tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H20/S7 | 14.0 s | 0.7 s | 88.2% | 83.3% | 4.98 | 88.2% | 22 / 47 |
| H20/S8 | 16.0 s | 0.8 s | 99.0% | 96.1% | 5.22 | 99.0% | 22 / 47 |
| H18/S10 | 18.0 s | 1.0 s | 100% | 99.0% | 4.45 | 100% | 20 / 45 |
| **H19/S10** | **19.0 s** | **1.0 s** | **100%** | **99.0%** | **4.51** | **100%** | **21 / 46** |
| H20/S9 | 18.0 s | 0.9 s | 100% | 98.0% | 4.93 | 100% | 22 / 47 |
| H20/S10 | 20.0 s | 1.0 s | 100% | 99.0% | 4.52 | 100% | 22 / 47 |
| H22/S8 | 17.6 s | 0.8 s | 100% | 99.0% | 5.56 | 100% | 24 / 49 |
| H24/S7 | 16.8 s | 0.7 s | 100% | 99.0% | 6.16 | 100% | 26 / 51 |
| **H25/S7** | **17.5 s** | **0.7 s** | **100%** | **100%** | **6.30** | **100%** | **27 / 52** |
| H30/S7 | 21.0 s | 0.7 s | 100% | 100% | 6.49 | 100% | 32 / 57 |

Episode bootstrap (5,000 samples) gives H20/S7 a 95% interval of 81.4--94.1%
for any strict cue at grasp and 75.5--90.2% for two cues. H25/S7 and H30/S7
are 100% in every observed episode and every bootstrap resample. This does not
prove population-level perfection, but it shows the recommendation is not
driven by an average that hides observed misses.

## Why this task can favor MTQL

The reward structure contains genuine counterfactual data. Successful grasp
locations form two disjoint routes: median grasp X is 0.449 for black cards
and 0.552 for white cards. Eleven failures reach a grasp, and all eleven take
the opposite route:

- 7/7 black-card grasp failures take the white-route shaker;
- 4/4 white-card grasp failures take the black-route shaker.

The remaining failures mostly terminate while approaching the wrong route.
Thus plain BC is trained to imitate both rewarded and unrewarded choices,
whereas MTQL can use terminal reward to value the correct card-conditioned
branch. This is a real reason to expect MTQL to beat BC; history length alone
is not.

The implementation comparison must be interpreted precisely:

- MTQL-transformer and MTQL-MLP use the same transformer flow actor. The MLP
  ablation replaces the Q critic, not the actor.
- Standalone BC also uses the same transformer flow actor but has no critic.
- H25/S7 therefore gives all agents identical visual history. The expected
  MTQL-transformer advantage is its order-aware reward critic, especially on
  the counterfactual failures—not privileged input.

A longer ordered sequence can expose the MLP critic's flattening weakness,
but no choice of H/S can guarantee a ranking before controlled robot
evaluation.

## Important dataset leakage warning

A low-capacity diagnostic using only a 3-by-4 grid of image-region mean RGB
values and five episode-index-grouped folds predicts card label from frames in
which the card should not be visible:

| Camera/phase | Card-label accuracy |
| --- | ---: |
| Wrist, initial frame | 93.9% |
| Side, initial frame | 91.6% |
| Wrist, core card frame | 96.9% |
| Side, corresponding phase | 87.0% |

Because the physical card is not visible in the initial side view, this is
session/lighting/background leakage, not task memory. The side images differ
by roughly 2--4 RGB levels between color collections. Existing brightness and
contrast augmentation should reduce this shortcut but does not prove it is
gone.

For a valid partial-observability benchmark, real-robot evaluation must flip
black/white conditions in randomized interleaved order without moving the
cameras, shakers, plate, or lighting. Better future data should also interleave
card colors within each collection session. Otherwise BC may infer color from
collection artifacts and the experiment will not isolate memory.

## Training and evaluation protocol

1. Train all three agents on the exact same full cache, egg norm statistics,
   H25/S7 histories, action chunk 25, image augmentation, training steps, and
   seed set. Do not provide card metadata or a cue token.
2. Keep the current temporally consistent augmentation: every frame in one
   camera history should share a geometric transform. Do not use grayscale,
   horizontal flips, cutout, or strong color jitter because the card itself is
   the signal.
3. Evaluate black and white trials in randomized interleaved order under one
   unchanged physical setup. Report both route-selection accuracy and final
   egg-task success; route selection is the cleaner memory endpoint.
4. Use at least 20 trials per card condition per agent. Keep action sampling
   noise and robot reset procedure identical.
5. Run H19/S10 as the secondary, fewer-than-20-frame temporal setting after
   the primary H25/S7 comparison. Treat H30/S7 only as an optional long-margin
   ablation. If H25/S7 does not beat BC, inspect side/current-frame shortcut
   use before increasing history further.

## Reproducibility

The numerical history analysis is generated by
`scripts/analyze_egg_history.py`; machine-readable results are in
`analysis/egg_history_metrics.json`. It uses the production history indices
`t-H*S,...,t-S`, episode-start clamping, all 102 successes for cue retention,
all finalized episodes for dataset counts, and path labels for outcome/color.
