# Cabinet transformer history-window report

## Bottom line

Use **history length 20, stride 50** for the next cabinet experiments.

This gives the transformer one remembered state every 50 physical control
steps (5 seconds) over the complete 1000-step evaluation horizon.  The actual
rollouts show that a drawer/search phase normally lasts about 100--150 steps,
so stride 50 preserves roughly 2--3 views of each phase while length 20 can
retain all earlier phases in a long search.

The best higher-resolution alternative is **length 24, stride 40**.  It covers
960 steps and preserves about three samples per typical phase, but it encodes
20% more historical images and increases actor self-attention work by about
40%.  It is a useful second ablation, not my first setting for the current
25-demo regime.

## What was inspected

I inspected both cameras in the locally stored W&B evaluation videos from:

- MTQL run `i1dln1b0` (H20/S50), checkpoints 250k, 300k, and 400k.
- BC run `rn2ev37p` (H12/S50), checkpoints 300k, 650k, and 800k.

Each video contains two extra rendered evaluation episodes.  The source video
keeps every second environment frame; the contact sheets below then sample
every 25 video frames, so adjacent cells are exactly 50 physical steps apart.
These video episodes are not part of the 20-episode reported evaluation
metric, and they use the same fixed evaluation layout.  They are useful for
temporal inspection, not for estimating success rates.

I also measured all exact first 25 successful demonstrations and first five
failed demonstrations selected by the V2 data loader.  That provides 30 full
trajectories rather than relying only on the 12 rendered episodes.

## What the rollout frames show

The robot repeatedly alternates between a close drawer/door interaction and a
return toward a wide/home pose.  A sample every 50 steps normally records an
approach or close-up frame and a later open/return frame.  A sample every 7 or
10 steps records many nearly identical views of the same reach.

Concrete rendered examples:

- MTQL at 250k: both rendered rollouts continue to the 1000-step timeout and
  contain repeated search phases throughout the full horizon.
- MTQL at 300k: both again use the entire 1000-step horizon.
- MTQL at 400k: one rollout finds the blue cube and ends around physical step
  608; the other continues to 1000.
- BC at 300k: one rollout finds the blue cube and ends around step 450; the
  other continues to 1000.
- BC at 650k and 800k: rendered successes end around steps 866 and 824,
  respectively, while the paired rollout continues to 1000.

This distinction matters.  At a 450-step success, H12/S50 already contains
the entire episode.  At step 850 it only covers steps 250--800, and at the
1000-step timeout it only covers steps 400--950.  Thus H12/S50 is adequate for
many easy/early successes but forgets the first several drawer visits in the
long cases where task memory is most valuable.  H20/S50 still contains the
start of all of those episodes.

Full two-camera contact sheets are in
[`full_contact_sheets`](./full_contact_sheets).  Each file contains one
rollout, ordered from step 0 through step 1000 in increments of 50.

## Measured phase duration

I used the seven arm joints in the stored demonstrations to identify sustained
returns near the episode's initial/home configuration (joint-space L2 distance
below 0.15 for at least five consecutive states).  Gaps between those returns
are a reproducible proxy for one drawer/search phase.  It is a heuristic, but
the boundaries agree with the visible close-up/return cycles in the videos.

Across the selected 25 successes plus five failures:

| Quantity | Result |
|---|---:|
| Detected complete phases | 110 |
| Phase length, median | 121.5 steps (12.2 s) |
| Phase length, 25th--75th percentile | 88--138 steps |
| Phase length, 10th--90th percentile | 74--152 steps |
| Full observed range | 62--198 steps |
| Phases per successful episode, median | 4 |
| Maximum phases in a successful episode | 8 |

At 10 Hz control frequency, the resulting temporal density is:

| Stride | Time between memories | Samples per median phase | Interpretation |
|---:|---:|---:|---|
| 7 | 0.7 s | 17.4 | Very redundant; mostly local arm motion |
| 10 | 1.0 s | 12.2 | Still heavily redundant |
| 25 | 2.5 s | 4.9 | Good local detail, limited global horizon |
| 40 | 4.0 s | 3.0 | Good phase-level summary |
| **50** | **5.0 s** | **2.4** | Sparse but sufficient phase-level summary |

The camera differences support the same conclusion.  Mean absolute RGB change
between frames (averaged over the selected successes and normalized to
0--1) rises substantially through lag 40--50; lag 7--10 is still visually
close to the preceding state.

| Lag | Agent camera change | Wrist camera change |
|---:|---:|---:|
| 7 | 0.0186 | 0.0838 |
| 10 | 0.0213 | 0.0976 |
| 25 | 0.0273 | 0.1387 |
| 40 | 0.0300 | 0.1601 |
| 50 | 0.0318 | 0.1698 |
| 100 | 0.0362 | 0.2029 |
| 200 | 0.0376 | 0.2045 |

The fixed agent camera changes less than the wrist camera because most of its
background is static.  The wrist view shows the expected stronger separation
as the robot moves between handles and drawer interiors.

## Candidate windows

One environment step is 0.1 seconds.  The action execution horizon is 25
steps, so the policy is queried every 2.5 seconds.  With S50, each historical
sample is separated by two policy queries.  The current observation still
provides exact local state; history can therefore specialize in recording
which parts of the cabinet were already visited.

| History / stride | Span | Typical samples per phase | Actor tokens | Critic tokens | Assessment |
|---|---:|---:|---:|---:|---|
| H12 / S7 | 84 steps | 17.4 | 14 | 39 | Only the current maneuver |
| H20 / S10 | 200 steps | 12.2 | 22 | 47 | About 1--2 phases |
| H20 / S25 | 500 steps | 4.9 | 22 | 47 | Good local detail; loses early visits |
| H12 / S50 | 600 steps | 2.4 | 14 | 39 | Good for short episodes; incomplete long memory |
| H20 / S40 | 800 steps | 3.0 | 22 | 47 | Strong compromise but loses first 200 steps at timeout |
| **H20 / S50** | **1000 steps** | **2.4** | **22** | **47** | **Recommended: full horizon and adequate phase sampling** |
| H24 / S40 | 960 steps | 3.0 | 26 | 51 | Best denser alternative; more compute |

Token counts reflect the code as configured with ResNet and
`tokenization_mode=per_modality`: each historical state's two image embeddings
and proprioception are concatenated into one observation token.  The actor has
CLS + history + current observation.  The critic additionally has 25 action
tokens, one per step in the action chunk.

## Why not use a much shorter stride?

The transformer is not the low-level controller at every simulator frame.  It
predicts a 25-step action chunk and then receives a new current observation.
Using H20/S7 spends nearly the whole history budget on the preceding 14
seconds--roughly one manipulation--while forgetting which drawers were opened
earlier.  The contact sheets show that this is the wrong allocation for the
failure mode of repeating or continuing a long search.

H20/S25 is more plausible, but its 500-step span still excludes about half of
a timeout rollout.  It is preferable only if the target behavior depends on
fine within-reach motion rather than remembering visited drawers.  The current
observation and 25-step action model already carry most of that local burden.

## Dataset-length context

The exact 25 successful demonstrations have lengths:

`72, 72, 88, 194, 196, 196, 198, 201, 217, 407, 459, 503, 550, 578, 602, 616, 634, 636, 700, 802, 863, 966, 999, 1014, 1048`.

- 40% finish by step 450.
- 56% finish by step 600.
- 76% finish by step 700.
- 16% run beyond step 900.

This explains why H12/S50 can look surprisingly strong for BC: it already
contains the complete past for over half of the successful demonstrations and
for the rendered BC success at step 450.  Its limitation appears primarily in
late-drawer and timeout cases, which are exactly the cases needed to test
long-horizon memory.

## Recommended experiment

For the main setting, keep **H20/S50** for MTQL.  For an apples-to-apples BC
comparison, BC should also use **H20/S50**; history should not be an accidental
capacity handicap.  If only one history ablation is affordable, compare it to
**H24/S40** with identical data selection, evaluation seeds, episode count,
and at least three training seeds.  That ablation isolates full-horizon sparse
memory versus nearly full-horizon denser memory.

No training jobs were launched as part of this analysis.
