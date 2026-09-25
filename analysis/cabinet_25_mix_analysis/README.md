# Cabinet 25-success mixture analysis

This analysis uses the exact reconstructed training split consumed by
`search_cabinet_two_cams_more_rand`. The split order is reproduced with the
same `random.Random(42)` shuffle used by reconstruction, so "first 25
successes" and "first n failures" match `_select_episode_mix`.

## Recommendation

- Keep BC fixed at **25 successful episodes and 0 failed episodes**.
- With MTQL exactly as currently implemented, use **25 successes + 5
  failures**. This gives useful negative coverage while only 15.3% of
  transition-uniform replay samples come from failed episodes.
- For task memory, the follow-up inspection of actual BC and MTQL evaluation
  rollouts favors **history length 20, stride 50**. It spans the full 1000-step
  evaluation horizon while retaining roughly 2--3 samples per measured drawer
  phase. **Length 24, stride 40** is the denser, higher-cost alternative. See
  [`../cabinet_rollout_history_report/README.md`](../cabinet_rollout_history_report/README.md)
  for the rollout-level measurements.

The recommendation for only five failures is conditional on current code:
offline failed trajectories do not contain `actor_mask`, so the MTQL flow
actor behavior-clones them in addition to the critic learning from them. If
failed offline episodes are masked out of actor imitation while remaining in
critic training, **10 failures** becomes a reasonable next candidate.

## Exact data composition

The selected 25 successful episodes contain 12,811 transitions. Their length
median is 550, range 72–1048. Eleven of the 25 are at least 600 transitions,
so transition-uniform replay is already strongly weighted toward later-drawer
search states.

| Failures | Failure transitions | Failure share of replay | Expected failures in batch 32 | Length-band coverage |
|---:|---:|---:|---:|---|
| 0 | 0 | 0.0% | 0.0 | none |
| 3 | 1,365 | 9.6% | 3.1 | 1 short, 1 medium, 1 long |
| **5** | **2,318** | **15.3%** | **4.9** | **2 short, 1 medium, 2 long** |
| 10 | 3,396 | 21.0% | 6.7 | 6 short, 2 medium, 2 long |
| 15 | 4,720 | 26.9% | 8.6 | 9 short, 4 medium, 2 long |
| 20 | 6,450 | 33.5% | 10.7 | 12 short, 5 medium, 3 long |
| 25 | 7,725 | 37.6% | 12.0 | 15 short, 7 medium, 3 long |
| 82 | 28,866 | 69.3% | 22.2 | all failures |

Length bands are empirical: short <180, medium 330–599, long >=600. Only a
few failures fall in 180–329.

The exact first five failures selected by the loader are:

1. `ep_00098`, 534 transitions: substantial search progress before failure.
2. `ep_00184`, 138 transitions: first-drawer manipulation failure.
3. `ep_00176`, 693 transitions: multiple drawer phases before failure.
4. `ep_00197`, 800 transitions: multiple drawer phases; the wrist view even
   shows the object/open drawer late in the rollout although the episode has
   no positive reward, making this a potentially ambiguous negative.
5. `ep_00015`, 153 transitions: another short first-operation failure.

Increasing from five to ten failures adds one medium episode and four more
short episodes. It therefore adds more mechanical-abort density than new
long-horizon memory coverage.

## What the trajectories show

These failures are not curated "wrong drawer continuation" or "early DONE"
examples. They were produced by the same planned random drawer-search policy
with noisy handle alignment. Long failures contain valid open/close/search
prefixes and then fail mechanically. Short failures usually abort during the
first drawer operation. This is useful critic data, but it is not a clean
negative-action dataset and should not dominate flow-matching imitation.

The long successful trajectory `ep_00003` demonstrates why a short temporal
window is insufficient: it traverses several open/close cycles and succeeds
after 1048 transitions. Near t=1000:

| History | Physical span | Time span | Avg. distinct history states over selected successes | Fraction of sampled transitions whose history reaches episode start |
|---|---:|---:|---:|---:|
| H12 / S7 | 84 | 8.4 s | 11.10 | 16.4% |
| H20 / S7 | 140 | 14 s | 17.46 | 26.0% |
| H20 / S10 | 200 | 20 s | 16.45 | 36.2% |
| H20 / S25 | 500 | 50 s | 12.30 | 72.7% |
| H20 / S40 | 800 | 80 s | 9.09 | 93.1% |
| H20 / S50 | 1000 | 100 s | 7.53 | 99.5% |
| **H24 / S40** | **960** | **96 s** | **9.28** | **98.6%** |

H12/S7 through H20/S10 mainly show the current drawer maneuver. They cannot
tell the model which cabinets were visited earlier. H20/S25 is a useful local
motion/search compromise but can lose the earliest drawer in the longest
episodes. H20/S50 retains essentially the whole episode, but early and medium
episodes contain many repeated episode-start padding frames. H24/S40 preserves
almost the same global horizon with denser temporal samples.

With the ResNet and `per_modality` configuration, each state's two camera
embeddings and proprioception are concatenated and become one transformer
observation token. Therefore:

| History length | Actor sequence | Critic sequence with 25-action chunk |
|---:|---:|---:|
| 12 | 14 tokens | 39 tokens |
| 20 | 22 tokens | 47 tokens |
| 24 | 26 tokens | 51 tokens |

Counts include the CLS token and current observation; critic counts also
include 25 action tokens. H24 costs 20% more image-history encoding than H20,
while critic attention grows from 47 to 51 tokens.

## Important implementation observations

1. Offline failure transitions currently have no `actor_mask`, and
   `MTQLTransformerAgent` defaults a missing mask to all ones. Thus both
   `bc_flow_loss` and the alpha=300 distillation path consume failed demos.
2. The current V2 launcher has `online_warmup_steps=1500000` and
   `train_steps=1500000`. It is effectively offline RL: collection begins only
   on the final update and can add at most one 25-step chunk.
3. Evaluation uses a 1000-step simulator horizon, but two of the selected 25
   successful demos have lengths 1014 and 1048. Any horizon change must be
   applied equally to BC and MTQL.
4. The launcher count arguments are now wired through to Python. Before this
   analysis, positional values were parsed but literal `25` and `5` were sent
   to `m_main.py`.

## Artifacts

- `representative_trajectories.pdf`: four successful trajectories spanning
  short to very long searches, the exact first five failures, and failures
  six through ten.
- `history_window_comparison_ep00003.pdf`: exact two-camera history frames at
  several decision points.
- `history_ep00003_t1000_extended.png`: direct comparison including H20/S40
  and H24/S40.
- `candidate_mixtures.json`: exact prefix composition for candidate values of
  n.
- `train_episode_inventory.csv`: episode order, outcome, length, seed, and
  motion statistics.

This is a dataset/architecture recommendation, not an empirical claim that a
particular n wins evaluation. The clean confirming ablation is three seeds
each for n in {0, 5, 10}, with BC held at the same 25 successes and all other
settings fixed.
