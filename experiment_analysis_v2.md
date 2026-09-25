# Candy-Scoop v2 history/stride recommendation

## Recommendation

Use **history length 20 and stride 25** (`H20/S25`) as the primary setting
for the matched comparison between:

- `mtql_transformer_real` (RL, transformer critic)
- `mtql_mlp_real` (RL, flattened MLP critic)
- `new_bc_flow_transformer_real` (BC baseline)

This is the best primary setting for the intended comparison, not a claim that
history selection alone can guarantee the ordering
`MTQL-transformer > MLP > BC`. That ordering must be measured with matched
training and held-out evaluation.

The distinct transformer-focused secondary setting is **H18/S30**. It has a
larger order-sensitive reward-separation signal and is a more meaningful
transformer hypothesis test than H21/S24, while sacrificing some event
coverage.

## Exact data analyzed

The analysis uses the same read-only-derived view used by training:

`/iris/u/ronpo/expo-ft-data/candy_scoop_v2_jitted`

It contains 144 valid episodes and 38,534 transitions:

| Outcome | target 1 | target 2 | target 3 | total |
| --- | ---: | ---: | ---: | ---: |
| Success | 30 | 30 | 30 | 90 |
| Failure | 14 | 18 | 22 | 54 |

The raw v2 tree initially contained 158 numeric episode directories. Fourteen
were empty HDF5 recording placeholders with no datasets, so they could not
provide a single observation or action and were removed from the raw tree:

- target 1 failures: `2, 3, 4, 6, 8, 10, 11, 13`;
- target 2 failures: `3, 7, 11, 14`;
- target 3 failures: `0, 8`.

The remaining raw target directories now contain the same 44/48/52 valid
episodes represented by the derived view. No valid trajectory was removed.

Fifteen valid cross-target success trajectories were intentionally added to
the failure side as counterfactual negatives. They are IDs `100--104` in each
target group, with exact provenance verified as:

| Failure label | Source success trajectory |
| --- | --- |
| target 1 / failure 100--104 | target 2 / success 0--4 |
| target 2 / failure 100--104 | target 3 / success 0--4 |
| target 3 / failure 100--104 | target 1 / success 0--4 |

Thus the `failure` label means failure for the requested target cue; the five
counterfactual trajectories in each group are successful executions of a
different target, not failed recordings.

The v2 HDF5 files use `saved_observation/cartesian_position`; the analysis
reader supports that layout and the older `action/cartesian_position` layout.
The training view is read-only and symlink-backed to avoid duplicating the
large HDF5/image files:

`/iris/u/ronpo/expo-ft-data/candy_scoop_v2_jitted`

The actual materialized cache is separate:

`/iris/u/ronpo/mtql-runs/candy_scoop_v2_jitted_cache_v2norm`

It contains the resized image arrays and normalized numeric arrays for all 144
valid episodes. The v2 normalization statistics were recomputed from all
38,534 transitions at:

`/iris/u/ronpo/projects/new_mtql_candy_scooping/norm_stats/candy_scoop_v2_jitted/norm_stats.json`

The previous cache using the old dataset's normalization statistics was
removed. Training must use the `_v2norm` cache and the v2 stats above. This
follows the v1 workflow: v1 used the materialized
`/iris/u/ronpo/mtql-runs/candy_scoop_cache_full` cache, which contained
preprocessed `.npy` arrays for 200 episodes and 54,235 transitions. The v2
`_v2norm` directory is the corresponding materialized cache for the corrected
v2 dataset. The additional `candy_scoop_v2_jitted` directory is only the
read-only selection view used before materialization; it avoids copying raw
HDF5 files. Neither cache stores augmentation: augmentation remains a
runtime JAX/JIT operation so history, stride, and augmentation RNG can vary
between runs.

## Selection method

The grid evaluates every history length 1--25 and strides
`1, 5, 7, 10, 12, 15, 17, 20, 24, 25, 30, 40, 50, 60`.

For each setting it measures:

1. terminal event recall;
2. the fraction of successful terminals whose complete detected event history
   is represented;
3. coverage in the final 25-action chunk;
4. target-3 and target-2/3 multi-scoop coverage;
5. an ordered-history reward-separability diagnostic versus a permutation-
   invariant history summary;
6. critic token count and physical time span.

The dataset is sampled at 10 Hz. With a 25-action chunk, the critic has
`H + 27` tokens and the actor has `H + 2` tokens. History samples are
`t-H*S, ..., t-S`, matching the production `HistoryDataset` semantics.

## Candidate comparison

| Setting | Span | Critic tokens | Target-3 all-events | Target 2/3 all-events | Event recall | Ordered-history gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H19/S25 | 47.5 s | 46 | 86.7% | 90.0% | 96.2% | 0.0 pp |
| **H20/S25** | **50.0 s** | **47** | **90.0%** | **91.7%** | **96.8%** | **0.0 pp** |
| H21/S24 | 50.4 s | 48 | 80.0% | 88.3% | 95.5% | **+2.0 pp** |
| H23/S24 | 55.2 s | 50 | 80.0% | 88.3% | 95.5% | **+2.0 pp** |
| H23/S25 | 57.5 s | 50 | 90.0% | 91.7% | 96.8% | +0.6 pp |
| H25/S20 | 50.0 s | 52 | 86.7% | 91.7% | 96.2% | -1.3 pp |

Target-stratified 2,000-resample bootstrap intervals support the same
selection. H20/S25 has 96.8% event-recall point estimate (95% interval
94.2--99.3%) and 90.0% target-3 complete-event coverage (76.7--100.0%).
H23/S25 has indistinguishable coverage intervals, but uses 50 rather than 47
critic tokens; H19/S25 and H25/S20 have lower target-3 point coverage.

I also checked the tempting order-first alternatives H16/S30, H17/S30, and
H18/S30. Their ordered-history diagnostic is larger (+7.2, +6.3, and +5.4
percentage points), but their target-3 complete-event coverage is only 80.0%,
83.3%, and 83.3%, respectively, with total event recall below 94%. They are
therefore useful transformer-specific ablations, not the primary setting:
they discard too much of the long-horizon reward context.

### Why H20/S25 is primary

- It reaches the best observed low-token coverage: 96.8% event recall and
  90.0% target-3 all-event coverage.
- H23/S25 does not improve those coverage numbers, but costs three additional
  critic tokens and has a larger input for the MLP baseline.
- The MLP implementation explicitly documents its parameter matching at
  `H20/C25`; H20/S25 therefore gives the cleanest architecture comparison.
- H19/S25 is a useful lower-cost control, but loses target-3 coverage.

### Why H18/S30 is the distinct transformer-focused alternative

- It samples every 3.0 seconds instead of every 2.5 seconds, giving a
  materially different temporal view rather than a near-duplicate of H20/S25.
- Its ordered-history reward-separation gain is +5.4 percentage points over
  the permutation-invariant summary, versus 0.0 pp for H20/S25.
- It uses 45 critic tokens, fewer than H20/S25.
- Its cost is lower event coverage: 93.6% total event recall and 83.3%
  target-3 all-event coverage. That makes it a secondary transformer
  hypothesis test, not the safest overall setting.

## Important data-quality observation

The geometric detector found 30, 48, and 79 scoop events for target labels
1, 2, and 3 respectively, rather than the nominal 30, 60, and 90. This is a
property of the recorded trajectories/threshold detector, not a reason to
choose a longer history. The comparisons above use the detected events
consistently across all settings.

## Experimental gate

Train all three agents with exactly the same v2 cache, augmentation, seed,
batch size, train steps, and H/S setting. Evaluate the same held-out episodes
by target number. Declare the transformer advantage only if:

```text
success_rate(mtql_transformer_real)
    > max(success_rate(mtql_mlp_real), success_rate(new_bc_flow_transformer_real))
```

History/stride analysis can select a setting that exposes long-range,
reward-relevant context; it cannot establish neural architecture ranking
without this matched training/evaluation.

## Matched v2 training protocol

The matched runs use the same dataset/cache, seed, batch size, history, stride,
augmentation, and optimization settings:

- visual cue project: `real_candy_scoop_v2_visual`;
- language cue project: `real_candy_scoop_v2_language`;
- primary setting: `HIST_LENGTH=20`, `HIST_STRIDE=25`;
- transformer-focused secondary: `HIST_LENGTH=18`, `HIST_STRIDE=30`;
- `BATCH_SIZE=16`;
- `TRAIN_STEPS=1,000,000`;
- checkpoint and logging interval: every 10,000 steps;
- `P_AUG=1.0`, seed `0`, attention target `((3.5,3.5),(3.0,3.0))`;
- `normalize_q_loss=false` for MTQL/MLP runs;
- all runs use the v2 derived dataset, v2 normalization stats, and corrected
  `_v2norm` cache.

The H20/S25 combined launcher is
`scripts/launch_candy_scoop_v2_h20s25_both_cues.sh`. The H18/S30 Sphinx
launcher is `scripts/launch_v2_h18s30_sphinx_both_cues.sh`.

## Reproducibility artifacts

- Full v2 coverage grid: `analysis/candy_history_grid_v2_h25.json`
- Candidate reward/order diagnostic:
  `analysis/candy_rl_history_leverage_v2_recommended.json`
- Transformer-order tradeoff diagnostic:
  `analysis/candy_rl_history_leverage_v2_transformer_tradeoff.json`
- Target-stratified bootstrap uncertainty:
  `analysis/candy_v2_history_bootstrap.json`
- Matched three-agent launch protocol:
  `scripts/launch_candy_scoop_v2_h20s25_matched.sh`
- Analysis readers:
  `scripts/analyze_candy_history_grid.py`,
  `scripts/analyze_candy_rl_leverage.py`, and
  `scripts/bootstrap_candy_v2_history.py`
