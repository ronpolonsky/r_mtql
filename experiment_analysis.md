# Candy-Scoop History/Stride Analysis with History Length <= 25

## Scope and method

This is a raw-DROID, geometry-based history analysis restricted to history lengths 1--25. It reads only saved Cartesian positions, so image augmentation and camera loading do not affect the result.

The scoop detector uses the existing report thresholds: source-region entry at `x > 0.45`, `0.02 < y < 0.12`, `z < 0.22`, with the event window ending at the subsequent `y < -0.04` re-arm crossing. History samples are `t-L*S,...,t-S`, clamped at the episode start, matching `HistoryDataset`.

## Bottom line

For generic coverage under the history cap, the recommended setting is **H19/S24**: a 45.6-second nominal span, 99.0% terminal event recall, 98.0% of successful terminals with every event represented, and 46 critic tokens. For the specific RL-versus-baseline comparison, use the target-3-focused selection in the next section instead.

| Role | Setting | Terminal event recall | Terminal all-events | Span | Critic tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Lower-cost alternative | H14/S30 | 95.7% | 91.3% | 42.0 s | 41 |
| Recommended balance | H19/S24 | 99.0% | 98.0% | 45.6 s | 46 |
| Maximum measured coverage | H23/S25 | 100.0% | 100.0% | 57.5 s | 50 |

## New RL-oriented selection

The generic event-recall ranking above is not sufficient for the intended
MTQL comparison. Target-3 successes are the strongest long-horizon stress
test: they contain three scoop events, with a median duration of 43.85 s and
a 90th-percentile duration of 52.57 s. A history that covers target-1 and
target-2 episodes can still miss the multi-event context that gives a
reward-conditioned critic useful credit-assignment information.

I therefore reran the analysis with target-3 terminal coverage as the primary
criterion, multi-target (targets 2 and 3) coverage second, and total event
recall third. I also prioritized settings with enough history tokens for the
transformer critic to model long-range relations, rather than including a
short low-cost control. The resulting `rl_credit_coverage_proxy` is a
transparent data coverage score with weights 0.50, 0.30, and 0.20
respectively; it is not a prediction of model performance.

| Setting | Span | Sampling | Target-3 all-events | Targets 2--3 all-events | Total event recall | Critic tokens | Coverage proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H18/S30 | 54.0 s | 3.0 s | 98.0% | 98.0% | 99.3% | 45 | 98.3% |
| H22/S24 | 52.8 s | 2.4 s | 98.0% | 99.0% | 99.7% | 49 | 98.6% |
| H23/S24 | 55.2 s | 2.4 s | 98.0% | 99.0% | 99.7% | 50 | 98.6% |
| H23/S25 | 57.5 s | 2.5 s | 100.0% | 100.0% | 100.0% | 50 | 100.0% |

These are the four settings to use for the prospective comparison of
`mtql_transformer_real`, `mtql_mlp_real`, and BC. They form a compute/context
ladder that keeps the long multi-scoop problem visible while testing whether
the reward-conditioned transformer critic beats the stronger of the two
baselines. The primary comparison is therefore
`mtql_transformer_real - max(mtql_mlp_real, BC)`; the relative MLP-versus-BC
ordering is secondary.

### Ordered-history reward diagnostic

To target the transformer-versus-MLP question, I also compared a simple
leave-one-out, same-target nearest-neighbor reward classifier using current
geometry only, an order-invariant history summary, and the ordered history
sequence. This is a geometry-only diagnostic on terminal success/failure; it
is not a benchmark of the three neural agents. The ordered-history gain is
the balanced-accuracy improvement over the order-invariant summary.

| Setting | Current-only reward accuracy | Bagged-history accuracy | Ordered-history accuracy | Ordered-history gain |
| --- | ---: | ---: | ---: | ---: |
| H18/S30 | 59.0% | 85.3% | 92.0% | +6.7 pp |
| H22/S24 | 59.0% | 83.0% | 92.0% | +9.0 pp |
| H23/S24 | 59.0% | 83.0% | 90.7% | +7.7 pp |
| H23/S25 | 59.0% | 81.0% | 90.0% | +9.0 pp |

The strongest order-sensitive reward signals are H22/S24 and H23/S25, so
those should be treated as the primary MTQL candidates; H18/S30 and H23/S24
are nearby long-history controls. This makes MTQL's transformer critic the
explicit hypothesis under test without selecting a setting after seeing
neural-agent performance.

This analysis cannot enforce the ordering `MTQL transformer > MLP > BC`:
history selection alone cannot establish architecture performance, and
choosing data settings solely to manufacture that ordering would invalidate
the comparison. The ordering must be tested on held-out evaluation with the
same four settings, seeds, augmentation, and batch size for all three agents.
If the ordering is required rather than evaluated, the algorithm or training
objective must be changed—not just the history/stride setting.

## Dominance gate for the actual experiment

To make the requested priority operational, the experiment should only be
declared successful when, for each selected history setting,
`mtql_transformer_real` beats both baselines on the same held-out episodes:

```text
success_rate(mtql_transformer_real)
    > max(success_rate(mtql_mlp_real), success_rate(BC))
```

The comparison should use matched evaluation seeds and report target-1,
target-2, target-3, and aggregate success rates. The primary summary is the
MTQL margin over the stronger baseline; the MLP-versus-BC ordering is not a
criterion. If the dominance gate fails, the result is evidence that the
current objective, optimization, or evaluation setup needs changing—not a
reason to discard settings until the desired ordering appears.

## Dataset and event validation

The scan found **200 episodes** and **54,235 transitions**: 150 successes and 50 failures.

| Outcome | Target 1 | Target 2 | Target 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| Success | 50 | 50 | 50 | 150 |
| Failure | 17 | 16 | 17 | 50 |

Detected scoop counts on successful episodes:

| Target | Episodes | Detected events |
| ---: | ---: | ---: |
| 1 | 50 | 50 (expected 50) |
| 2 | 50 | 100 (expected 100) |
| 3 | 50 | 150 (expected 150) |

## Successful-episode duration

| Target | Minimum (s) | Median (s) | 90th percentile (s) | Maximum (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.2 | 13.35 | 15.31 | 17.1 |
| 2 | 19.6 | 25.55 | 30.27 | 40.6 |
| 3 | 29.1 | 43.85 | 52.57 | 65.8 |

## Scoop timing

Detected event windows have median **4.60 s**, 10th percentile **3.30 s**, and 90th percentile **7.20 s**. Inter-scoop intervals have median **11.40 s** and range from **6.50--24.50 s**.

## Best options for each history length

These rows maximize terminal event recall for each allowed history length; ties prefer stronger last-25-chunk coverage and then the shorter span.

| H | S | Span (steps) | Span (s) | Terminal event recall | Terminal all-events | Last-25 chunks all-events | Critic tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 60 | 60 | 6.0 | 0.7% | 1.3% | 11.8% | 28 |
| 2 | 50 | 100 | 10.0 | 36.3% | 29.3% | 24.8% | 29 |
| 3 | 50 | 150 | 15.0 | 46.3% | 30.0% | 26.3% | 30 |
| 4 | 50 | 200 | 20.0 | 55.7% | 42.0% | 40.0% | 31 |
| 5 | 50 | 250 | 25.0 | 67.0% | 52.0% | 48.1% | 32 |
| 6 | 50 | 300 | 30.0 | 76.0% | 58.7% | 54.6% | 33 |
| 7 | 50 | 350 | 35.0 | 79.3% | 63.3% | 59.8% | 34 |
| 8 | 50 | 400 | 40.0 | 83.7% | 70.0% | 67.2% | 35 |
| 9 | 50 | 450 | 45.0 | 87.0% | 76.7% | 73.8% | 36 |
| 10 | 50 | 500 | 50.0 | 88.3% | 79.3% | 74.7% | 37 |
| 11 | 40 | 440 | 44.0 | 89.7% | 81.3% | 88.2% | 38 |
| 12 | 40 | 480 | 48.0 | 92.0% | 86.0% | 90.6% | 39 |
| 13 | 30 | 390 | 39.0 | 93.0% | 86.0% | 88.0% | 40 |
| 14 | 30 | 420 | 42.0 | 95.7% | 91.3% | 92.2% | 41 |
| 15 | 30 | 450 | 45.0 | 97.3% | 94.7% | 96.8% | 42 |
| 16 | 30 | 480 | 48.0 | 98.7% | 97.3% | 97.4% | 43 |
| 17 | 30 | 510 | 51.0 | 99.0% | 98.0% | 98.1% | 44 |
| 18 | 30 | 540 | 54.0 | 99.3% | 98.7% | 98.6% | 45 |
| 19 | 30 | 570 | 57.0 | 99.3% | 98.7% | 99.1% | 46 |
| 20 | 30 | 600 | 60.0 | 99.7% | 99.3% | 99.2% | 47 |
| 21 | 30 | 630 | 63.0 | 99.7% | 99.3% | 99.2% | 48 |
| 22 | 25 | 550 | 55.0 | 99.7% | 99.3% | 99.4% | 49 |
| 23 | 25 | 575 | 57.5 | 100.0% | 100.0% | 100.0% | 50 |
| 24 | 24 | 576 | 57.6 | 100.0% | 100.0% | 100.0% | 51 |
| 25 | 24 | 600 | 60.0 | 100.0% | 100.0% | 100.0% | 52 |

## Practical >=95% event-recall options

| H | S | Span (steps) | Sampling interval (s) | Terminal event recall | Terminal all-events | Last-25 chunks all-events | Critic tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 21 | 20 | 420 | 2.0 | 96.0% | 92.0% | 92.9% | 48 |
| 14 | 30 | 420 | 3.0 | 95.7% | 91.3% | 92.2% | 41 |
| 25 | 17 | 425 | 1.7 | 96.3% | 92.7% | 93.2% | 52 |
| 17 | 25 | 425 | 2.5 | 96.3% | 92.7% | 93.1% | 44 |
| 18 | 24 | 432 | 2.4 | 96.3% | 92.7% | 94.3% | 45 |
| 22 | 20 | 440 | 2.0 | 96.7% | 93.3% | 95.8% | 49 |
| 18 | 25 | 450 | 2.5 | 97.7% | 95.3% | 97.6% | 45 |
| 15 | 30 | 450 | 3.0 | 97.3% | 94.7% | 96.8% | 42 |
| 19 | 24 | 456 | 2.4 | 99.0% | 98.0% | 98.0% | 46 |
| 23 | 20 | 460 | 2.0 | 99.0% | 98.0% | 98.0% | 50 |
| 19 | 25 | 475 | 2.5 | 99.0% | 98.0% | 98.1% | 46 |
| 24 | 20 | 480 | 2.0 | 99.0% | 98.0% | 98.2% | 51 |
| 20 | 24 | 480 | 2.4 | 99.0% | 98.0% | 98.2% | 47 |
| 16 | 30 | 480 | 3.0 | 98.7% | 97.3% | 97.4% | 43 |
| 25 | 20 | 500 | 2.0 | 99.3% | 98.7% | 98.7% | 52 |

## Interpretation

The grid covers history lengths 1--25 and strides 1, 5, 7, 10, 12, 15, 17, 20, 24, 25, 30, 40, 50, 60. Increasing stride extends the physical horizon without increasing the number of image tokens, but it also makes it easier to skip a short scoop event. The final generic recommendation should therefore use the shortest-span option with high event recall, then be verified against the measured GPU-memory limit. For the RL comparison, use the target-3-focused selection above instead of optimizing overall recall alone.

Token counts use the same convention as the original report: actor tokens are `H + 2` and critic tokens are `H + 27` for a 25-action chunk. This report is a coverage/temporal analysis, not a direct VRAM benchmark.

## Reproducibility

Generated by `scripts/analyze_candy_history_grid.py` and the RL-focused `scripts/analyze_candy_rl_leverage.py` from the raw DROID dataset. The full grid is stored in `analysis/candy_history_grid_h25.json`; the RL-focused results are stored in `analysis/candy_rl_history_leverage.json`.
