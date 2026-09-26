# Shuffle v2 edited7 benchmark protocol

## Fixed data and history contract

The benchmark uses the validated edited7 view, normalization statistics, and
device cache:

- 195 episodes: 150 successful and 45 failed
- 11,540 transitions; episode lengths 37--111, median 57
- seven cue observations are local frames 0--6
- local frame 7 is the one-frame-before-motion observation
- the first sustained arm motion is local frame 8 in every edited episode
- cue expansion is index-only; image data and HDF5 files are not duplicated

The production history builder expands only local frames 0--6 by repeating
their virtual positions `hist_stride` times. Local frame 7 and all later
frames occur once. The live evaluator must use this same integer index map.

## History choice made before rollout results

H14/S6 is the fixed primary setting. The choice is based only on the edited7
dataset audit:

| history | all seven cues at motion onset | terminal >=1 cue | terminal >=2 cues | nominal span |
|---|---:|---:|---:|---:|
| H10/S6 | 100.0% | 81.0% | 65.1% | 6.0 s |
| H14/S6 | 100.0% | 96.9% | 95.9% | 8.4 s |
| H14/S7 | 100.0% | 99.0% | 98.5% | 9.8 s |
| H19/S7 | 100.0% | 100.0% | 100.0% | 13.3 s |

All candidates preserve all seven cues at the first motion decision. H14/S6
is the primary setting because it retains nearly all cue information through
the trajectory while using the shorter temporal span and lower stale-history
risk than H14/S7 or H19/S7. H14/S7 is a predeclared sensitivity check, not a
post-hoc replacement selected from policy results. H10/S6 is retained only as
an ablation because it loses cue information too early.

This analysis does not imply that MTQL will win. Superiority must be measured
with matched physical Shuffle rollouts; training loss or offline action error
alone cannot establish it.

## Required comparison

Compare the Transformer success-actor, MLP success-actor, and BC-SL at the
same checkpoint steps, data mixture, seed, image size, H14/S6 history, cue
protocol, action-chunk size, normalization statistics, and rollout task
configuration. Record every episode outcome, target/cup condition, timeout,
manual discard, and checkpoint step. Report success rate with binomial
confidence intervals and per-condition counts; do not select the best method
or checkpoint after inspecting only a favorable subset.

RL sampling remains `critic_all_outcomes_actor_success_only`; BC remains
success-only, as specified by the training contracts.
