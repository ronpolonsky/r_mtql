# Hard-BC cabinet dataset and MTQL settings

## Objective and limits

This dataset is deliberately constructed to make success-only behavioral
cloning imitate inefficient/revisiting behavior while giving offline MTQL
additional outcome supervision. It increases the probability of an MTQL > BC
gap; it cannot guarantee one before matched multi-seed training.

Two MTQL implementations are intentionally retained:

- `mtql_transformer` is exactly the committed branch agent. Its behavior flow,
  distillation, critic, and Q terms all train on the full mixed dataset.
- `mtql_transformer_v2` changes only behavior-flow fitting, distillation, and
  diagnostic action MSE to use successful (`actor_mask=1`) transitions. Its
  critic and Q-improvement term still use the full mixed dataset.

The launcher defaults to the base agent. Pass `v2` as its sixth argument, or
set `VARIANT=v2` on the 15-run wrapper, to select the masked variant explicitly.

The original, cabinet-v2, and first curated datasets are unchanged. This
selection is stored separately in
`cabinet_dataset_two_cams_more_rand_hard_bc_s25_f15`.

## Composition

- 5 efficient successful episodes covering direct through four-phase search.
- 20 successful episodes with nine phases and repeated drawer revisits.
- 15 visually audited failures with four to seven phases and no visible target.
- Failed transitions retain `actor_mask=0` as dataset metadata, but the restored
  branch version of `mtql_transformer.py` does not consume that mask. The
  standalone BC launcher instead removes failed episodes explicitly.
- The actor dataset has 20,803 transitions; 19,846 (95.4%) come from the
  deliberately inefficient success group.
- The critic dataset has 31,276 transitions; 10,473 (33.5%) are failures.

All 69 failure drawer-phase bigrams also occur in the success set. Thus the
negative data occupy the same search/motion support rather than being a set of
unrelated early crashes. Several failures have phase sequences only two edits
away from selected successes. See `analysis.json` for every nearest match.

`ep_00064` and `ep_00218` remain excluded because a target is visible despite
their failure label. `ep_00105` was considered as an additional high-revisit
success but excluded because its wrist stream has a severe render artifact.

## Why BC should be harder

BC sees only successful actions and therefore treats all twenty long revisit
trajectories as behavior to reproduce. Because training samples transitions
uniformly, those long episodes contribute 95.4% of its examples; the five
clean episodes contribute only 4.6%. The context window lets BC remember the
revisit, but gives it no label saying that the revisit was inefficient.

This does not make BC impossible. A sufficiently capable flow policy may infer
the latent search logic or learn the clean mode. The matched BC run is required
to measure the actual gap.

## Why MTQL has information BC lacks

The original MTQL behavior-flow network and critic both train on the 25
successes plus 15 phase-matched failures, exactly as the branch implementation
handles an offline mixed dataset. The standalone BC baseline sees only the 25
successes. Runtime reward relabeling uses:

- nonterminal reward: `-0.001`
- successful terminal: `+1`
- failed terminal: `-1`

This fixes the source reward problem where a short failure can outrank a long
success, while retaining a small efficiency preference among successes. At the
recommended gamma 0.9995, the worst selected success return is -0.248 and the
best failure return is -1.220, leaving a minimum return gap of 0.972. The
relabeling happens after loading and never modifies dataset files.

MTQL still has normal offline-RL risks: sparse terminal propagation,
bootstrapping error, and Q-driven out-of-distribution actions. Normalized Q,
alpha regularization, critic clipping, and the behavior flow are kept to control
those risks.

## Temporal settings

| Context | Span | All prior phases at decisions | All phases at terminal | Mean unique frames from previous phase |
|---|---:|---:|---:|---:|
| H20/S50 | 1000 | 100% | 100% | 1.89 |
| **H24/S40** | **960** | **100%** | **100%** | **2.53** |
| H28/S36 | 1008 | 100% | 100% | 2.93 |

H24/S40 remains the predicted best tradeoff. H20 is cheaper but temporally
sparse. H28 gives only 0.40 additional unique frames from the preceding phase
on average while encoding 16.7% more history images than H24.

Action chunk and execution horizon remain 25. A median drawer phase is roughly
116 steps, so the policy replans about four to five times per phase.

The attention-entropy target is held fixed at `((3.0, 3.0), (2.5, 2.5))`, the
stable setting used by the existing cabinet MTQL launcher. Trajectory geometry
supports the history/stride choice, but does not identify an optimal entropy
target; changing it inside this 15-run grid would confound temporal coverage
with critic-attention regularization. Both matched agents explicitly use two
actor transformer layers, four heads, hidden dimension 256, and ten flow steps.

## Alpha and discount grid

Previous normalized-Q alpha-300 runs had a Q actor term normalized to magnitude
1. Their weighted distillation term was 0.44--0.74 at 50k, 0.24--0.30 at 100k,
and below 0.10 by 200k. Alpha 300 is therefore the center: it constrains the
actor while the sparse critic is immature and later permits Q optimization.

The five `(gamma, alpha)` pairs are:

1. `(0.999, 300)`: shorter credit horizon.
2. `(0.9995, 100)`: aggressive Q improvement.
3. `(0.9995, 300)`: predicted best.
4. `(0.9995, 600)`: stronger behavior constraint.
5. `(1.0, 300)`: exact finite-horizon outcome objective, but without contraction.

Crossing those five pairs with H20/S50, H24/S40, and H28/S36 gives 15 MTQL
runs. Every run uses one seed, 1.5M updates, evaluation every 50k, 20 fixed-seed
evaluation episodes, action chunk 25, normalized Q, and checkpoints every 250k.

## Matched comparison

The primary matched comparison is:

- BC: hard dataset, successful episodes only, H24/S40.
- MTQL: identical successes plus all 15 failures, H24/S40, gamma 0.9995,
  alpha 300.

Use at least three training seeds for both before claiming a separation. The
20 evaluation episodes are deterministic, but training variance remains.
