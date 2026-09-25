# Cabinet offline-RL curation: 25 successes + 15 failures

## Intended learning split

- Actor / behavior flow / distillation: 25 successful episodes only.
- Critic: the same 25 successes plus 15 targeted failures.
- The split is enforced per transition by `actor_mask`, not merely by episode
  ordering. Failed transitions have `actor_mask=0`.
- The original source dataset and validation split are unchanged.

The reports in `candidate_reports/` show 25 uniformly spaced states per
episode. Each state stacks the agent camera over the wrist camera; the bottom
panel plots the first seven proprio joints, distance from the initial pose, and
detected leave-home/return phases.

The temporal phase map uses every proprio state in all 40 selected episodes
(29,741 states). The visual audit uses the 25-state, two-camera contact sheet
for each of 48 candidates (2,400 camera images total), including the candidates
later excluded. Thus “all frames” in the numeric coverage results means all
proprio time steps; visual conclusions come from dense uniform contact sheets,
not from claiming manual inspection of all 59,482 raw camera images.

## Successful actor trajectories

The first ten form a search-depth curriculum. They cover direct success and
then progressively longer sequences of opening, inspecting, closing, returning
home, and continuing to another drawer.

| Episode | Steps | Proprio phases | Audit result |
|---|---:|---:|---|
| `ep_00000` | 70 | 1 | Direct success; clean terminal target discovery |
| `ep_00044` | 88 | 1 | Direct success; clean terminal target discovery |
| `ep_00232` | 193 | 2 | One continuation before success |
| `ep_00012` | 217 | 2 | One continuation before success |
| `ep_00219` | 389 | 4 | Multi-drawer continuation, then success |
| `ep_00045` | 503 | 4 | Multi-drawer continuation, then success |
| `ep_00113` | 602 | 6 | Long clean search, then success |
| `ep_00199` | 700 | 6 | Long clean search, then success |
| `ep_00094` | 750 | 7 | Near-exhaustive clean search, then success |
| `ep_00174` | 863 | 7 | Near-exhaustive clean search, then success |

The next fifteen are deliberately long successful trajectories. Every one has
nine detected proprio phases and repeated negative inspections before its final
positive reward. They provide positive supervision for "continue searching"
instead of allowing the successful set to collapse to easy one-drawer demos.

| Episode | Steps | Proprio phases | Audit result |
|---|---:|---:|---|
| `ep_00155` | 920 | 9 | Repeated search/recovery; terminal success |
| `ep_00157` | 1077 | 9 | Repeated search/recovery; terminal success |
| `ep_00179` | 1069 | 9 | Repeated search/recovery; terminal success |
| `ep_00236` | 1057 | 9 | Repeated search/recovery; terminal success |
| `ep_00222` | 999 | 9 | Repeated search/recovery; terminal success |
| `ep_00119` | 995 | 9 | Repeated search/recovery; terminal success |
| `ep_00058` | 994 | 9 | Repeated search/recovery; terminal success |
| `ep_00128` | 988 | 9 | Repeated search/recovery; terminal success |
| `ep_00092` | 985 | 9 | Repeated search/recovery; terminal success |
| `ep_00196` | 975 | 9 | Repeated search/recovery; terminal success |
| `ep_00168` | 970 | 9 | Repeated search/recovery; terminal success |
| `ep_00096` | 969 | 9 | Repeated search/recovery; terminal success |
| `ep_00063` | 968 | 9 | Repeated search/recovery; terminal success |
| `ep_00051` | 966 | 9 | Repeated search/recovery; terminal success |
| `ep_00060` | 961 | 9 | Repeated search/recovery; terminal success |

## Targeted critic-only failures

All fifteen show plausible drawer-search behavior, contain at least four
proprio phases, and end without a visible colored target or a positive reward.
The long failures overlap heavily with successful search prefixes, which makes
them useful hard negatives for long-horizon continuation rather than examples
of irrelevant early collisions.

| Episode | Steps | Proprio phases | Audit result |
|---|---:|---:|---|
| `ep_00203` | 498 | 4 | Two-drawer search; no target; failure |
| `ep_00226` | 879 | 7 | Long continuation/recovery; no target; failure |
| `ep_00244` | 855 | 7 | Long continuation; no target; failure |
| `ep_00116` | 854 | 7 | Long continuation; no target; failure |
| `ep_00143` | 852 | 7 | Long continuation; no target; failure |
| `ep_00177` | 783 | 6 | Multi-drawer continuation; no target; failure |
| `ep_00037` | 775 | 6 | Multi-drawer continuation; no target; failure |
| `ep_00120` | 761 | 6 | Multi-drawer continuation; no target; failure |
| `ep_00159` | 736 | 6 | Multi-drawer continuation; no target; failure |
| `ep_00176` | 693 | 6 | Multi-drawer continuation; no target; failure |
| `ep_00053` | 691 | 6 | Multi-drawer continuation; no target; failure |
| `ep_00098` | 534 | 4 | Two-drawer search; no target; failure |
| `ep_00217` | 538 | 4 | Two-drawer search; no target; failure |
| `ep_00095` | 522 | 4 | Two-drawer search; no target; failure |
| `ep_00249` | 502 | 4 | Two-drawer search; no target; failure |

## Explicit exclusions

- `ep_00064`: rejected because the colored target is visible near step 1044
  despite the episode being labeled failure.
- `ep_00218`: rejected because a colored target is visible late in the
  trajectory despite the failure label.
- Short one-phase failures were not used because they are primarily mechanical
  failures and provide little supervision about search order or continuation.

## Context choice

The detected drawer interaction phases last 58--200 physical steps (median
116), while selected successful episodes have median length 966 and maximum
length 1077. Context coverage was evaluated at every detected phase boundary,
not inferred from episode length alone. H20/S50, H24/S40, and H28/S36 all
capture every prior phase at every phase boundary and at every episode end in
this selection. Their oldest tokens are 1000, 960, and 1008 physical steps
back, respectively.

H24/S40 is the data-derived recommendation. At the start of a new phase it
retains an average 2.55 unique frames from the immediately preceding phase,
versus 1.90 for H20/S50 and 2.97 for H28/S36. It therefore recovers most of the
resolution gain without the image-encoder cost of H28/S36. With this
implementation it gives the actor 26 total tokens
(24 history + current + CLS) and the critic 51 (24 history + current + 25
action-chunk + CLS). H16/S64 was rejected despite its 1024-step span because
its coarse samples miss at least one immediately preceding phase in 2.3% of
phase boundaries and miss complete terminal phase coverage in 15% of selected
episodes. H12/S50 captures every phase at the end of only 42.5% of episodes.

The detailed measurements are in `context_grid_analysis.json`.

## Reward semantics

The source simulator reward is -1 on every non-success step and +1 at success.
In the curated data this means a 500-step failed episode can have a higher
discounted return than a roughly 966-step successful search. That is unsuitable
for using failures as critic-only outcome supervision. The recommended launcher
therefore selects the `terminal_sparse` environment alias, which reads the
same source arrays but relabels rewards in memory to 0 on nonterminal steps,
+1 on successful terminals, and -1 on failed terminals. No dataset file is
modified, and the original reward semantics remain available through the old
environment alias.

For the selected 500--1077 step trajectories, gamma 0.99 retains only 0.0043%
of a terminal outcome over 1000 steps. The proposed grid consequently tests
0.997, 0.999, 0.9995, 0.9999, and 1.0. The recommendation is 0.9995: it retains
60.6% of a 1000-step terminal signal while remaining a contraction; 1.0 is
included as the exact episodic-outcome objective but may be less numerically
stable under bootstrapping.

## Previous loss-scale audit and controlled 15-setting grid

The local logs for the previous cabinet-v2 H20/S50 runs contain no run that
reached 1.5M updates; the two most informative normalized-Q runs reached 410k
and 640k. Both used alpha 300. Their policy Q loss was normalized to 1.0, while
the weighted distillation loss evolved as follows:

| Step | `300 * distill_loss`, seed 1 | `300 * distill_loss`, seed 3 |
|---:|---:|---:|
| 50k | 0.439 | 0.737 |
| 100k | 0.240 | 0.296 |
| 200k | 0.039 | 0.090 |
| 400k | 0.016 | 0.018 |

This makes alpha 300 a useful scale-matched anchor: imitation regularizes the
actor early, then the Q term dominates after the one-step actor has learned to
match the behavior flow. The 15-run budget is spent on context resolution and
long-horizon credit, which can be inferred from the curated trajectories;
there is no prior evidence from which to infer an alpha optimum.

The normalized-Q runs reached best evaluation success 0.55 and 0.65. Similar
unnormalized-Q runs had policy Q losses around 53--72 and collapsed to zero tail
success, so `normalize_q_loss=true` is retained in every proposed setting.

The prior mixed-data runs reported `actor_batch_fraction=1.0`, meaning their
failed transitions were also treated as actor targets. The curated dataset
should instead report approximately `19268 / 29741 = 0.648`; a materially
different value is a loading/configuration error.

The controlled grid uses one common seed and the Cartesian product of contexts
`H20/S50`, `H24/S40`, `H28/S36` and discounts `0.997`, `0.999`, `0.9995`,
`0.9999`, `1.0`, for exactly 15 runs. Alpha remains 300, normalized Q is on,
training lasts 1.5M updates, and deterministic 20-episode evaluation runs every
50k updates. The predicted best setting is H24/S40, gamma 0.9995, alpha 300.
