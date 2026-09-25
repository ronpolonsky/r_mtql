# Four-cube drawer memory task

This directory is isolated from the original cabinet task and datasets.

Each episode contains four differently colored cubes, one per drawer. The
target cube is shown as a visual cube cue that fades over the first ten 10 Hz
control steps. The robot searches drawers, immediately grasps the target once
found, and places it on the table. Opening distractor drawers is valid.

Success requires target placement. Wrong-object grasp and the 2,000-step
timeout are terminal failures. Ordinary steps receive `-0.0001`, opening the
target drawer adds `+0.25`, successful placement terminates with `+0.75`, and
terminal failures receive `-1`.

The v1 collection contains 300 successful and 50 failed training episodes.
Episodes are stored individually so later experiments can select exact
success/failure subsets without rewriting the source data.

`train.sh` accepts `NUM_SUCCESS_DEMOS` and `NUM_FAILURE_DEMOS`. Subsets are
selected deterministically and balanced across search strategy, target drawer,
target color, and episode-length quartile. For example, the 25-success subset
has five examples of each success strategy and target drawer counts of
7/7/6/5; the 15-failure subset has eight wrong-object grasps and seven
revisit/timeouts.

For evaluation, use `evaluation/success`. The inherited
`evaluation/raw_success`/`evaluation/open_enough` metrics only report that the
target drawer opened; they do not require retrieving the cube.

The controlled 25-success comparison is in `submit_controlled_pilot.sh`. It
uses the normal `agents/mtql_transformer.py`, not the success-masked v2 agent.
Both agents use batch size 8 because the MTQL actor+critic update at batch size
32 does not fit on 24 GB GPUs.

For the full one-seed allocation, `submit_remaining_iris_hi_seed1.sh` adds four
jobs to the two-job H24/S80 anchor (six iris-hi jobs total), while
`submit_iris_seed1_sweep.sh` submits exactly 10 regular-iris jobs: one H20/S100
BC baseline and a 3x3 MTQL grid over 5/10/15 failures and alpha 30/100/300.
Regular-iris jobs use batch size 4 so the MTQL update fits on the 8 GB GPUs in
that partition; iris-hi jobs use batch size 8.

The higher-data follow-up uses `submit_higher_data_iris_hi.sh` (exactly six
jobs) and `submit_higher_data_iris.sh` (exactly ten jobs). It compares 75 and
100 successful episodes, 15 and 30 failed episodes, and alpha 30/100/300. The
RL follow-up uses `MTQL_VARIANT=v2`, which keeps failed transitions in critic
training while masking them from behavior-flow fitting and distillation. These
runs log to the separate W&B project `offline-drawer-task-v1-s75-s100`.

`submit_higher_data_iris_history.sh` adds exactly ten regular-Iris MTQL v2
runs at alpha 100. It completes the H16/S125, H24/S84, and H32/S64 comparisons
for the S75/S100 and F15/F30 mixtures; every window spans at least the full
2,000-step episode.

The intermediate-data follow-up uses `submit_mid_data_iris_hi.sh` (exactly six
jobs) and `submit_mid_data_iris.sh` (exactly ten jobs). It tests 150 and 200
successful demonstrations, keeping BC below the rejected all-300 regime while
giving MTQL more successful action coverage than S75/S100. MTQL v2 additionally
uses 30/50 failures. The primary sweep compares four approximately 2,000-step
history windows: H16/S125, H20/S100, H24/S84, and H32/S64. These runs log to
the separate W&B project `offline-drawer-task-v1-s150-s200`. H32 is restricted
to Iris-Hi because H32/batch-4 previously exceeded regular Iris GPU memory.
