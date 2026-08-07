# Formal Results — RHI-MatSci v4

## Scope

This is a five-seed offline evaluation on 15,717 historical benchmark-derived action records:

- preferential optimization: 2,730 records, 4 objective regimes;
- unique-material discovery: 10,987 records, 7 crystal-system regimes;
- extreme-property discovery: 2,000 records, 10 target regimes.

The data are offline proxy trajectories, not online MatBot trajectories or expert annotations.
Post-outcome signals are excluded from model features and visible evidence/context is sanitized.

## Protocol evidence

- Train, feedback, acceptance, and test record IDs are disjoint.
- Historical test regimes are group-disjoint from all source partitions.
- Experiment 2 target data are excluded from zero-shot source training/calibration/acceptance.
- RHI candidate selection uses one-shot acceptance shards per iteration.
- Experiment 3 uses inverse-frequency task weighting, macro calibration, and per-task test slices.

## Result summary

The current implementation does **not** establish that RHI outperforms the strong learned baselines.

| Experiment | RHI vs non-RHI seed | RHI vs static full | Main interpretation |
| --- | --- | --- | --- |
| 1. Single task | 0.0075 score difference; win rate 0.400 | -0.0026; win rate 0.733 | Essentially tied; no significant improvement |
| 2. Leave-one-task-out | 0.1004; win rate 0.200 | -0.0642; win rate 0.667 | Transfer is unstable; RHI loses to seed baseline |
| 3. Joint stability | -0.0024; win rate 0.400 | 0.0030; win rate 0.467 | Stable but not better than learned baselines |

Lower primary score is better. Differences are paired over seed/task runs; five seeds are not enough for a strong claim from the exact sign test.

## Scientific interpretation

- RHI beats the weak evidence heuristic on average, but this is not the relevant novelty comparison.
- The non-RHI learned seed and static learned gate are the decisive baselines; RHI is generally comparable, not superior.
- Leave-one-task-out is the weakest result, especially for unique-material and pairwise transfer.
- The deterministic proposer accepts 12 of 45 candidate revisions in each experiment family; recursive mutation is active but does not reliably improve held-out action selection.
- Risk must always be read with coverage and fixed-budget discovery efficiency; low risk caused by abstention is not a successful scientific agent.

## Paper status

The experiment suite is now rigorous enough to support a **diagnostic/negative result** and motivate the next method revision. It is not yet sufficient for a positive claim that RHI improves MatBot scientific discovery. The next positive-result attempt should use real trajectory utility/expert labels, a stronger LLM proposer, task-specific utility heads, and an explicit coverage–utility objective.
