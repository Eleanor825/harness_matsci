# Experiment 4 — Self-Evolution Ablation Results

## Scope

Five-seed historical offline-proxy evaluation of the active harness checkpoints
`H0`, `H1`, `H2`, and `H3` on pairwise optimization, unique-material
discovery, and extreme-property discovery. The same train, feedback,
acceptance, and untouched test partitions are used for both policies.

- `guarded`: accept a candidate only when the independent acceptance shard
  selects it over its predecessor.
- `always_accept`: accept every schema-valid candidate; it never uses test
  performance to select a checkpoint.
- Every checkpoint is evaluated on test only after all selection decisions.
- Lower primary score is better.

## Aggregate Results

| Checkpoint | Policy | Primary score | Budget hit rate | Oracle-normalized utility |
| --- | --- | ---: | ---: | ---: |
| H0 | guarded / always-accept | 0.4390 ± 0.0449 | 0.3805 ± 0.1686 | 0.5396 ± 0.2595 |
| H1 | guarded | 0.4436 ± 0.0523 | 0.3586 ± 0.1849 | 0.5537 ± 0.2436 |
| H2 | guarded | 0.4487 ± 0.0636 | 0.3399 ± 0.1986 | 0.5706 ± 0.2286 |
| H3 | guarded | 0.4465 ± 0.0649 | 0.3440 ± 0.2021 | 0.5792 ± 0.2223 |
| H1/H2/H3 | always-accept | 0.4455 ± 0.0655 | 0.3427 ± 0.1962 | 0.5923 ± 0.2092 |

## Paired Change From H0

| Policy/checkpoint | Δ primary score | Checkpoint wins | Exact sign-test p |
| --- | ---: | ---: | ---: |
| guarded H1 | +0.0045 | 0.067 | 0.6250 |
| guarded H2 | +0.0096 | 0.267 | 0.7539 |
| guarded H3 | +0.0075 | 0.400 | 1.0000 |
| always-accept H1/H2/H3 | +0.0065 | 0.533 | 1.0000 |

A positive Δ is worse because lower score is better. None of the checkpoint
comparisons shows a statistically convincing improvement over H0.

## Task-Level Pattern

- Unique-material discovery: guarded improves from `0.4104` at H0 to `0.4056`
at H3; always-accept improves to `0.4027` at H1, but this is task-local and
not reflected in the macro result.
- Extreme-property discovery: both policies degrade from H0 (`0.4974`) to
approximately `0.528` after mutation.
- Pairwise optimization: always-accept improves slightly (`0.4093` to
`0.4060`); guarded is non-monotone and ends at `0.4060`.

## Interpretation

The experiment confirms that active checkpoints and acceptance policies are
being evaluated correctly, but it does not support the claim that the current
self-evolution mechanism improves action-worthiness decisions. Acceptance
selection is active, yet the aggregate H0-to-H3 score worsens. The clearest
failure is extreme-property discovery, where mutation adds no general benefit
and increases error.

This is a useful negative ablation: the performance gap cannot currently be
attributed to recursive evolution as a positive factor. The next method revision
should improve mutation proposals and acceptance objectives before claiming
self-improvement.

## Limitations

- Labels are historical benchmark-derived proxies, not online MatBot outcomes or
expert next-action-worthiness annotations.
- The proposer is deterministic and trajectory-conditioned, not an LLM proposer.
- The primary score combines ranking, calibration, and fixed-budget utility; task
utilities remain task-specific and are not pooled as a physical unit.
