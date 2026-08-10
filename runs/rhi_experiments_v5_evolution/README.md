# RHI-MatSci Experiment Suite

## Protocol
- Data regime: `historical_offline_proxy`.
- Tasks: preferential_bo, discover_unique, extreme_properties.
- Seeds: 1, 7, 13, 21, 42.
- Target risk: 0.10; minimum validation coverage: 0.10; fixed scientific budget: 0.10.
- Primary score is threshold-independent and combines AURC, calibration, and oracle-normalized fixed-budget discovery efficiency; threshold risk and coverage are reported together.
- `llm_direct_judge` is included only when configured; it is a one-shot LLM judge with no training, recursive mutation, or trajectory feedback, and its threshold is calibrated on validation/feedback records.

## What Each Experiment Tests
### Experiment 1 — task-specific validity
- **Claim tested**: recursive harness revisions improve ranking and fixed-budget scientific action selection within each task family
- **Reviewer challenge**: a low selective risk may be produced by abstaining, a random split may leak regimes, or gains may come from a learned classifier rather than recursion
- **Design response**: use regime-held-out test groups, independent feedback and acceptance sets, strong non-RHI learned baselines, fixed-coverage risk, AURC, fixed-budget utility, and five paired seeds
### Experiment 2 — cross-task transfer
- **Claim tested**: an action-worthiness contract learned on two task families transfers zero-shot to a third
- **Reviewer challenge**: target calibration leakage or task-size imbalance can manufacture transfer, and failure may simply reflect an impossible target
- **Design response**: exclude all target records from source training/calibration/acceptance, balance source tasks in loss and calibration, evaluate on held-out target regimes, and report a target-supervised upper bound
### Experiment 3 — joint robustness
- **Claim tested**: one jointly trained harness remains stable across seeds and does not sacrifice a smaller task to optimize the largest dataset
- **Reviewer challenge**: pooled metrics are dominated by the 10,987-record unique-material task
- **Design response**: use inverse-frequency task weighting, macro acceptance, apply exactly the same selected joint gate to every task slice, and report per-task and macro-over-task results
### Experiment 4 — self-evolution ablation
- **Claim tested**: recursive checkpoint selection, rather than a single learned gate, improves the scientific action-worthiness decision
- **Reviewer challenge**: an accepted revision may be selected by acceptance noise, or a final score may hide regressions across intermediate harness versions
- **Design response**: evaluate active H0, H1, H2, and H3 checkpoints on one untouched test split, compare guarded acceptance with always-accept mutation, and report paired change from H0 by task and policy
## Leakage and Split Audit
- Benchmark-derived post-outcome signals are excluded from model features and retained only as provenance metadata.
- Record IDs are disjoint across train/feedback/acceptance/test; historical test regimes are group-disjoint from all source partitions.
- Experiment 2 calibrates only on source tasks. `target_supervised_reference` is explicitly non-zero-shot and is reported only as a target-data reference, not as a strict oracle upper bound.

## Experiment 4: Self-Evolution Ablation
- Runs: 120; seeds: 5; tasks: 3; policies: 2.
- `guarded` uses the held-out acceptance gate; `always_accept` accepts every schema-valid mutation without consulting test data.

| Checkpoint | Policy | Primary score ↓ | Budget hit rate ↑ | Oracle-normalized utility ↑ |
| --- | --- | ---: | ---: | ---: |
| `h0` | `always_accept` | 0.439 ± 0.045 | 0.380 ± 0.169 | 0.540 ± 0.259 |
| `h0` | `guarded` | 0.439 ± 0.045 | 0.380 ± 0.169 | 0.540 ± 0.259 |
| `h1` | `always_accept` | 0.446 ± 0.066 | 0.343 ± 0.196 | 0.592 ± 0.209 |
| `h1` | `guarded` | 0.444 ± 0.052 | 0.359 ± 0.185 | 0.554 ± 0.244 |
| `h2` | `always_accept` | 0.446 ± 0.066 | 0.343 ± 0.196 | 0.592 ± 0.209 |
| `h2` | `guarded` | 0.449 ± 0.064 | 0.340 ± 0.199 | 0.571 ± 0.229 |
| `h3` | `always_accept` | 0.446 ± 0.066 | 0.343 ± 0.196 | 0.592 ± 0.209 |
| `h3` | `guarded` | 0.447 ± 0.065 | 0.344 ± 0.202 | 0.579 ± 0.222 |

Paired comparisons against H0 (negative score difference favors the checkpoint):
- `always_accept/h1`: Δscore=0.0065 [-0.0049, 0.0215], wins=0.533, exact sign p=1.0000, n=15.
- `always_accept/h2`: Δscore=0.0065 [-0.0049, 0.0215], wins=0.533, exact sign p=1.0000, n=15.
- `always_accept/h3`: Δscore=0.0065 [-0.0049, 0.0215], wins=0.533, exact sign p=1.0000, n=15.
- `guarded/h1`: Δscore=0.0045 [-0.0002, 0.0135], wins=0.067, exact sign p=0.6250, n=15.
- `guarded/h2`: Δscore=0.0096 [-0.0007, 0.0237], wins=0.267, exact sign p=0.7539, n=15.
- `guarded/h3`: Δscore=0.0075 [-0.0037, 0.0222], wins=0.400, exact sign p=1.0000, n=15.

Task-specific checkpoint scores:
### Unique-material discovery
- `always_accept`: h0=0.410 ± 0.000, h1=0.403 ± 0.000, h2=0.403 ± 0.000, h3=0.403 ± 0.000.
- `guarded`: h0=0.410 ± 0.000, h1=0.410 ± 0.000, h2=0.407 ± 0.004, h3=0.406 ± 0.004.
### Extreme-property discovery
- `always_accept`: h0=0.497 ± 0.030, h1=0.528 ± 0.051, h2=0.528 ± 0.051, h3=0.528 ± 0.051.
- `guarded`: h0=0.497 ± 0.030, h1=0.510 ± 0.039, h2=0.528 ± 0.051, h3=0.528 ± 0.051.
### Pairwise optimization
- `always_accept`: h0=0.409 ± 0.009, h1=0.406 ± 0.008, h2=0.406 ± 0.008, h3=0.406 ± 0.008.
- `guarded`: h0=0.409 ± 0.009, h1=0.410 ± 0.008, h2=0.411 ± 0.009, h3=0.406 ± 0.008.
## Limitations
- Historical outcomes are benchmark-derived offline action-worthiness proxies, not expert annotations or online MatBot trajectory outcomes.
- The deterministic proposer is a reproducible trajectory-conditioned RHI implementation, not evidence that an LLM proposer improves the harness.
- llm_direct_judge is an optional one-shot baseline: it sees only the visible action context and does not train, mutate a harness, or receive trajectory feedback.
- Scientific utilities are reported per task and as macro summaries; pooled utility is not interpreted as a common physical unit.
- Source train, feedback, and acceptance records are disjoint but can share source regimes; final test regimes remain group-disjoint.
