# RHI-MatSci Experiment Suite

## Protocol
- Data regime: `synthetic_benchmark`.
- Tasks: discover_unique.
- Seeds: 1.
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

## Experiment 1: Single-Task Action Worthiness
- Runs: 6; seeds: 1; tasks: 1.

| Method | Primary score ↓ | AURC ↓ | Risk @ 10% ↓ | Budget hit rate ↑ | Oracle-normalized utility ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `evidence_heuristic` | 0.690 ± 0.000 | 0.915 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| `llm_direct_judge` | 0.294 ± 0.000 | 0.741 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| `non_rhi_seed` | 0.256 ± 0.000 | 0.741 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| `rhi` | 0.262 ± 0.000 | 0.741 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| `static_full` | 0.251 ± 0.000 | 0.741 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| `verbal_confidence` | 0.699 ± 0.000 | 0.825 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

Threshold-selected operating point (secondary; interpret risk with coverage):

| Method | Selective risk ↓ | Coverage ↑ | ECE ↓ | Brier ↓ |
| --- | ---: | ---: | ---: | ---: |
| `evidence_heuristic` | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.403 ± 0.000 | 0.244 ± 0.000 |
| `llm_direct_judge` | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.214 ± 0.000 | 0.103 ± 0.000 |
| `non_rhi_seed` | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.167 ± 0.000 | 0.065 ± 0.000 |
| `rhi` | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.175 ± 0.000 | 0.074 ± 0.000 |
| `static_full` | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.141 ± 0.000 | 0.066 ± 0.000 |
| `verbal_confidence` | 1.000 ± 0.000 | 0.083 ± 0.000 | 0.472 ± 0.000 | 0.286 ± 0.000 |

Paired comparisons against RHI (negative score difference favors RHI):
- `evidence_heuristic`: Δscore=-0.4283 [-0.4283, -0.4283], RHI wins=1.000, exact sign p=1.0000, n=1.
- `llm_direct_judge`: Δscore=-0.0324 [-0.0324, -0.0324], RHI wins=1.000, exact sign p=1.0000, n=1.
- `non_rhi_seed`: Δscore=0.0052 [0.0052, 0.0052], RHI wins=0.000, exact sign p=1.0000, n=1.
- `static_full`: Δscore=0.0112 [0.0112, 0.0112], RHI wins=0.000, exact sign p=1.0000, n=1.
- `verbal_confidence`: Δscore=-0.4373 [-0.4373, -0.4373], RHI wins=1.000, exact sign p=1.0000, n=1.

### Unique-material discovery
- `evidence_heuristic`: score=0.690 ± 0.000, AURC=0.915 ± 0.000, risk@10%=1.000 ± 0.000, hit=0.000 ± 0.000, unique_material_recall=0.000 ± 0.000.
- `llm_direct_judge`: score=0.294 ± 0.000, AURC=0.741 ± 0.000, risk@10%=0.500 ± 0.000, hit=1.000 ± 0.000, unique_material_recall=1.000 ± 0.000.
- `non_rhi_seed`: score=0.256 ± 0.000, AURC=0.741 ± 0.000, risk@10%=0.500 ± 0.000, hit=1.000 ± 0.000, unique_material_recall=1.000 ± 0.000.
- `rhi`: score=0.262 ± 0.000, AURC=0.741 ± 0.000, risk@10%=0.500 ± 0.000, hit=1.000 ± 0.000, unique_material_recall=1.000 ± 0.000.
- `static_full`: score=0.251 ± 0.000, AURC=0.741 ± 0.000, risk@10%=0.500 ± 0.000, hit=1.000 ± 0.000, unique_material_recall=1.000 ± 0.000.
- `verbal_confidence`: score=0.699 ± 0.000, AURC=0.825 ± 0.000, risk@10%=0.500 ± 0.000, hit=0.000 ± 0.000, unique_material_recall=0.000 ± 0.000.
## Limitations
- Historical outcomes are benchmark-derived offline action-worthiness proxies, not expert annotations or online MatBot trajectory outcomes.
- The deterministic proposer is a reproducible trajectory-conditioned RHI implementation, not evidence that an LLM proposer improves the harness.
- llm_direct_judge is an optional one-shot baseline: it sees only the visible action context and does not train, mutate a harness, or receive trajectory feedback.
- Scientific utilities are reported per task and as macro summaries; pooled utility is not interpreted as a common physical unit.
- Source train, feedback, and acceptance records are disjoint but can share source regimes; final test regimes remain group-disjoint.
