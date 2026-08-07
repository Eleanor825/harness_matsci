# RHI-MatSci Experiment Suite

## Protocol
- Data regime: `historical_offline_proxy`.
- Tasks: preferential_bo, discover_unique, extreme_properties.
- Seeds: 1, 7, 13, 21, 42.
- Target risk: 0.10; minimum validation coverage: 0.10; fixed scientific budget: 0.10.
- Primary score is threshold-independent and combines AURC, calibration, and oracle-normalized fixed-budget discovery efficiency; threshold risk and coverage are reported together.

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
## Leakage and Split Audit
- Benchmark-derived post-outcome signals are excluded from model features and retained only as provenance metadata.
- Record IDs are disjoint across train/feedback/acceptance/test; historical test regimes are group-disjoint from all source partitions.
- Experiment 2 calibrates only on source tasks. `target_supervised_reference` is explicitly non-zero-shot and is reported only as a target-data reference, not as a strict oracle upper bound.

## Experiment 1: Single-Task Action Worthiness
- Runs: 75; seeds: 5; tasks: 3.

| Method | Primary score ↓ | AURC ↓ | Risk @ 10% ↓ | Budget hit rate ↑ | Oracle-normalized utility ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `evidence_heuristic` | 0.696 ± 0.190 | 0.664 ± 0.167 | 0.615 ± 0.176 | 0.386 ± 0.176 | 0.529 ± 0.273 |
| `non_rhi_seed` | 0.439 ± 0.045 | 0.665 ± 0.166 | 0.620 ± 0.169 | 0.380 ± 0.169 | 0.540 ± 0.259 |
| `rhi` | 0.447 ± 0.065 | 0.677 ± 0.167 | 0.657 ± 0.202 | 0.344 ± 0.202 | 0.579 ± 0.222 |
| `static_full` | 0.449 ± 0.042 | 0.688 ± 0.175 | 0.666 ± 0.197 | 0.334 ± 0.196 | 0.602 ± 0.205 |
| `verbal_confidence` | 0.436 ± 0.151 | 0.565 ± 0.241 | 0.442 ± 0.370 | 0.556 ± 0.372 | 0.673 ± 0.319 |

Threshold-selected operating point (secondary; interpret risk with coverage):

| Method | Selective risk ↓ | Coverage ↑ | ECE ↓ | Brier ↓ |
| --- | ---: | ---: | ---: | ---: |
| `evidence_heuristic` | 0.564 ± 0.321 | 0.072 ± 0.047 | 0.519 ± 0.236 | 0.507 ± 0.180 |
| `non_rhi_seed` | 0.579 ± 0.312 | 0.067 ± 0.046 | 0.070 ± 0.067 | 0.191 ± 0.074 |
| `rhi` | 0.461 ± 0.327 | 0.046 ± 0.046 | 0.076 ± 0.076 | 0.194 ± 0.077 |
| `static_full` | 0.741 ± 0.227 | 0.079 ± 0.054 | 0.077 ± 0.064 | 0.193 ± 0.074 |
| `verbal_confidence` | 0.681 ± 0.189 | 0.692 ± 0.436 | 0.227 ± 0.176 | 0.265 ± 0.026 |

Paired comparisons against RHI (negative score difference favors RHI):
- `evidence_heuristic`: Δscore=-0.2498 [-0.3575, -0.1491], RHI wins=1.000, exact sign p=0.0001, n=15.
- `non_rhi_seed`: Δscore=0.0075 [-0.0037, 0.0222], RHI wins=0.400, exact sign p=1.0000, n=15.
- `static_full`: Δscore=-0.0026 [-0.0180, 0.0159], RHI wins=0.733, exact sign p=0.1185, n=15.
- `verbal_confidence`: Δscore=0.0106 [-0.0903, 0.1196], RHI wins=0.533, exact sign p=1.0000, n=15.

### Unique-material discovery
- `evidence_heuristic`: score=0.933 ± 0.000, AURC=0.844 ± 0.000, risk@10%=0.724 ± 0.000, hit=0.278 ± 0.000, unique_material_recall=0.273 ± 0.000.
- `non_rhi_seed`: score=0.410 ± 0.000, AURC=0.844 ± 0.000, risk@10%=0.724 ± 0.000, hit=0.278 ± 0.000, unique_material_recall=0.273 ± 0.000.
- `rhi`: score=0.406 ± 0.004, AURC=0.844 ± 0.000, risk@10%=0.713 ± 0.009, hit=0.289 ± 0.009, unique_material_recall=0.284 ± 0.009.
- `static_full`: score=0.441 ± 0.001, AURC=0.871 ± 0.002, risk@10%=0.864 ± 0.002, hit=0.137 ± 0.002, unique_material_recall=0.135 ± 0.002.
- `verbal_confidence`: score=0.632 ± 0.000, AURC=0.898 ± 0.000, risk@10%=0.906 ± 0.000, hit=0.089 ± 0.000, unique_material_recall=0.087 ± 0.000.
### Extreme-property discovery
- `evidence_heuristic`: score=0.686 ± 0.023, AURC=0.704 ± 0.023, risk@10%=0.750 ± 0.055, hit=0.250 ± 0.055, extreme_hit_recall=0.088 ± 0.030.
- `non_rhi_seed`: score=0.497 ± 0.030, AURC=0.704 ± 0.023, risk@10%=0.750 ± 0.055, hit=0.250 ± 0.055, extreme_hit_recall=0.088 ± 0.030.
- `rhi`: score=0.528 ± 0.051, AURC=0.735 ± 0.032, risk@10%=0.850 ± 0.134, hit=0.150 ± 0.134, extreme_hit_recall=0.058 ± 0.055.
- `static_full`: score=0.500 ± 0.024, AURC=0.738 ± 0.022, risk@10%=0.720 ± 0.103, hit=0.280 ± 0.103, extreme_hit_recall=0.101 ± 0.048.
- `verbal_confidence`: score=0.265 ± 0.019, AURC=0.350 ± 0.052, risk@10%=0.000 ± 0.000, hit=1.000 ± 0.000, extreme_hit_recall=0.345 ± 0.052.
### Pairwise optimization
- `evidence_heuristic`: score=0.470 ± 0.015, AURC=0.445 ± 0.021, risk@10%=0.371 ± 0.031, hit=0.629 ± 0.031, pairwise_latent_regret=0.001 ± 0.001.
- `non_rhi_seed`: score=0.409 ± 0.009, AURC=0.446 ± 0.019, risk@10%=0.387 ± 0.023, hit=0.613 ± 0.023, pairwise_latent_regret=0.001 ± 0.001.
- `rhi`: score=0.406 ± 0.008, AURC=0.451 ± 0.010, risk@10%=0.407 ± 0.040, hit=0.593 ± 0.040, pairwise_latent_regret=0.001 ± 0.001.
- `static_full`: score=0.406 ± 0.008, AURC=0.454 ± 0.006, risk@10%=0.416 ± 0.029, hit=0.584 ± 0.029, pairwise_latent_regret=0.001 ± 0.001.
- `verbal_confidence`: score=0.411 ± 0.006, AURC=0.446 ± 0.019, risk@10%=0.420 ± 0.004, hit=0.580 ± 0.004, pairwise_latent_regret=0.002 ± 0.001.
## Experiment 2: Leave-One-Task-Out Transfer
- Runs: 90; seeds: 5; tasks: 3.

| Method | Primary score ↓ | AURC ↓ | Risk @ 10% ↓ | Budget hit rate ↑ | Oracle-normalized utility ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `evidence_heuristic` | 0.696 ± 0.190 | 0.664 ± 0.167 | 0.615 ± 0.176 | 0.386 ± 0.176 | 0.529 ± 0.273 |
| `non_rhi_seed` | 0.507 ± 0.066 | 0.685 ± 0.140 | 0.667 ± 0.106 | 0.333 ± 0.106 | 0.504 ± 0.307 |
| `rhi` | 0.608 ± 0.093 | 0.683 ± 0.140 | 0.667 ± 0.113 | 0.332 ± 0.113 | 0.530 ± 0.317 |
| `static_full` | 0.672 ± 0.121 | 0.696 ± 0.143 | 0.711 ± 0.160 | 0.289 ± 0.160 | 0.509 ± 0.293 |
| `target_supervised_reference` | 0.449 ± 0.042 | 0.688 ± 0.175 | 0.666 ± 0.197 | 0.334 ± 0.196 | 0.602 ± 0.205 |
| `verbal_confidence` | 0.436 ± 0.151 | 0.565 ± 0.241 | 0.442 ± 0.370 | 0.556 ± 0.372 | 0.673 ± 0.319 |

Threshold-selected operating point (secondary; interpret risk with coverage):

| Method | Selective risk ↓ | Coverage ↑ | ECE ↓ | Brier ↓ |
| --- | ---: | ---: | ---: | ---: |
| `evidence_heuristic` | 0.646 ± 0.288 | 0.514 ± 0.456 | 0.519 ± 0.236 | 0.507 ± 0.180 |
| `non_rhi_seed` | 0.467 ± 0.368 | 0.530 ± 0.410 | 0.211 ± 0.111 | 0.240 ± 0.116 |
| `rhi` | 0.470 ± 0.368 | 0.404 ± 0.446 | 0.399 ± 0.181 | 0.374 ± 0.103 |
| `static_full` | 0.300 ± 0.424 | 0.333 ± 0.471 | 0.415 ± 0.149 | 0.376 ± 0.110 |
| `target_supervised_reference` | 0.741 ± 0.227 | 0.079 ± 0.054 | 0.077 ± 0.064 | 0.193 ± 0.074 |
| `verbal_confidence` | 0.551 ± 0.328 | 0.734 ± 0.377 | 0.227 ± 0.176 | 0.265 ± 0.026 |

Paired comparisons against RHI (negative score difference favors RHI):
- `evidence_heuristic`: Δscore=-0.0887 [-0.1724, 0.0004], RHI wins=0.667, exact sign p=0.3018, n=15.
- `non_rhi_seed`: Δscore=0.1004 [0.0377, 0.1676], RHI wins=0.200, exact sign p=0.1460, n=15.
- `static_full`: Δscore=-0.0642 [-0.1228, -0.0043], RHI wins=0.667, exact sign p=0.3018, n=15.
- `target_supervised_reference`: Δscore=0.1585 [0.0928, 0.2173], RHI wins=0.200, exact sign p=0.0352, n=15.
- `verbal_confidence`: Δscore=0.1717 [0.1351, 0.2078], RHI wins=0.000, exact sign p=0.0001, n=15.

### Unique-material discovery
- `evidence_heuristic`: score=0.933 ± 0.000, AURC=0.844 ± 0.000, risk@10%=0.724 ± 0.000, hit=0.278 ± 0.000, unique_material_recall=0.273 ± 0.000.
- `non_rhi_seed`: score=0.431 ± 0.002, AURC=0.844 ± 0.000, risk@10%=0.724 ± 0.000, hit=0.278 ± 0.000, unique_material_recall=0.273 ± 0.000.
- `rhi`: score=0.712 ± 0.033, AURC=0.847 ± 0.001, risk@10%=0.765 ± 0.013, hit=0.234 ± 0.012, unique_material_recall=0.230 ± 0.012.
- `static_full`: score=0.640 ± 0.034, AURC=0.846 ± 0.001, risk@10%=0.758 ± 0.016, hit=0.241 ± 0.014, unique_material_recall=0.237 ± 0.014.
- `target_supervised_reference`: score=0.441 ± 0.001, AURC=0.871 ± 0.002, risk@10%=0.864 ± 0.002, hit=0.137 ± 0.002, unique_material_recall=0.135 ± 0.002.
- `verbal_confidence`: score=0.632 ± 0.000, AURC=0.898 ± 0.000, risk@10%=0.906 ± 0.000, hit=0.089 ± 0.000, unique_material_recall=0.087 ± 0.000.
### Extreme-property discovery
- `evidence_heuristic`: score=0.686 ± 0.023, AURC=0.704 ± 0.023, risk@10%=0.750 ± 0.055, hit=0.250 ± 0.055, extreme_hit_recall=0.088 ± 0.030.
- `non_rhi_seed`: score=0.505 ± 0.027, AURC=0.704 ± 0.023, risk@10%=0.750 ± 0.055, hit=0.250 ± 0.055, extreme_hit_recall=0.088 ± 0.030.
- `rhi`: score=0.494 ± 0.020, AURC=0.696 ± 0.023, risk@10%=0.720 ± 0.051, hit=0.280 ± 0.051, extreme_hit_recall=0.097 ± 0.025.
- `static_full`: score=0.550 ± 0.051, AURC=0.738 ± 0.032, risk@10%=0.850 ± 0.134, hit=0.150 ± 0.134, extreme_hit_recall=0.058 ± 0.055.
- `target_supervised_reference`: score=0.500 ± 0.024, AURC=0.738 ± 0.022, risk@10%=0.720 ± 0.103, hit=0.280 ± 0.103, extreme_hit_recall=0.101 ± 0.048.
- `verbal_confidence`: score=0.265 ± 0.019, AURC=0.350 ± 0.052, risk@10%=0.000 ± 0.000, hit=1.000 ± 0.000, extreme_hit_recall=0.345 ± 0.052.
### Pairwise optimization
- `evidence_heuristic`: score=0.470 ± 0.015, AURC=0.445 ± 0.021, risk@10%=0.371 ± 0.031, hit=0.629 ± 0.031, pairwise_latent_regret=0.001 ± 0.001.
- `non_rhi_seed`: score=0.586 ± 0.016, AURC=0.505 ± 0.005, risk@10%=0.529 ± 0.039, hit=0.471 ± 0.039, pairwise_latent_regret=0.001 ± 0.001.
- `rhi`: score=0.618 ± 0.018, AURC=0.507 ± 0.005, risk@10%=0.518 ± 0.031, hit=0.482 ± 0.031, pairwise_latent_regret=0.001 ± 0.001.
- `static_full`: score=0.825 ± 0.025, AURC=0.505 ± 0.004, risk@10%=0.524 ± 0.046, hit=0.476 ± 0.046, pairwise_latent_regret=0.000 ± 0.001.
- `target_supervised_reference`: score=0.406 ± 0.008, AURC=0.454 ± 0.006, risk@10%=0.416 ± 0.029, hit=0.584 ± 0.029, pairwise_latent_regret=0.001 ± 0.001.
- `verbal_confidence`: score=0.411 ± 0.006, AURC=0.446 ± 0.019, risk@10%=0.420 ± 0.004, hit=0.580 ± 0.004, pairwise_latent_regret=0.002 ± 0.001.
## Experiment 3: Joint Training and Stability
- Runs: 75; seeds: 5; tasks: 3.

| Method | Primary score ↓ | AURC ↓ | Risk @ 10% ↓ | Budget hit rate ↑ | Oracle-normalized utility ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `evidence_heuristic` | 0.696 ± 0.190 | 0.664 ± 0.167 | 0.615 ± 0.176 | 0.386 ± 0.176 | 0.529 ± 0.273 |
| `non_rhi_seed` | 0.455 ± 0.039 | 0.676 ± 0.152 | 0.647 ± 0.133 | 0.353 ± 0.133 | 0.503 ± 0.308 |
| `rhi` | 0.452 ± 0.040 | 0.683 ± 0.162 | 0.672 ± 0.158 | 0.329 ± 0.158 | 0.564 ± 0.277 |
| `static_full` | 0.449 ± 0.041 | 0.684 ± 0.172 | 0.665 ± 0.184 | 0.336 ± 0.184 | 0.589 ± 0.214 |
| `verbal_confidence` | 0.436 ± 0.151 | 0.565 ± 0.241 | 0.442 ± 0.370 | 0.556 ± 0.372 | 0.673 ± 0.319 |

Threshold-selected operating point (secondary; interpret risk with coverage):

| Method | Selective risk ↓ | Coverage ↑ | ECE ↓ | Brier ↓ |
| --- | ---: | ---: | ---: | ---: |
| `evidence_heuristic` | 0.609 ± 0.337 | 0.451 ± 0.453 | 0.519 ± 0.236 | 0.507 ± 0.180 |
| `non_rhi_seed` | 0.659 ± 0.333 | 0.666 ± 0.466 | 0.088 ± 0.062 | 0.194 ± 0.073 |
| `rhi` | 0.459 ± 0.367 | 0.653 ± 0.462 | 0.081 ± 0.062 | 0.193 ± 0.073 |
| `static_full` | 0.696 ± 0.275 | 0.465 ± 0.448 | 0.085 ± 0.056 | 0.192 ± 0.071 |
| `verbal_confidence` | 0.693 ± 0.175 | 0.966 ± 0.061 | 0.227 ± 0.176 | 0.265 ± 0.026 |

Paired comparisons against RHI (negative score difference favors RHI):
- `evidence_heuristic`: Δscore=-0.2441 [-0.3468, -0.1476], RHI wins=1.000, exact sign p=0.0001, n=15.
- `non_rhi_seed`: Δscore=-0.0024 [-0.0105, 0.0060], RHI wins=0.400, exact sign p=1.0000, n=15.
- `static_full`: Δscore=0.0030 [-0.0075, 0.0136], RHI wins=0.467, exact sign p=1.0000, n=15.
- `verbal_confidence`: Δscore=0.0163 [-0.0746, 0.1116], RHI wins=0.400, exact sign p=0.6072, n=15.

### Unique-material discovery
- `evidence_heuristic`: score=0.933 ± 0.000, AURC=0.844 ± 0.000, risk@10%=0.724 ± 0.000, hit=0.278 ± 0.000, unique_material_recall=0.273 ± 0.000.
- `non_rhi_seed`: score=0.421 ± 0.000, AURC=0.844 ± 0.000, risk@10%=0.724 ± 0.000, hit=0.278 ± 0.000, unique_material_recall=0.273 ± 0.000.
- `rhi`: score=0.428 ± 0.007, AURC=0.854 ± 0.010, risk@10%=0.778 ± 0.035, hit=0.224 ± 0.036, unique_material_recall=0.220 ± 0.035.
- `static_full`: score=0.446 ± 0.002, AURC=0.868 ± 0.003, risk@10%=0.844 ± 0.008, hit=0.157 ± 0.008, unique_material_recall=0.155 ± 0.008.
- `verbal_confidence`: score=0.632 ± 0.000, AURC=0.898 ± 0.000, risk@10%=0.906 ± 0.000, hit=0.089 ± 0.000, unique_material_recall=0.087 ± 0.000.
### Extreme-property discovery
- `evidence_heuristic`: score=0.686 ± 0.023, AURC=0.704 ± 0.023, risk@10%=0.750 ± 0.055, hit=0.250 ± 0.055, extreme_hit_recall=0.088 ± 0.030.
- `non_rhi_seed`: score=0.501 ± 0.028, AURC=0.704 ± 0.023, risk@10%=0.750 ± 0.055, hit=0.250 ± 0.055, extreme_hit_recall=0.088 ± 0.030.
- `rhi`: score=0.501 ± 0.030, AURC=0.725 ± 0.043, risk@10%=0.780 ± 0.051, hit=0.220 ± 0.051, extreme_hit_recall=0.077 ± 0.026.
- `static_full`: score=0.494 ± 0.033, AURC=0.729 ± 0.022, risk@10%=0.730 ± 0.068, hit=0.270 ± 0.068, extreme_hit_recall=0.097 ± 0.038.
- `verbal_confidence`: score=0.265 ± 0.019, AURC=0.350 ± 0.052, risk@10%=0.000 ± 0.000, hit=1.000 ± 0.000, extreme_hit_recall=0.345 ± 0.052.
### Pairwise optimization
- `evidence_heuristic`: score=0.470 ± 0.015, AURC=0.445 ± 0.021, risk@10%=0.371 ± 0.031, hit=0.629 ± 0.031, pairwise_latent_regret=0.001 ± 0.001.
- `non_rhi_seed`: score=0.441 ± 0.017, AURC=0.478 ± 0.026, risk@10%=0.469 ± 0.043, hit=0.531 ± 0.043, pairwise_latent_regret=0.000 ± 0.000.
- `rhi`: score=0.428 ± 0.019, AURC=0.471 ± 0.021, risk@10%=0.458 ± 0.052, hit=0.542 ± 0.052, pairwise_latent_regret=0.000 ± 0.000.
- `static_full`: score=0.407 ± 0.008, AURC=0.455 ± 0.007, risk@10%=0.420 ± 0.028, hit=0.580 ± 0.028, pairwise_latent_regret=0.000 ± 0.000.
- `verbal_confidence`: score=0.411 ± 0.006, AURC=0.446 ± 0.019, risk@10%=0.420 ± 0.004, hit=0.580 ± 0.004, pairwise_latent_regret=0.002 ± 0.001.
## Limitations
- Historical outcomes are benchmark-derived offline action-worthiness proxies, not expert annotations or online MatBot trajectory outcomes.
- The deterministic proposer is a reproducible trajectory-conditioned RHI implementation, not evidence that an LLM proposer improves the harness.
- Scientific utilities are reported per task and as macro summaries; pooled utility is not interpreted as a common physical unit.
- Source train, feedback, and acceptance records are disjoint but can share source regimes; final test regimes remain group-disjoint.
