# First Paper-Bootstrap Experiment

This run evaluates the first historical-paper bootstrap version of the runtime uncertainty harness.

## Setup

- Data source: `/Users/huan/matsci_uncertainty/data/raw/nature_electrolyte/training_actions_500.jsonl`
- Domain: battery electrolyte papers
- Records: 500 action-level paper-derived `ActionRecord`s
- Groups: 42 paper groups, split by `group_id` to reduce paper leakage
- Labels: weak post-hoc `outcome_success` labels reconstructed from historical paper evidence
- Seeds: 0, 1, 2, 3, 7
- Versions compared:
  - `v1_compact_uncertainty_contract`: compact evidence/confidence/cost features
  - `v2_extended_paper_harness`: extended evidence, OOD, source, perturbation, and context features

## Main Result

Across all seeds, the selected version is always `v2_extended_paper_harness`.

| Calibration alpha | Coverage | Selective accuracy | Selective risk | ECE | Brier | Discovery hit rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.565 ± 0.061 | 0.838 ± 0.083 | 0.162 ± 0.083 | 0.076 ± 0.011 | 0.008 ± 0.003 | 1.000 ± 0.000 |
| 0.05 | 0.537 ± 0.044 | 0.877 ± 0.062 | 0.123 ± 0.062 | 0.076 ± 0.011 | 0.008 ± 0.003 | 1.000 ± 0.000 |
| 0.00 | 0.448 ± 0.027 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.076 ± 0.011 | 0.008 ± 0.003 | 1.000 ± 0.000 |

## Baselines

At the same test splits, verbal confidence and evidence heuristic baselines accept far fewer useful actions and have much higher selective risk.

| Method | Coverage | Selective accuracy | Selective risk | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| Verbal confidence | 0.046 | 0.441 | 0.559 | 0.083 | 0.253 |
| Evidence heuristic | 0.013 | 0.000 | 1.000 | 0.384 | 0.390 |

## Interpretation

- The extended harness is necessary: the compact v1 feature contract is not enough for robust paper-derived action gating.
- The learned harness strongly outperforms simple confidence and heuristic baselines on calibration and selective decision quality.
- With empirical `alpha=0.10`, validation risk is controlled, but held-out paper-group test risk averages 0.162, showing domain/group shift.
- The conservative zero-risk validation threshold gives 0.448 coverage with zero false accepted test actions across these five seeds.
- This is a paper-derived weak-label bootstrap experiment, not a replacement for online MatBot runtime trajectories.

## Artifacts

- Multi-seed report: `runs/paper_bootstrap_v1_multiseed/summary.json`
- Single-seed workdirs can be regenerated with `python -m harness_matsci paper-bootstrap --workdir runs/paper_bootstrap_v1`.
