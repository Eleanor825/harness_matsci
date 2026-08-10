# Experiment Runs Index

This directory intentionally keeps both positive and negative experiment
artifacts so method revisions can be compared without overwriting prior
results. All reported numbers are offline historical/proxy results, not online
MatBot trajectory outcomes.

## Current Positive Run

| Run | Purpose | Key result |
| --- | --- | --- |
| `scivoi_rhi_v1/` | Sci-VoI-RHI held-out-regime evaluation over pairwise optimization, unique-material discovery, and extreme-property discovery. | Best direct RHI-style VoI variant reaches net utility `0.6443 ± 0.2052`, risk-adjusted utility `0.6043`, and selective risk `0.1600` over 105 held-out-regime folds. |
| `mechanism_ablation_v2/` | Clean mechanism ablation for utility, uncertainty, routing, cost, and recursive updates without component bundle loops. | `scivoi_policy_always_accept` beats frozen `scivoi_policy_never_accept` by `+0.0454` utility and `-0.3522` selective risk; utility-only beats reliability-only by `+0.0447`. |

## Preserved Comparison Runs

| Run | Purpose | Why it is kept |
| --- | --- | --- |
| `rhi_experiments_v4/` | Five-seed reliability-only RHI evaluation on 15,717 historical benchmark-derived records. | Diagnostic negative result: reliability-only RHI is comparable to learned baselines but does not establish a positive self-improvement claim. |
| `rhi_experiments_v5_evolution/` | Self-evolution checkpoint ablation for `H0`–`H3` under guarded and always-accept policies. | Shows recursive mutation is active, but aggregate H0-to-H3 primary score worsens; useful contrast for why Sci-VoI is needed. |
| `paper_bootstrap_v1_multiseed/` | First paper-derived weak-label bootstrap on 500 electrolyte action records across five seeds. | Demonstrates feasibility of historical-paper action labels and records domain/group-shift limitations. |
| `label_audit_v1/` | Full historical label/utility consistency and leakage audit. | Confirms the proxy labels and utilities are internally reproducible and that oracle fields are withheld from visible text. |
| `direct_judge_baseline_v1/` | One-shot direct LLM-as-judge baseline protocol. | Preserves the baseline definition without publishing mock scores; generated score caches are intentionally not committed. |

## Reading Order

For paper writing or result comparison, read:

1. `scivoi_rhi_v1/README.md` for the current Sci-VoI result.
2. `rhi_experiments_v5_evolution/RESULTS.md` for the prior self-evolution ablation.
3. `rhi_experiments_v4/RESULTS.md` for the broader reliability-only diagnostic suite.
4. `paper_bootstrap_v1_multiseed/README.md` for historical-paper weak-label evidence.
5. `direct_judge_baseline_v1/README.md` for the direct LLM judge baseline protocol.

## Claim Boundaries

- Positive claim supported here: Sci-VoI-style harness optimization improves
  offline proxy action-value decisions over reliability-only RHI and static full
  reliability baselines.
- Claim not yet supported here: online MatBot scientific discovery improves in
  real lab/DFT trajectories.
- Important limitation: labels and utilities are reconstructed from published or
  benchmark-derived outcomes, not expert annotations of live agent decisions.
