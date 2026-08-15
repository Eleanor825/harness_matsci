# Claim–Evidence Matrix

| Claim | Required evidence | Current artifact |
| --- | --- | --- |
| Original reliability-only RHI is implemented | H0→H3 checkpoints, guarded vs always-accept ablation | `runs/rhi_experiments_v5_evolution/RESULTS.md` |
| Original RHI is not yet a positive result | paired H0/H3 test comparison shows no improvement | `runs/rhi_experiments_v5_evolution/RESULTS.md` |
| Sci-VoI-RHI is not the same as original RHI | utility head, epistemic ensemble, executable routing, source-regime acceptance | `src/harness_matsci/voi.py` |
| Contracts/hops affect runtime behavior | evolved fields change `decision_mode`, costs, uncertainty floor, routes, features | `src/harness_matsci/voi.py` |
| Main materials data provenance is reproducible | Matbench pairwise, DiSCoVeR/Matbench, and RL-CC raw files, label/utility equations, leakage filters, and example records are listed | `docs/research/DATA_PROVENANCE_AND_ACTION_CONSTRUCTION.md` |
| Real-material pairwise preference is implemented | A/B actions are generated from Matbench `log10(K_VRH)` rows and audited against hidden property values | `src/harness_matsci/matbench_pairwise.py` and `runs/label_audit_matbench_pairwise_v1/README.md` |
| Direct LLM judge baseline is implemented separately | one-shot `p_success` judge uses visible action context only and no harness feedback | `src/harness_matsci/direct_judge.py` and `runs/direct_judge_subset500_v1/README.md` |
| Hybrid LLM + VoI judging is implemented as a subset experiment | cached direct LLM scores are fused with local VoI scores and compared on the same 500-record split | `src/harness_matsci/hybrid_judge.py` and `runs/hybrid_llm_judge_subset_v1/README.md` |
| Evaluation is regime-held-out | complete `group_id` outer folds; train/feedback/acceptance/test IDs disjoint | `src/harness_matsci/voi_experiments.py` and run audit |
| Method improves action worthiness on the preserved legacy setup | paired held-out-regime utility gain with risk guardrails | `runs/scivoi_rhi_v1/README.md` |
| Method improves action worthiness on updated main materials setup | full rerun over `matbench_pairwise,discover_unique,extreme_properties` | pending |
| Method is ablated | component and acceptance-policy rows | `runs/scivoi_rhi_v1/summary.json` |
| Related-work baselines are covered offline | confidence, evidence, self-consistency proxy, semantic-entropy proxy, ensembles, acquisition policies | `runs/related_work_baselines_v1/SCIVOI_COMPARISON.md` and `docs/research/RELATED_WORK_BASELINES.md` |
| Leadership/reporting tables are current | compact method, ablation, and direct-judge tables | `docs/research/EXPERIMENT_RESULTS_TABLES.md` and `docs/research/LEADERSHIP_BRIEF.md` |

## Current Scientific Status

The repository now has an updated main-materials data layer: `8,000` Matbench
real-material pairwise preference actions, `10,987` DiSCoVeR/Matbench screening
actions, and `2,000` RL-CC extreme-property actions. The combined main-materials
audit covers `20,987` records across `45` regimes and passes label consistency,
utility consistency, and visible-leakage checks.

The repository also contains a rigorous negative result for reliability-only RHI
and a positive non-LLM result for executable Sci-VoI-RHI on the preserved legacy
setup that included PBO. The direct RHI-style always-accept variant obtains
`0.6443` net utility and `0.1600` selective risk, beating original RHI by
`+0.0808` utility and static full reliability by `+0.0809` over 105 paired
held-out-regime folds. This result should now be treated as legacy until the
full 20,987-record main-materials rerun is completed.

The new related-work baseline sweep adds non-LLM analogues of agentic judge and
uncertainty methods. Several baselines obtain high raw utility by being
aggressive, but they also have high selective risk: `verbal_confidence` has net
utility `0.7577` with risk `0.7889`, and `uncertainty_sampling` has net utility
`0.7412` with risk `0.7962`. In the merged table, `scivoi_policy_always_accept`
has the best risk-adjusted utility (`0.6043`) and much lower risk (`0.1600`).

The real direct LLM judge baseline is now partially populated. On the balanced
500-record subset, `gpt-5.5` direct judge obtains score `0.3602`, Risk@10%
`0.3462`, hit rate `0.6800`, and ECE `0.1548`. This is a strong one-shot
baseline and should be reported separately from both the preserved legacy run
and the updated main-materials benchmark. The attempted `gpt-5.6-luna` run is
not counted because the provider returned HTTP 429.

The hybrid LLM + VoI subset run uses the same cached `gpt-5.5` scores as a
semantic sensor. `hybrid_static_blend` improves score to `0.3315` and ECE to
`0.0974`, but it leaves Risk@10% and hit rate unchanged relative to direct
`gpt-5.5` and still uses an LLM call for every tested action. This supports a
calibration/fusion claim, not yet a cheaper or stronger top-k discovery claim.
