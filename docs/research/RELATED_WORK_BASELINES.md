# Related-Work Baselines for Sci-VoI-RHI

## Current Baseline Coverage

| Baseline family | What it tests | Current status |
| --- | --- | --- |
| Random / cheap-action heuristics | Whether fixed-budget utility gains are trivial artifacts of random ordering or low-cost actions. | Implemented: `random_policy`, `cost_only`; results in `runs/related_work_baselines_v1/README.md`. |
| Confidence / evidence judges | Whether simple confidence or rationale/evidence scores are enough. | Implemented: `verbal_confidence`, `evidence_heuristic`, `cost_aware_confidence`. |
| Self-consistency / agreement uncertainty | Whether agreement-style uncertainty can replace Sci-VoI. | Implemented offline proxies: `tool_agreement`, `self_consistency_proxy`. True repeated-LLM self-consistency still requires API calls. |
| Semantic-entropy uncertainty | Whether text/output dispersion alone identifies good actions. | Implemented offline proxy: `semantic_entropy_proxy`. True semantic entropy still requires multiple LLM generations. |
| Selective prediction / conformal risk control | Whether calibrated accept/abstain reliability solves the problem. | Implemented through risk-calibrated thresholds for all score baselines and `h0_reliability` / `static_full_reliability`. |
| Ensemble uncertainty | Whether model-disagreement lower-confidence bounds are sufficient. | Implemented: `ensemble_reliability`, `ensemble_lcb`. |
| Active-learning / BO acquisitions | Whether UCB/LCB/uncertainty-sampling style discovery policies are enough. | Implemented: `utility_ucb`, `utility_lcb`, `uncertainty_sampling`, plus `static_utility` and `static_voi`. |
| Recursive harness optimization | Whether the original RHI mechanism is enough without scientific utility. | Implemented and preserved: `original_rhi`, `rhi_experiments_v4`, `rhi_experiments_v5_evolution`. |
| Direct LLM-as-judge | Whether a frontier LLM can directly score action worthiness. | Protocol implemented in `runs/direct_judge_baseline_v1/README.md`; needs API/model run. |
| Agentic LLM judge / debate / reflection | Whether multi-call LLM judging beats an executable harness. | Not yet run; requires API calls and a fixed token budget. |

## Preserved Non-LLM Experiments

The sweep in `runs/related_work_baselines_v1/` adds all non-LLM baseline
families above under the earlier `15,717`-record, `21`-regime, five-seed
held-out protocol that still included PBO. It intentionally does not rerun
recursive Sci-VoI mutations; the preserved Sci-VoI result remains in
`runs/scivoi_rhi_v1/`. After adding real-material `matbench_pairwise`, these
baselines should be rerun on the updated `20,987`-record main materials setup.

Key combined comparison: `scivoi_policy_always_accept` reaches risk-adjusted
utility `0.6043` with risk `0.1600`. The strongest raw-utility judge-style
baselines are more aggressive but unsafe: `verbal_confidence` has net utility
`0.7577` but risk `0.7889`, and `uncertainty_sampling` has net utility `0.7412`
but risk `0.7962`. The best static utility/VoI baselines have much lower risk
than confidence heuristics but still lower risk-adjusted utility than Sci-VoI-RHI.

See `runs/related_work_baselines_v1/SCIVOI_COMPARISON.md` for the merged table.

## Remaining ICLR-Critical Baselines

1. **True direct LLM judge**: zero-shot `p_success` / action-worthiness scoring
   with the same visible context and no labels.
2. **True LLM self-consistency / semantic entropy**: K repeated judgments per
   action, summarized by agreement or semantic clustering.
3. **Agentic judge baseline**: a judge that can critique evidence, request a
   short verification rationale, and return a structured action score.
4. **Human or expert spot-check**: small action-level audit validating that proxy
   labels match scientific intuition.
5. **MatBot online trajectory validation**: emit the same `ActionRecord` schema at
   real action boundaries and evaluate future utility or expert labels.

## Claim Boundary

The preserved non-LLM baseline sweep strengthens the offline method claim on the
earlier setup, not the online MatBot claim. The updated data layer now provides a
stronger materials-grounded pairwise task from Matbench, but final paper claims
need the full baseline rerun on `matbench_pairwise,discover_unique,extreme_properties`.
The paper should not claim demonstrated improvement in real laboratory or DFT
trajectories until online records or expert labels are added.
