# Claim–Evidence Matrix

| Claim | Required evidence | Current artifact |
| --- | --- | --- |
| Original reliability-only RHI is implemented | H0→H3 checkpoints, guarded vs always-accept ablation | `runs/rhi_experiments_v5_evolution/RESULTS.md` |
| Original RHI is not yet a positive result | paired H0/H3 test comparison shows no improvement | `runs/rhi_experiments_v5_evolution/RESULTS.md` |
| Sci-VoI-RHI is not the same as original RHI | utility head, epistemic ensemble, executable routing, source-regime acceptance | `src/harness_matsci/voi.py` |
| Contracts/hops affect runtime behavior | evolved fields change `decision_mode`, costs, uncertainty floor, routes, features | `src/harness_matsci/voi.py` |
| No LLM direct judge is included | protocol flag false; CLI has no direct judge argument | `src/harness_matsci/voi_experiments.py` |
| Evaluation is regime-held-out | complete `group_id` outer folds; train/feedback/acceptance/test IDs disjoint | `src/harness_matsci/voi_experiments.py` and run audit |
| Method improves action worthiness | paired held-out-regime utility gain with risk guardrails | `runs/scivoi_rhi_v1/README.md` |
| Method is ablated | component and acceptance-policy rows | `runs/scivoi_rhi_v1/summary.json` |

## Current Scientific Status

The repository contains a rigorous negative result for reliability-only RHI and
a positive non-LLM result for executable Sci-VoI-RHI. The direct RHI-style
always-accept variant obtains `0.6443` net utility and `0.1600` selective risk,
beating original RHI by `+0.0808` utility and static full reliability by
`+0.0809` over 105 paired held-out-regime folds. It is utility-comparable to
static utility/VoI but much safer.
