# Sci-VoI-RHI: Scientific Value-of-Information Harness Self-Improvement

## One-Sentence Claim

Scientific agents should not only estimate whether the next action will succeed;
they should estimate whether the action has positive conservative scientific
value after accounting for uncertainty, verification, and cost. We propose an
executable Value-of-Information RHI layer that recursively improves this action
contract for materials discovery agents.

## Motivation

Materials agents fail before the final answer: they over-trust weak evidence,
promote candidates outside the known regime, ignore disagreement between tools,
or spend costly experiments on actions whose uncertainty could first be reduced.
Standard confidence calibration answers “am I likely correct?”; the scientific
runtime question is sharper: “is this action worth doing now, or should the
agent verify, retrieve, ask, simulate, or stop?”

The original Recursive Harness Self-Improvement paper shows that a user-level
harness can be improved by pairwise feedback over its own revision history. That
is useful but insufficient for materials discovery: output preference is not the
same as action value, and textual roles/contracts do not matter unless they are
executed by the runtime.

## Method

Sci-VoI-RHI turns the harness into an executable decision contract. Each harness
checkpoint specifies active reliability features, utility features, uncertainty
penalties, verification costs, routing floors, roles, and hops. At runtime it
computes:

```text
p_success       = calibrated action reliability
u_hat           = predicted continuous scientific utility
sigma_epistemic = ensemble disagreement over utility heads
V_execute       = u_hat - cost - failure_penalty - epistemic_penalty
V_verify        = reducible_uncertainty * sigma_epistemic - verification_cost
decision        = execute / verify / stop
```

Recursive improvement follows the RHI loop, but the mutation is science-specific:

```text
H_i -> action feedback -> executable contract mutation
    -> source-regime acceptance -> accept or rollback -> H_{i+1}
```

Acceptance uses source-regime lower-confidence improvement plus catastrophe
guards. The held-out scientific regime is never used for fitting, mutation,
thresholding, acceptance, or stopping.

## Difference from Prior Work

| Line of work | What it estimates | Why it is insufficient here | Sci-VoI-RHI difference |
| --- | --- | --- | --- |
| LLM confidence / calibration | answer correctness or verbal confidence | no action cost, no materials utility | estimates action-level reliability and scientific value |
| Semantic entropy / self-consistency | generation instability | targets text hallucination | uses utility-head epistemic uncertainty for action routing |
| Selective prediction | accept/abstain under risk | abstention is not a scientific next step | routes to execute, verify, or stop |
| Conformal/risk control | calibrated risk threshold | static threshold, no recursive harness | recursively mutates executable contracts |
| Original RHI | pairwise output preference | no scientific utility or cost objective | pairwise history optimizes value-of-information contracts |
| Prompt/workflow search | prompt or graph performance | expensive population search, often black-box | few-iteration source-regime RHI with transparent heads |

## Experimental Design

The non-LLM benchmark uses three published materials-discovery task families:

1. preferential optimization from pairwise “A better than B” decisions;
2. DiSCoVeR-style unique-material screening;
3. extreme-property molecular discovery.

The formal split is leave-one-scientific-regime-out: 21 complete regimes are
held out one at a time, repeated over five seeds. Source records are split into
train, feedback, and acceptance partitions. Direct LLM-as-judge is excluded by
design for this experiment package.

Primary metric is oracle-normalized continuous net scientific utility at a fixed
10% action budget. Guardrails are selective risk, hit rate, outcome-conditioned
utility, simple regret, verification rate, and worst-slice behavior.

The related-work baseline sweep covers random and cost-only policies, confidence
and evidence judges, self-consistency and semantic-entropy proxies, calibrated
selective-prediction gates, ensemble lower-confidence bounds, and acquisition
functions such as utility UCB/LCB and uncertainty sampling. True direct LLM and
multi-call agentic judge baselines are protocol-ready but require API calls.

## Expected Reviewer Questions

- **Is this just feature engineering?** No. The new `voi.py` runtime consumes the
  evolved contract fields directly; mutations alter scores, decisions, and
  routes, not only JSON text.
- **Is test used for self-improvement?** No. Held-out regimes are touched only
  after source-regime selection completes.
- **Does RHI actually help?** The decisive comparison is Sci-VoI-RHI versus both
  original reliability-only RHI and strong static VoI/static full baselines.
- **Is this MatBot online evidence?** Not yet. The current benchmark is a
  historical offline proxy; MatBot online trajectory logging is the next step.

## Current Results

The full non-LLM suite in `runs/scivoi_rhi_v1/README.md` evaluates 15,717
records, 21 leave-one-regime-out folds, and five seeds. The direct RHI-style
Sci-VoI variant (`scivoi_policy_always_accept`) obtains:

- net utility `0.6443 ± 0.2052`;
- risk-adjusted utility `0.6043`;
- selective risk `0.1600`;
- paired utility gain over original RHI `+0.0808` with bootstrap CI
  `[0.0549, 0.1059]`;
- paired utility gain over static full reliability `+0.0809` with bootstrap CI
  `[0.0584, 0.1025]`.

Compared with strong static utility/VoI heads, direct Sci-VoI-RHI is roughly
tied on continuous utility but much safer: static VoI risk is `0.3845`, while
direct Sci-VoI-RHI risk is `0.1600`. Verbal confidence has high raw utility but
unacceptable risk (`0.7889`), so it is not a safe scientific action policy.

The non-LLM related-work sweep in
`runs/related_work_baselines_v1/SCIVOI_COMPARISON.md` strengthens this pattern.
`verbal_confidence`, `tool_agreement`, `cost_only`, and `uncertainty_sampling`
can obtain high raw utility by selecting aggressive actions, but their risks are
around `0.79`. Sci-VoI-RHI remains best by risk-adjusted utility (`0.6043`) and
has the lowest risk among competitive methods (`0.1600`).

The conservative guarded variant improves over original RHI but is not the best
variant. This is useful scientifically: in this benchmark, once the executable
VoI contract is introduced, the original RHI-style direct update is stronger
than an overly pessimistic held-out acceptance gate.

## Paper Positioning

The strongest ICLR framing is not “we beat every static model.” It is:

> Scientific action uncertainty is a value-of-information problem. Making the
> harness executable converts RHI from prompt revision into a safe scientific
> decision layer that improves over reliability-only self-evolution and exposes
> where static utility heads remain competitive.

The current evidence supports a method paper centered on executable scientific
action value. The strongest claim should be: Sci-VoI-RHI fixes the failure mode
of reliability-only RHI, improves over static reliability and original RHI, and
matches static utility while substantially reducing unsafe action selection.
