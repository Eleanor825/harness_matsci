# Frozen Protocol: Scientific Value-of-Information RHI

## Hypothesis

An executable harness that recursively adds task utility, epistemic uncertainty,
and value-of-information routing will select more valuable actions in unseen
scientific regimes than reliability-only RHI or a static full-feature gate.

## Data and outer split

- Use all 15,717 sanitized historical proxy records from the three published
  task sources.
- Treat `group_id` as the scientific regime.
- Hold out each of the 21 regimes once. No held-out-regime record may influence
  fitting, calibration, mutation, acceptance, or stopping.
- Split source-regime records into disjoint train, feedback, and acceptance
  partitions by a deterministic record hash. Repeat stochastic fitting over
  five fixed seeds only if fitting depends on a seed.

## Methods

1. Verbal confidence and evidence heuristic.
2. Reliability-only H0 gate.
3. Static full-feature reliability gate.
4. Original reliability-only RHI.
5. Static utility model.
6. Static value-of-information harness.
7. Sci-VoI RHI with recursive robust acceptance.

## Primary endpoint

Macro average over outer regimes of oracle-normalized continuous net scientific
utility among the top 10% actions. This follows the source papers: preferential
optimization uses latent preference margin, DiSCoVeR-style screening uses the
performance--uniqueness discovery score, and extreme-property discovery uses
continuous target reward. Binary success labels are not ignored: selective risk,
hit rate, and outcome-conditioned utility are guardrails, so a method cannot be
claimed successful by selecting high-property but unreliable actions.

## Guardrails

- hit efficiency and simple regret at the same budget;
- selective risk and AURC for action reliability;
- worst-regime utility efficiency;
- proceed/verify/stop coverage and confidently-wrong proceed rate.

## Ablations

- reliability only;
- utility without epistemic uncertainty;
- utility plus uncertainty without cost/verification routing;
- full executable VoI contract;
- mean-source acceptance versus regime-robust Pareto acceptance;
- guarded versus always-accept recursion;
- feature, route, and full-contract mutation;
- 1%, 5%, 10%, and 20% action budgets;
- 25%, 50%, and 100% source-data scale;
- 0%, 10%, and 20% source-label corruption.

## Statistical analysis

- Paired differences use the same held-out regimes for every method.
- Report mean, bootstrap 95% confidence interval, win rate, and exact sign test.
- Regime, not record, is the unit of inference.
- No positive claim is made from record-level significance or from the earlier
  v4/v5 test partitions.

## Failure criterion

Reject the primary hypothesis if Sci-VoI RHI does not improve paired
held-out-regime utility over both original RHI and static full learning, or if
the utility gain is bought by a material selective-risk regression.
