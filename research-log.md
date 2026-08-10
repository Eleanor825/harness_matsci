# Research Log

## 2026-08-10 — Intended-vs-implemented audit

- **Intended claim:** RHI evolves scientific roles, contracts, routes, and hops
  to decide whether an action is worth doing.
- **Implemented reality:** `required_features` changes the fitted logistic gate,
  while most role, gate, and hop edits are declarative JSON and do not alter
  offline predictions or routing.
- **Consequence:** the current experiment establishes recursive feature-set
  search, not executable harness evolution.
- **Observed result:** five-seed H0-to-H3 performance is flat or worse; the
  extreme-property task degrades most.
- **Data diagnosis:** five of ten extreme-property regimes contain no positive
  binary labels although their continuous scientific utilities vary. A pure
  `p_success` objective therefore cannot express whether actions within those
  regimes are more or less valuable.
- **Decision:** replace reliability-only action selection with an executable
  value-of-information contract and test it under a frozen leave-one-regime-out
  protocol. Existing v4/v5 test results remain diagnostic and will not be used
  for new hyperparameter selection.

