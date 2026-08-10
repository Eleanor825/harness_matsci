# Evolving Findings

## Problem

Output confidence is not the same as scientific action worth. A materials agent
must trade off expected scientific gain, failure risk, action cost, and whether
additional evidence can reduce uncertainty. RHI supplies a low-cost mechanism
for recursively editing a harness, but its pairwise output preference does not
define this scientific decision objective.

## Candidate contribution

**Scientific Value-of-Information RHI (Sci-VoI RHI)** evolves executable
contracts rather than inert text. A harness checkpoint specifies which
reliability, utility, epistemic-uncertainty, cost, and verification components
are active. It produces a conservative action value and one of three semantic
decisions: proceed, verify, or stop. Recursive acceptance uses pairwise history
over held-out source regimes and requires Pareto-safe improvement in discovery
utility and risk.

## Distinction from RHI

RHI asks whether one generated output is preferred to the previous output and
uses that preference history to revise information flow. Sci-VoI RHI asks
whether the *next scientific action* has positive conservative net value, and
uses decomposed outcome feedback to revise the decision contract. The proposed
method adds a scientific utility head, epistemic uncertainty, cost-aware
value-of-information routing, executable mutations, and regime-robust
acceptance; none is part of the original RHI objective.

## Evidence status

The original reliability-only RHI has a rigorous negative result. The full
non-LLM Sci-VoI-RHI experiment now provides a positive method result under
historical proxy data: the direct RHI-style executable VoI variant obtains
`0.6443` net utility and `0.1600` selective risk over 105 held-out-regime folds,
improving utility over original RHI by `+0.0808` and over static full reliability
by `+0.0809`. It is not a replacement for online MatBot trajectories or expert
action labels.
