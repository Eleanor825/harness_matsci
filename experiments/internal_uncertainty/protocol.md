# Internal Logit Uncertainty → VoI → RHI Pilot

## Claim under test

Candidate-token logit margins from a causal LM provide an action-level internal
confidence signal that improves scientific action selection after calibration,
and RHI can improve the resulting VoI harness on held-out actions.

## Protocol

- Tasks: `preferential_bo`, `discover_unique`, `extreme_properties`.
- Pilot data: 600 deterministic synthetic benchmark records, 200 per task.
- Split: 60% train, 10% feedback, 10% acceptance, 20% test.
- Decision format: `A=EXECUTE`, `B=DEFER_OR_STOP`; score is the final-position
  candidate-token logit margin `logit(A)-logit(B)`.
- Temperature is fitted on feedback only; test labels/utilities are never used
  for model fitting or threshold selection.
- Compared systems: local VoI, logit-augmented VoI, within-task permutation
  control, local VoI-RHI, and logit-augmented VoI-RHI.
- Primary diagnostics: internal-logit correctness/ECE/Brier/Risk@10%,
  oracle-normalized net utility, hit rate, and execute selective risk.

## Execution command

```bash
PYTHONPATH=src /Users/huanzhang/anaconda3/bin/python \
  experiments/internal_uncertainty/run.py \
  --model distilgpt2 --n-per-task 200 \
  --out runs/internal_uncertainty_v1/summary.json
```

## Model boundary

`distilgpt2` is used only because it exposes local logits in the current
environment. It is not a materials-science instruction model and its result is
not evidence about GPT-5.5 or a frontier scientific judge.

## Current verdict

The pilot successfully extracts real candidate-token logits and runs them
through VoI and RHI. However, the local model's action-choice accuracy is below
chance (`0.4333`), the within-task permutation control matches the
logit-augmented VoI score, and local/logit RHI converge to the same accepted
checkpoint and test score. Therefore the pilot does **not** establish that
internal logit uncertainty contributes causally to RHI or scientific utility.

The next valid experiment requires a logit-accessible instruction model or
provider, repeated seeds, group-held-out materials regimes, and an ablation
where the logit signal changes the harness decision while all other features
and RHI acceptance rules remain fixed.
