# harness_matsci

Runtime uncertainty harness for scientific agents in materials discovery.

## What it does

- Turns each scientific action into an `ActionRecord` with visible context, evidence, feature signals, and a binary label for whether the action was worth executing.
- Trains a transparent `LogisticGate` to estimate `P(action worth doing)`.
- Routes low-confidence actions to `retrieve_more`, `simulate`, `ask_expert`, or `abstain` instead of forcing a bad proceed.
- Ships three offline benchmarks aligned with our paper targets: pairwise optimization, unique-material discovery, and extreme-property discovery.

## Data format

Use JSONL with one `ActionRecord` per line. Key fields:

- `visible_context`: what the agent saw
- `candidate_action`: the proposed step
- `features`: uncertainty and cost signals such as `evidence_support`, `ood_score`, `tool_agreement`, `cost`, `reversibility`
- `label`: `1` if the step was worth doing, else `0`
- `utility`: optional downstream gain for discovery metrics

## Commands

Generate a benchmark dataset:

```bash
python -m harness_matsci make-benchmark --benchmark preferential_bo --out data/preferential_bo.jsonl
```

Train a gate:

```bash
python -m harness_matsci train --data data/preferential_bo.jsonl --out artifacts/gate.json
```

Evaluate a saved gate:

```bash
python -m harness_matsci evaluate --data data/preferential_bo.jsonl --model artifacts/gate.json --out reports/eval.json
```

Run a benchmark campaign:

```bash
python -m harness_matsci campaign --out reports/campaign.json
```

Run the first historical-paper bootstrap experiment:

```bash
python -m harness_matsci paper-bootstrap \
  --data /Users/huan/matsci_uncertainty/data/raw/nature_electrolyte/training_actions_500.jsonl \
  --workdir runs/paper_bootstrap_v1
```

## Design note

This repository reuses the harness idea from recursive self-improvement systems, but the task layer is scientific: the model learns whether a materials action is worth executing, not whether code changes are good.
