# harness_matsci

Runtime uncertainty harness for scientific agents in materials discovery.

## What it does

- Turns each scientific action into an `ActionRecord` with visible context, evidence, feature signals, and a binary label for whether the action was worth executing.
- Trains a transparent `LogisticGate` to estimate `P(action worth doing)`.
- Routes low-confidence actions to `retrieve_more`, `simulate`, `ask_expert`, or `abstain` instead of forcing a bad proceed.
- Ships three offline benchmarks aligned with our paper targets: pairwise optimization, unique-material discovery, and extreme-property discovery.
- Provides an RHI-MatSci loop that evolves the scientific agent's prompt-level contracts and orchestrator hops from trajectory feedback, then accepts or rejects each revision on held-out actions.

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

Run trajectory-feedback Recursive Harness Self-Improvement:

```bash
python -m harness_matsci rhi \
  --data data/preferential_bo.jsonl \
  --iterations 3 \
  --out runs/rhi_preferential_bo/report.json
```

The default proposer is deterministic and reproducible. An OpenAI-compatible
proposer can be supplied from Python through `JSONLLMHarnessProposer`; its
candidate is still schema-validated and must pass the held-out acceptance gate.

Run the first historical-paper bootstrap experiment:

```bash
python -m harness_matsci paper-bootstrap \
  --data /Users/huan/matsci_uncertainty/data/raw/nature_electrolyte/training_actions_500.jsonl \
  --workdir runs/paper_bootstrap_v1
```

## RHI-MatSci loop

The implementation follows the reusable RHI mechanism from
`alphaXiv/recursive-harness-self-improvement` without copying its repository
task code:

```text
H_i -> action trajectories -> failure feedback -> proposer mutation
    -> schema validation -> held-out H_i/H_{i+1} comparison -> accept/rollback
```

The evolved state includes `required_features`, scientific agent roles and
contracts, routing gates, and orchestrator hops. The base material-science
agent is not fine-tuned. The current offline path replays labeled
`ActionRecord` trajectories; real MatBot integration should emit the same
records at action boundaries and add future utility or expert review labels.

## Current results

The tracked snapshot in `reports/route_rhi_v3_results.md` summarizes the
previous route-aware offline run on 15,717 reconstructed records. It is useful
as a diagnostic baseline, not as a final positive claim: the route targets are
deterministic action-worthiness proxies, coverage collapses on some tasks, and
the data are not online MatBot trajectories.

## Design note

This repository reuses the harness idea from recursive self-improvement systems, but the task layer is scientific: the model learns whether a materials action is worth executing, not whether code changes are good.
