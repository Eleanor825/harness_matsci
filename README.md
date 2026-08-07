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

Run the three paper experiments:

```bash
python -m harness_matsci experiment-suite \
  --out runs/rhi_experiments_v4/summary.json \
  --markdown-out runs/rhi_experiments_v4/README.md \
  --data-dir /path/to/material_discovery_tasks \
  --seeds 1,7,13,21,42 \
  --n-per-task 300
```

The suite reports: (1) single-task main results, (2) leave-one-task-out transfer,
and (3) balanced joint multi-task training with per-task slices. Each experiment is paired with a short
design note explaining what it proves and the main reviewer challenge.

Enable the optional one-shot LLM direct-as-judge baseline:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4.1-mini
python -m harness_matsci experiment-suite \
  --data-dir /path/to/material_discovery_tasks \
  --direct-judge-model "$OPENAI_MODEL" \
  --direct-judge-cache runs/direct_judge_cache/scores.json \
  --out runs/rhi_experiments_v5_direct_judge/summary.json \
  --markdown-out runs/rhi_experiments_v5_direct_judge/README.md
```

`llm_direct_judge` is a one-shot baseline: the LLM receives only the visible
context, candidate action, and pre-execution evidence, returns `p_success`, and
does not train, receive trajectory feedback, or mutate a harness. Its threshold
is calibrated on the validation/feedback partition, while the test partition is
used only for final evaluation. Calls are cached by model and prompt version;
the repository does not include API keys or generated score caches.

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

The formal five-seed snapshot in `runs/rhi_experiments_v4/RESULTS.md` is the
current reference. It is a rigorous diagnostic result, not a positive claim: RHI
does not consistently beat strong learned baselines, transfer is unstable, and
the data are offline benchmark proxies rather than online MatBot trajectories.

In the tables, `non_rhi_seed` means a single learned logistic gate using the
initial RHI feature contract but with no recursive harness mutation or
trajectory-conditioned acceptance. `static_full` is a single learned gate with
the full available feature set. `verbal_confidence` is not an LLM judge; it is a
text/feature heuristic. The direct LLM comparison is implemented separately and
must be run with an explicitly configured API model.

## Design note

This repository reuses the harness idea from recursive self-improvement systems, but the task layer is scientific: the model learns whether a materials action is worth executing, not whether code changes are good.
