# Direct LLM Judge Baseline

## Status

Protocol and implementation are ready. A formal score table is intentionally
not included until an OpenAI-compatible model and API key are explicitly
configured. No mock scores are reported as scientific results.

## Definition

`llm_direct_judge` is a one-shot baseline for the same action-worthiness task as
RHI. For each record, the model sees only the decision-time visible context,
candidate action, and sanitized pre-execution evidence. It returns a JSON
`p_success` in `[0, 1]` and a route/rationale. It receives no labels, feedback,
acceptance shards, or previous harness versions.

The score threshold is calibrated on the same validation/feedback partition
used by the corresponding weak baseline. The test partition remains held out
until final evaluation. Experiment 2 is zero-shot: target-task records are not
used to calibrate the direct judge.

## Run

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5.4
export OPENAI_BASE_URL=https://coding.beehears.com
export OPENAI_REASONING_EFFORT=xhigh
PYTHONPATH=src python3 -m harness_matsci experiment-suite \
  --data-dir /path/to/material_discovery_tasks \
  --seeds 1,7,13,21,42 \
  --direct-judge-model "$OPENAI_MODEL" \
  --direct-judge-base-url "$OPENAI_BASE_URL" \
  --direct-judge-reasoning-effort "$OPENAI_REASONING_EFFORT" \
  --direct-judge-cache runs/direct_judge_cache/scores.json \
  --out runs/direct_judge_baseline_v1/summary.json \
  --markdown-out runs/direct_judge_baseline_v1/README.md
```

The cache is ignored by Git. The generated report records the model name,
protocol, calibration split, and method-level metrics without storing the API
key.
