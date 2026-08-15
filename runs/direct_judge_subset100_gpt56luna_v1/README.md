# Direct Judge Subset100 GPT-5.6-Luna Attempt

- Records: `100` sampled from `runs/direct_judge_subset500_v1/records.jsonl`.
- Split: validation `50`, test `50`.
- Tasks: `discover_unique`=34, `extreme_properties`=33, `preferential_bo`=33.
- Target model: `gpt-5.6-luna`.
- Status: blocked by provider/gateway, so no valid `gpt-5.6-luna` scores were produced.

## Gateway Checks

| Model/base URL | Result |
| --- | --- |
| `gpt-5.6-luna` via `https://www.hi-code.cc` | HTTP `429 Too Many Requests` after retries |
| `gpt-5.6-luna` via `https://www.hi-code.cc`, effort `medium/high/xhigh` | all HTTP `429 Too Many Requests` |
| `gpt-5.6-lun` via `https://www.hi-code.cc` | HTTP `404 Not Found` |
| `gpt-5.6-luna` via `https://coding.beehears.com` | connection refused |
| `gpt-5.6-luna` via `https://coding.beehears.com/v1` | connection refused |
| `gpt-5.5` via `https://www.hi-code.cc` | smoke test OK |

## Model Availability Smoke

Single-record checks on `https://www.hi-code.cc` with the same direct-judge
prompt showed which models can currently be used for a larger subset run. These
are provider-availability checks, not benchmark results.

| Model | Status |
| --- | --- |
| `gpt-5.5` | OK |
| `gpt-5.4` | OK |
| `gpt-5.4-mini` | OK |
| `gpt-5.3-codex` | HTTP `503 Service Unavailable` |
| `gpt-5.2` | HTTP `503 Service Unavailable` |
| `gpt-5.6-luna` | HTTP `429 Too Many Requests` |

## Same-100 GPT-5.5 Cached Control

These numbers reuse the existing `gpt-5.5` cache from the 500-record run for the same 100 records.

| Method | Score ↓ | Selective risk ↓ | Coverage ↑ | Risk@10% ↓ | Hit rate ↑ | ECE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llm_direct_judge_gpt55_cached` | 0.3484 | 0.6667 | 0.7800 | 0.2000 | 0.8000 | 0.1213 |
| `verbal_confidence` | 0.5432 | 0.7234 | 0.9400 | 0.6000 | 0.4000 | 0.2381 |
| `evidence_heuristic` | 0.8310 | 0.8261 | 0.4600 | 1.0000 | 0.0000 | 0.5370 |

## Files

- Local records: `runs/direct_judge_subset100_gpt56luna_v1/records.jsonl` is kept locally and ignored by Git with other run JSONL data.
- Cached-control summary: `runs/direct_judge_subset100_gpt56luna_v1/summary_gpt55_cached_control.json`.
