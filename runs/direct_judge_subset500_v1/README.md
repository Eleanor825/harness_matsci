# Direct Judge Subset500

- Model: `gpt-5.5` via `https://www.hi-code.cc`.
- Records: `500` total; validation `249`, test `251`.
- Tasks: `discover_unique`=167, `extreme_properties`=167, `preferential_bo`=166.
- Positive rate: validation `0.2410`, test `0.2430`.
- LLM score mean/std: `0.2866 ± 0.2695`.
- Abstract-action fallback records: `2`.

## Aggregate Results

| Method | Score ↓ | Selective risk ↓ | Coverage ↑ | Risk@10% ↓ | Coverage@10% ↑ | Hit rate ↑ | Utility efficiency ↑ | ECE ↓ | Brier ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llm_direct_judge_gpt55` | 0.3602 | 0.7436 | 0.7769 | 0.3462 | 0.1036 | 0.6800 | 0.0809 | 0.1548 | 0.1673 |
| `verbal_confidence` | 0.4757 | 0.7654 | 0.9681 | 0.4231 | 0.1036 | 0.6000 | 0.0696 | 0.3017 | 0.2669 |
| `evidence_heuristic` | 0.8131 | 0.8150 | 0.7968 | 0.8846 | 0.1036 | 0.1200 | 0.4094 | 0.5879 | 0.5511 |

## Task Slices

| Task | Method | Score ↓ | Selective risk ↓ | Coverage ↑ | Risk@10% ↓ | Coverage@10% ↑ | Utility efficiency ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `discover_unique` | `llm_direct_judge_gpt55` | 0.3935 | 0.8000 | 0.1190 | 0.7778 | 0.1071 | 0.9094 |
| `discover_unique` | `verbal_confidence` | 0.6702 | 0.9524 | 1.0000 | 1.0000 | 0.1071 | 0.7700 |
| `discover_unique` | `evidence_heuristic` | 0.9888 | 0.8462 | 0.1548 | 0.8889 | 0.1071 | 0.8624 |
| `extreme_properties` | `llm_direct_judge_gpt55` | 0.5508 | 0.8696 | 0.2738 | 0.8889 | 0.1071 | 0.7032 |
| `extreme_properties` | `verbal_confidence` | 0.5342 | 0.8214 | 1.0000 | 0.7778 | 0.1071 | 0.8042 |
| `extreme_properties` | `evidence_heuristic` | 0.8007 | 0.8750 | 0.0952 | 0.8889 | 0.1071 | 0.6983 |
| `preferential_bo` | `llm_direct_judge_gpt55` | 0.3548 | 0.0667 | 0.1807 | 0.0000 | 0.1084 | 0.0669 |
| `preferential_bo` | `verbal_confidence` | 0.4072 | 0.3529 | 0.2048 | 0.2222 | 0.1084 | 0.2334 |
| `preferential_bo` | `evidence_heuristic` | 0.4066 | 0.2353 | 0.2048 | 0.0000 | 0.1084 | 0.0856 |

## Limitations
- 500-record subset only, not full 15,717-record benchmark.
- Deterministic balanced task subset, not complete held-out-regime evaluation.
- Two repeated-524 RL-CC molecule records were scored with an abstracted action fallback that omits the exact molecule string.
- LLM cache is not committed.
