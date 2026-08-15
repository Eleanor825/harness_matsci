from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from harness_matsci.hybrid_judge import HybridJudgeConfig, run_hybrid_judge_experiment, save_hybrid_judge_experiment
from harness_matsci.io import write_jsonl
from harness_matsci.schema import ActionRecord


class HybridJudgeExperimentTests(unittest.TestCase):
    def test_cached_llm_scores_can_be_combined_with_local_harness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [_record(index, "validation" if index < 18 else "test") for index in range(36)]
            records_path = root / "records.jsonl"
            cache_path = root / "judge_cache.json"
            out_dir = root / "run"
            write_jsonl(records, records_path)
            cache_path.write_text(
                json.dumps({"scores": {record.record_id: 0.82 if record.label else 0.18 for record in records}}),
                encoding="utf-8",
            )

            config = HybridJudgeConfig(
                records_path=str(records_path),
                judge_cache_path=str(cache_path),
                out_dir=str(out_dir),
                epochs=3,
                local_train_fraction=0.5,
            )
            report = run_hybrid_judge_experiment(config)
            save_hybrid_judge_experiment(report, out_dir)

            self.assertIn("hybrid_static_blend", report["method_summary"])
            self.assertTrue(math.isfinite(report["method_summary"]["hybrid_static_blend"]["score"]))
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "README.md").exists())


def _record(index: int, split: str) -> ActionRecord:
    benchmark = ("preferential_bo", "discover_unique", "extreme_properties")[index % 3]
    label = 0 if index % 4 == 0 else 1
    signal = 0.75 if label else 0.25
    features = {
        "verbal_confidence": signal,
        "evidence_support": signal,
        "evidence_conflict": 1.0 - signal,
        "source_reliability": signal,
        "tool_agreement": signal,
        "model_disagreement": 1.0 - signal,
        "perturbation_stability": signal,
        "ood_score": 1.0 - signal,
        "cost": 0.2 + 0.01 * (index % 5),
        "reversibility": 0.8,
        "action_complexity": 0.3,
        "evidence_count": 0.5,
    }
    return ActionRecord(
        record_id=f"record-{index:03d}",
        benchmark=benchmark,
        split=split,
        visible_context=f"Synthetic context {index}",
        candidate_action=f"Synthetic action {index}",
        action_type="recommend",
        evidence=[f"support={signal:.2f}"],
        features=features,
        label=label,
        utility=0.8 if label else 0.05,
        metadata={"group_id": benchmark},
    )


if __name__ == "__main__":
    unittest.main()
