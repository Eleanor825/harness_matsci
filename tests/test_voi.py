from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class VoIRhiTests(unittest.TestCase):
    def _write_unique_records(self, directory: Path) -> None:
        path = directory / "unique_materials_actions.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for group in ("cubic", "hexagonal"):
                for index in range(24):
                    success = index >= 18
                    score = 0.80 + 0.005 * index if success else 0.15 + 0.01 * index
                    payload = {
                        "record_id": f"{group}-{index}",
                        "group_id": f"discover::toy::{group}",
                        "domain": "toy_materials",
                        "task": "unique_materials_screening",
                        "visible_context": f"Screen toy {group} material {index}.",
                        "candidate_action": f"Screen candidate {group}-{index}.",
                        "action_type": "choose_candidate",
                        "outcome_success": success,
                        "metric_value": score,
                        "evidence": [f"candidate={index}", f"crystal_system={group}"],
                        "context_features": {"space_group": float(index + 1)},
                        "uncertainty_signals": {
                            "source_risk": 0.1 if success else 0.6,
                            "extraction_confidence": 0.9,
                        },
                        "tool_outputs": {"utility": score},
                        "cost_level": "medium",
                        "reversibility": "medium",
                    }
                    handle.write(json.dumps(payload) + "\n")

    def test_voi_runtime_produces_executable_decisions(self) -> None:
        from harness_matsci.historical import load_historical_task_records
        from harness_matsci.voi import VOI_SEED_HARNESS, evaluate_voi, fit_voi_model, propose_voi_harness

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_unique_records(directory)
            records = load_historical_task_records(directory, "discover_unique")
            train, feedback, test = records[:24], records[24:36], records[36:]
            harness, rationale = propose_voi_harness(
                VOI_SEED_HARNESS,
                {"failure_counts": {"confidently_wrong": 3}},
                iteration=1,
                available_features=sorted({key for record in records for key in record.features}),
            )
            model = fit_voi_model(train, feedback, harness, epochs=10)
            report = evaluate_voi(test, model)
            self.assertIn("utility_estimator", {role["id"] for role in harness["roles"]})
            self.assertTrue(rationale)
            self.assertIn("predictions", report)
            self.assertTrue({item["decision"] for item in report["predictions"]} <= {"execute", "verify", "stop"})
            self.assertIn("outcome_conditioned_oracle_normalized_net_utility", report)

    def test_voi_experiment_cli_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            data_dir = directory / "data"
            data_dir.mkdir()
            self._write_unique_records(data_dir)
            json_path = directory / "voi.json"
            markdown_path = directory / "voi.md"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            command = [
                sys.executable,
                "-m",
                "harness_matsci",
                "voi-experiment-suite",
                "--data-dir",
                str(data_dir),
                "--tasks",
                "discover_unique",
                "--methods",
                "h0_reliability,static_voi,scivoi_rhi",
                "--components",
                "",
                "--acceptance-policies",
                "",
                "--seeds",
                "1",
                "--epochs",
                "10",
                "--out",
                str(json_path),
                "--markdown-out",
                str(markdown_path),
            ]
            subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
            report = json.loads(json_path.read_text())
            self.assertEqual(report["protocol"]["direct_llm_judge"], False)
            self.assertIn("scivoi_rhi", report["summary"]["methods"])
            self.assertIn("Sci-VoI-RHI", markdown_path.read_text())


if __name__ == "__main__":
    unittest.main()
