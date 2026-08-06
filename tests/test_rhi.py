from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harness_matsci.benchmarks import make_records
from harness_matsci.harnesses import validate_harness
from harness_matsci.rhi import DeterministicTrajectoryProposer, train_rhi


ROOT = Path(__file__).resolve().parents[1]


class RhiTests(unittest.TestCase):
    def test_harness_validation_preserves_safe_hops(self) -> None:
        candidate, status = validate_harness(
            {
                "name": "candidate",
                "roles": [
                    {"id": "evidence_auditor", "kind": "adviser", "instruction": "audit", "contract": ["evidence_support"]},
                    {"id": "uncertainty_gate", "kind": "builder", "instruction": "gate", "contract": ["probability"]},
                    {"id": "fallback_router", "kind": "reviewer", "instruction": "route", "contract": ["route"]},
                ],
                "hops": [
                    {"from": "orchestrator", "to": "evidence_auditor", "purpose": "audit"},
                    {"from": "evidence_auditor", "to": "uncertainty_gate", "purpose": "handoff"},
                    {"from": "evil", "to": "uncertainty_gate", "purpose": "reject"},
                ],
            },
            iteration=1,
        )
        self.assertEqual(status, "model")
        self.assertEqual(len(candidate["hops"]), 2)
        self.assertNotIn("evil", json.dumps(candidate))

    def test_rhi_records_feedback_and_compares_consecutive_versions(self) -> None:
        records = make_records("preferential_bo", n=120, seed=7)
        report = train_rhi(records, iterations=2, seed=7, epochs=100)

        self.assertEqual(report["method"]["name"], "RHI-MatSci")
        self.assertEqual(len(report["proposals"]), 2)
        self.assertEqual(len(report["comparisons"]), 2)
        self.assertIn("failure_counts", report["proposals"][0]["feedback"])
        self.assertIn("required_features", report["proposals"][0]["candidate"])
        self.assertIn(report["comparisons"][0]["winner"], {"candidate", "previous"})

    def test_rhi_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            dataset = tmpdir / "data.jsonl"
            output = tmpdir / "rhi.json"
            command = [
                sys.executable,
                "-m",
                "harness_matsci",
                "make-benchmark",
                "--benchmark",
                "preferential_bo",
                "--n",
                "90",
                "--seed",
                "3",
                "--out",
                str(dataset),
            ]
            subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
            command = [
                sys.executable,
                "-m",
                "harness_matsci",
                "rhi",
                "--data",
                str(dataset),
                "--out",
                str(output),
                "--iterations",
                "1",
                "--epochs",
                "80",
            ]
            subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
            report = json.loads(output.read_text())
            self.assertEqual(report["method"]["name"], "RHI-MatSci")
            self.assertIn("test", report)


if __name__ == "__main__":
    unittest.main()
