from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "harness_matsci", *args]
    return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=True)


class CliTests(unittest.TestCase):
    def test_make_train_evaluate_and_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

            dataset = tmpdir / "data.jsonl"
            model_path = tmpdir / "model.json"
            report_path = tmpdir / "report.json"
            campaign_path = tmpdir / "campaign.json"

            run_cli("make-benchmark", "--benchmark", "preferential_bo", "--n", "90", "--seed", "3", "--out", str(dataset), cwd=ROOT, env=env)
            self.assertTrue(dataset.exists())

            run_cli("train", "--data", str(dataset), "--out", str(model_path), "--alpha", "0.2", "--seed", "3", cwd=ROOT, env=env)
            self.assertTrue(model_path.exists())

            run_cli("evaluate", "--data", str(dataset), "--model", str(model_path), "--out", str(report_path), cwd=ROOT, env=env)
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text())
            self.assertGreaterEqual(report["metrics"]["coverage"], 0.0)
            self.assertIn("decisions", report)

            run_cli("campaign", "--benchmark", "preferential_bo", "--seeds", "3", "--n", "60", "--out", str(campaign_path), cwd=ROOT, env=env)
            self.assertTrue(campaign_path.exists())
            campaign = json.loads(campaign_path.read_text())
            self.assertEqual(campaign["aggregate"]["n_runs"], 1)
            self.assertEqual(len(campaign["runs"]), 1)


if __name__ == "__main__":
    unittest.main()
