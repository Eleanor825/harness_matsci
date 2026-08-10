from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harness_matsci.experiments import ExperimentSuiteConfig, run_experiment_suite
from harness_matsci.historical import grouped_four_way_split, random_four_way_split
from harness_matsci.benchmarks import make_records


ROOT = Path(__file__).resolve().parents[1]


class ExperimentSuiteTests(unittest.TestCase):
    def test_synthetic_four_way_split_is_disjoint(self) -> None:
        records = make_records("preferential_bo", n=100, seed=11)
        splits = random_four_way_split(records, seed=11)
        ids_by_split = {name: {record.record_id for record in rows} for name, rows in splits.items()}
        self.assertEqual(sum(len(ids) for ids in ids_by_split.values()), len(records))
        for left_name, left_ids in ids_by_split.items():
            for right_name, right_ids in ids_by_split.items():
                if left_name != right_name:
                    self.assertTrue(left_ids.isdisjoint(right_ids))

    def test_historical_split_does_not_select_tiny_regime_as_sole_test(self) -> None:
        records = make_records("preferential_bo", n=100, seed=11)
        grouped = []
        for index, record in enumerate(records):
            payload = record.to_json()
            payload["metadata"]["group_id"] = "tiny" if index < 2 else f"group-{index // 20}"
            grouped.append(type(record).from_json(payload))
        splits = grouped_four_way_split(grouped, seed=11)
        test_groups = {record.metadata["group_id"] for record in splits["test"]}
        self.assertNotEqual(test_groups, {"tiny"})

    def test_historical_split_targets_declared_test_mass(self) -> None:
        records = make_records("preferential_bo", n=100, seed=11)
        grouped = []
        for index, record in enumerate(records):
            payload = record.to_json()
            payload["metadata"]["group_id"] = f"group-{index // 10}"
            grouped.append(type(record).from_json(payload))
        splits = grouped_four_way_split(grouped, seed=11)
        expected_test_size = len(grouped) * 0.15
        self.assertLessEqual(abs(len(splits["test"]) - expected_test_size), 5)
        ids_by_split = {name: {record.record_id for record in rows} for name, rows in splits.items()}
        for left_name, left_ids in ids_by_split.items():
            for right_name, right_ids in ids_by_split.items():
                if left_name != right_name:
                    self.assertTrue(left_ids.isdisjoint(right_ids))

    def test_suite_contains_three_protocols_and_fixed_target_split(self) -> None:
        config = ExperimentSuiteConfig(seeds=(3,), n_per_task=60, rhi_iterations=1, epochs=60)
        report = run_experiment_suite(config)
        self.assertEqual(report["summary"], {"n_evolution_runs": 0, "n_joint_runs": 15, "n_single_runs": 15, "n_transfer_runs": 18})
        single_rows = report["runs"]["single_task"]
        for row in single_rows:
            self.assertEqual(row["split"]["train"] + row["split"]["feedback"] + row["split"]["acceptance"] + row["split"]["test"], 60)
            self.assertNotEqual(row["split"]["feedback"], row["split"]["acceptance"])
        transfer_rows = report["runs"]["leave_one_out"]
        for row in transfer_rows:
            self.assertEqual(row["experiment"], "experiment_2_leave_one_task_out")
            self.assertNotIn(row["holdout_task"], row["source_tasks"])
            self.assertGreater(row["split"]["source_feedback"], 0)
            self.assertGreater(row["split"]["source_acceptance"], 0)
            self.assertGreater(row["split"]["target_test"], 0)
            self.assertLessEqual(row["split"]["target_test"], row["split"]["target_all"])
            self.assertGreater(row["split"]["target_test"], 0)
        self.assertTrue(report["data_audit"]["all_record_partitions_disjoint"])
        self.assertIn("target_supervised_reference", {row["method"] for row in transfer_rows})
        joint_rows = report["runs"]["joint"]
        self.assertEqual({row["task"] for row in joint_rows}, {"preferential_bo", "discover_unique", "extreme_properties"})
        self.assertEqual(len(joint_rows), 15)

    def test_suite_can_run_only_experiments_one_and_two(self) -> None:
        config = ExperimentSuiteConfig(experiments=(1, 2), seeds=(3,), n_per_task=60, rhi_iterations=0, epochs=20)
        report = run_experiment_suite(config)
        self.assertEqual(report["summary"], {"n_evolution_runs": 0, "n_joint_runs": 0, "n_single_runs": 15, "n_transfer_runs": 18})
        self.assertNotIn("experiment_3_joint_training_stability", report["experiments"])

    def test_evolution_ablation_reports_checkpoints_and_paired_h0(self) -> None:
        config = ExperimentSuiteConfig(experiments=(4,), seeds=(3,), n_per_task=60, rhi_iterations=2, epochs=30)
        report = run_experiment_suite(config)
        summary = report["experiments"]["experiment_4_self_evolution_ablation"]
        self.assertEqual(summary["n_runs"], 18)
        self.assertEqual(sorted(summary["by_policy"]), ["always_accept", "guarded"])
        self.assertEqual(sorted(summary["by_policy"]["guarded"]), ["h0", "h1", "h2"])
        self.assertEqual(summary["paired_to_h0"]["guarded"]["h1"]["n"], 3)

    def test_historical_converter_excludes_post_outcome_signals(self) -> None:
        from harness_matsci.historical import _convert_historical_record

        records = [
            _convert_historical_record(
                {
                    "record_id": "extreme-1",
                    "group_id": "target-1",
                    "visible_context": "Extreme candidate",
                    "candidate_action": "Advance molecule: CCO",
                    "action_type": "choose_candidate",
                    "outcome_success": True,
                    "metric_value": 0.8,
                    "uncertainty_signals": {
                        "hit_fraction": 1.0,
                        "target_hit_score": 0.9,
                        "evidence_support": 1.0,
                        "ood_score": 0.95,
                    },
                    "context_features": {"hit_fraction": 1.0},
                    "tool_outputs": {"utility": 0.8},
                },
                "extreme_properties",
            )
        ]
        self.assertNotIn("hit_fraction", records[0].features)
        self.assertNotIn("target_hit_score", records[0].features)
        self.assertNotIn("log10(K_VRH)", records[0].visible_context)
        self.assertIn("hit_fraction", records[0].metadata["excluded_oracle_features"])

    def test_unique_context_sanitization_removes_property_value(self) -> None:
        from harness_matsci.historical import _convert_historical_record

        record = _convert_historical_record(
            {
                "record_id": "unique-1",
                "group_id": "crystal-1",
                "visible_context": "Screen candidate; log10(K_VRH)=1.70757.",
                "candidate_action": "Screen CaAgGe.",
                "action_type": "choose_candidate",
                "outcome_success": True,
                "metric_value": 0.8,
                "uncertainty_signals": {},
                "context_features": {},
            },
            "discover_unique",
        )
        self.assertNotIn("1.70757", record.visible_context)

        extreme = _convert_historical_record(
            {
                "record_id": "extreme-2",
                "group_id": "target-2",
                "visible_context": "Extreme candidate",
                "candidate_action": "Advance molecule: CCO",
                "action_type": "choose_candidate",
                "outcome_success": True,
                "evidence": ["hit_fraction=1.0", "reward=12.0", "all_hit=True"],
                "uncertainty_signals": {},
                "context_features": {},
            },
            "extreme_properties",
        )
        self.assertTrue(all("hit_fraction=1.0" not in item for item in extreme.evidence))
        self.assertTrue(all("reward=12.0" not in item for item in extreme.evidence))

    def test_baseline_threshold_is_calibrated_on_validation(self) -> None:
        config = ExperimentSuiteConfig(seeds=(3,), n_per_task=60, rhi_iterations=0, epochs=60)
        report = run_experiment_suite(config)
        row = next(
            row
            for row in report["runs"]["leave_one_out"]
            if row["task"] == "extreme_properties" and row["method"] == "verbal_confidence"
        )
        self.assertEqual(row["report"]["validation"]["n_records"], row["split"]["source_feedback"])
        self.assertEqual(row["report"]["test"]["n_records"], row["split"]["target_test"])

    def test_suite_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            json_path = tmpdir / "suite.json"
            markdown_path = tmpdir / "suite.md"
            command = [
                sys.executable,
                "-m",
                "harness_matsci",
                "experiment-suite",
                "--seeds",
                "3",
                "--n-per-task",
                "60",
                "--rhi-iterations",
                "1",
                "--epochs",
                "60",
                "--out",
                str(json_path),
                "--markdown-out",
                str(markdown_path),
            ]
            subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
            payload = json.loads(json_path.read_text())
            self.assertIn("experiment_1_single_task", payload["experiments"])
            self.assertIn("Experiment 2", markdown_path.read_text())


if __name__ == "__main__":
    unittest.main()
