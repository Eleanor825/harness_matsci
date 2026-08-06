from __future__ import annotations

import unittest

from harness_matsci.paper_bootstrap import paper_json_to_action_record, split_records_by_group


class PaperBootstrapTests(unittest.TestCase):
    def test_paper_record_mapping(self) -> None:
        record = paper_json_to_action_record(
            {
                "record_id": "paper-action::1",
                "group_id": "paper::g1",
                "domain": "battery_electrolyte",
                "task": "paper_to_runtime_decision",
                "visible_context": "context",
                "candidate_action": "check evidence",
                "action_type": "ask_more",
                "expected_outcome": "more evidence",
                "metric_key": "action_reliability",
                "metric_direction": "maximize",
                "metric_value": 0.7,
                "outcome_success": True,
                "evidence": ["e1"],
                "context_features": {"segment_length": 12, "title_length": 4, "abstract_length": 8, "has_pdf": True},
                "uncertainty_signals": {
                    "evidence_support": 0.6,
                    "evidence_conflict": 0.1,
                    "verbal_confidence": 0.4,
                    "source_risk": 0.2,
                    "extraction_confidence": 0.8,
                    "ood_score": 0.3,
                    "consensus_spread": 0.25,
                    "perturbation_stability": 0.7,
                    "tool_agreement": 0.8,
                },
                "verbal_confidence": 0.5,
                "cost_level": "medium",
                "reversibility": "high",
                "risk_level": "low",
            }
        )

        self.assertEqual(record.action_type, "retrieve_more")
        self.assertEqual(record.label, 1)
        self.assertEqual(record.metadata["group_id"], "paper::g1")
        self.assertIn("source_reliability", record.features)
        self.assertGreater(record.features["source_reliability"], 0.7)

    def test_grouped_split_keeps_groups_together(self) -> None:
        records = []
        for idx, group_id in enumerate(["g1", "g1", "g2", "g2"]):
            records.append(
                paper_json_to_action_record(
                    {
                        "record_id": f"paper-action::{idx}",
                        "group_id": group_id,
                        "domain": "battery_electrolyte",
                        "task": "paper_to_runtime_decision",
                        "visible_context": "context",
                        "candidate_action": "act",
                        "action_type": "retrieve_more",
                        "expected_outcome": "more evidence",
                        "metric_key": "action_reliability",
                        "metric_direction": "maximize",
                        "metric_value": 0.5,
                        "outcome_success": bool(idx % 2),
                        "evidence": [],
                        "context_features": {},
                        "uncertainty_signals": {},
                        "verbal_confidence": 0.4,
                        "cost_level": "low",
                        "reversibility": "high",
                        "risk_level": "low",
                    }
                )
            )

        train, val, test, summary = split_records_by_group(records, seed=0, train_fraction=0.5, val_fraction=0.25)
        self.assertEqual(summary["groups"], 2)
        self.assertEqual(summary["train_groups"] + summary["val_groups"] + summary["test_groups"], 2)
        group_to_split = {}
        for split_name, split_records in [("train", train), ("val", val), ("test", test)]:
            for record in split_records:
                group_id = record.metadata["group_id"]
                if group_id in group_to_split:
                    self.assertEqual(group_to_split[group_id], split_name)
                else:
                    group_to_split[group_id] = split_name


if __name__ == "__main__":
    unittest.main()
