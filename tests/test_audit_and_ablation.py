from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness_matsci.audit_experiments import LabelAuditConfig, run_label_utility_audit
from harness_matsci.voi_experiments import _static_utility_harness, _static_voi_harness, _static_voi_no_cost_harness


class AuditAndAblationTests(unittest.TestCase):
    def test_label_utility_audit_passes_on_minimal_historical_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _write_jsonl(
                data_dir / "pairwise_optimization_actions.jsonl",
                [
                    _pairwise_payload("pair-a", "group-1", 0.2),
                    _pairwise_payload("pair-b", "group-1", 0.8),
                ],
            )
            _write_jsonl(
                data_dir / "unique_materials_actions.jsonl",
                [
                    {
                        "record_id": "unique-1",
                        "group_id": "unique-group",
                        "visible_context": "Screen candidate; log10(K_VRH)=1.23.",
                        "candidate_action": "Screen CaAgGe.",
                        "action_type": "choose_candidate",
                        "outcome_success": True,
                        "metric_value": 0.7,
                        "uncertainty_signals": {"discovery_score": 0.7, "evidence_support": 1.0},
                        "context_features": {},
                        "tool_outputs": {"utility": 0.7},
                    }
                ],
            )
            _write_jsonl(
                data_dir / "extreme_properties_actions.jsonl",
                [
                    {
                        "record_id": "extreme-1",
                        "group_id": "extreme-group",
                        "visible_context": "Extreme candidate",
                        "candidate_action": "Advance molecule: CCO",
                        "action_type": "choose_candidate",
                        "outcome_success": False,
                        "metric_value": 0.4,
                        "evidence": ["hit_fraction=0.5", "reward=1.0", "all_hit=False"],
                        "uncertainty_signals": {"hit_fraction": 0.5, "target_hit_score": 0.4},
                        "context_features": {"hit_fraction": 0.5},
                        "tool_outputs": {"utility": 0.4},
                    }
                ],
            )
            report = run_label_utility_audit(LabelAuditConfig(data_dir=str(data_dir), sample_per_task=2))
            self.assertTrue(report["aggregate"]["all_label_consistency_passed"])
            self.assertTrue(report["aggregate"]["all_utility_consistency_passed"])
            self.assertTrue(report["aggregate"]["all_visible_leakage_free"])

    def test_static_voi_no_cost_zeroes_decision_cost_weights(self) -> None:
        harness = _static_voi_no_cost_harness(["cost", "evidence_support"])
        self.assertEqual(harness["name"], "static_voi_no_cost")
        self.assertEqual(harness["decision_mode"], "voi")
        self.assertEqual(harness["execute_cost_weight"], 0.0)
        self.assertEqual(harness["verification_cost_weight"], 0.0)
        self.assertNotIn("cost", harness["required_features"])
        self.assertNotIn("cost", harness["utility_features"])

    def test_static_harness_builders_drop_cost_feature_when_requested(self) -> None:
        utility = _static_utility_harness(["cost", "evidence_support"], use_cost_signal=False)
        voi = _static_voi_harness(["cost", "model_disagreement", "evidence_support"], use_cost_signal=False, use_uncertainty_signal=False, allow_verification=False)
        self.assertNotIn("cost", utility["required_features"])
        self.assertNotIn("cost", utility["utility_features"])
        self.assertNotIn("cost", voi["required_features"])
        self.assertNotIn("cost", voi["utility_features"])
        self.assertNotIn("model_disagreement", voi["required_features"])
        self.assertNotIn("model_disagreement", voi["utility_features"])


def _pairwise_payload(record_id: str, group_id: str, utility: float) -> dict[str, object]:
    return {
        "record_id": record_id,
        "group_id": group_id,
        "visible_context": "Preferential BO candidate.",
        "candidate_action": f"Select candidate point for the next duel: x0={utility}",
        "action_type": "choose_candidate",
        "metric_value": utility,
        "outcome_success": utility > 0.5,
        "tool_outputs": {"utility": utility},
        "context_features": {"dimension": 1.0, "domain_center_distance": 0.3},
        "uncertainty_signals": {"latent_utility": utility, "preference_strength": utility},
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
