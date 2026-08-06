from __future__ import annotations

import unittest

from harness_matsci.schema import ActionRecord, HarnessSpec


class SchemaTests(unittest.TestCase):
    def test_action_record_roundtrip(self) -> None:
        record = ActionRecord(
            record_id="r1",
            benchmark="preferential_bo",
            split="train",
            visible_context="context",
            candidate_action="choose A",
            action_type="choose",
            evidence=["e1", "e2"],
            features={"audited_ld": 0.8, "cost": 0.2, "evidence_support": 0.7},
            label=1,
            utility=0.33,
            metadata={"source": "synthetic"},
        )

        restored = ActionRecord.from_json(record.to_json())
        self.assertEqual(restored, record)

    def test_harness_requires_features(self) -> None:
        spec = HarnessSpec(
            name="gate",
            required_features=["audited_ld", "cost"],
            proceed_routes=["proceed"],
            fallback_routes=["abstain"],
            target_selective_risk=0.1,
        )
        record = ActionRecord(
            record_id="r2",
            benchmark="toy",
            split="train",
            visible_context="context",
            candidate_action="act",
            action_type="recommend",
            evidence=[],
            features={"audited_ld": 0.8, "cost": 0.2},
            label=1,
        )
        spec.validate_record(record)
        with self.assertRaises(ValueError):
            spec.validate_record(
                ActionRecord(
                    record_id="r3",
                    benchmark="toy",
                    split="train",
                    visible_context="context",
                    candidate_action="act",
                    action_type="recommend",
                    evidence=[],
                    features={"audited_ld": 0.8},
                    label=1,
                )
            )


if __name__ == "__main__":
    unittest.main()
