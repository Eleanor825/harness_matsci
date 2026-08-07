from __future__ import annotations

import unittest

from harness_matsci.calibration import threshold_for_selective_risk
from harness_matsci.models import LogisticGate


class CalibrationAndModelTests(unittest.TestCase):
    def test_selective_risk_threshold(self) -> None:
        labels = [1, 0, 1, 0]
        probs = [0.9, 0.8, 0.4, 0.1]
        calibration = threshold_for_selective_risk(labels, probs, alpha=0.25)
        self.assertAlmostEqual(calibration["threshold"], 0.9)
        self.assertAlmostEqual(calibration["coverage"], 0.25)
        self.assertAlmostEqual(calibration["selective_risk"], 0.0)

    def test_logistic_gate_learns_monotone_signal(self) -> None:
        gate = LogisticGate.fresh(["x"])
        rows = [[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]]
        labels = [0, 0, 0, 1, 1, 1]
        gate.fit(rows, labels, epochs=1200, learning_rate=0.1, l2=0.0)
        low = gate.predict_proba_row([-1.5])
        high = gate.predict_proba_row([1.5])
        self.assertLess(low, high)
        self.assertGreater(high, 0.7)
        self.assertLess(low, 0.3)

    def test_infeasible_risk_contract_respects_minimum_coverage(self) -> None:
        calibration = threshold_for_selective_risk(
            [0, 0, 1, 0],
            [0.9, 0.8, 0.7, 0.6],
            alpha=0.0,
            min_coverage=0.5,
        )
        self.assertGreaterEqual(calibration["coverage"], 0.5)
        self.assertEqual(calibration["constraint_satisfied"], 0.0)


if __name__ == "__main__":
    unittest.main()
