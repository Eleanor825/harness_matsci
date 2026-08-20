from __future__ import annotations

import math

from harness_matsci.logit_uncertainty import calibrated_confidence, margin_to_uncertainty


def test_logit_confidence_is_symmetric_in_margin_magnitude() -> None:
    assert math.isclose(calibrated_confidence(2.0), calibrated_confidence(-2.0))
    assert calibrated_confidence(2.0) > calibrated_confidence(0.0)


def test_uncertainty_is_complement_of_confidence() -> None:
    assert math.isclose(margin_to_uncertainty(1.5), 1.0 - calibrated_confidence(1.5))


def test_temperature_controls_confidence_sharpness() -> None:
    assert calibrated_confidence(2.0, temperature=0.5) > calibrated_confidence(2.0, temperature=2.0)
