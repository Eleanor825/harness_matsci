from __future__ import annotations

import pytest

from harness_matsci.benchmarks import make_records
from harness_matsci.uncertainty_signals import CachedConfidenceProvider, DirectJudgeSignalProvider, attach_signal_features


def test_cached_provider_attaches_pluggable_signal() -> None:
    records = make_records("preferential_bo", n=4, seed=3)
    provider = CachedConfidenceProvider({record.record_id: 0.7 for record in records})
    augmented = attach_signal_features(records, provider)
    assert all(record.features["llm_signal_confidence"] == 0.7 for record in augmented)
    assert all(record.features["llm_signal_uncertainty"] == pytest.approx(0.3) for record in augmented)
    assert all(record.metadata["uncertainty_signal_source"] == "llm_self_report" for record in augmented)


def test_cached_provider_rejects_missing_scores() -> None:
    records = make_records("preferential_bo", n=2, seed=4)
    with pytest.raises(ValueError):
        CachedConfidenceProvider({records[0].record_id: 0.5}).score(records)


def test_direct_judge_adapter_is_provider_agnostic() -> None:
    records = make_records("preferential_bo", n=3, seed=5)

    class FakeJudge:
        def score_records(self, items):
            return {item.record_id: 0.8 for item in items}

    provider = DirectJudgeSignalProvider(FakeJudge())
    signals = provider.score(records)
    assert all(signal.confidence == 0.8 for signal in signals.values())
