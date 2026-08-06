from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import mean, pstdev

from .schema import ActionRecord


DEFAULT_FEATURES = [
    "audited_ld",
    "preference_margin",
    "verbal_confidence",
    "perturbation_stability",
    "evidence_support",
    "evidence_conflict",
    "source_reliability",
    "tool_agreement",
    "model_disagreement",
    "ood_score",
    "experimental_variance",
    "cost",
    "reversibility",
    "lineage_impact",
]

POSITIVE_TERMS = {
    "improve",
    "improved",
    "increase",
    "increased",
    "enhance",
    "enhanced",
    "stable",
    "robust",
    "promising",
    "effective",
    "superior",
    "high",
    "better",
    "valid",
    "hit",
}

NEGATIVE_TERMS = {
    "decrease",
    "decreased",
    "degrade",
    "degraded",
    "unstable",
    "risk",
    "conflict",
    "failed",
    "failure",
    "weak",
    "poor",
    "invalid",
    "uncertain",
    "unknown",
}


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def text_confidence(text: str) -> tuple[float, float, float]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return 0.4, 0.0, 0.0
    positives = sum(token in POSITIVE_TERMS for token in tokens)
    negatives = sum(token in NEGATIVE_TERMS for token in tokens)
    support = clamp(positives / max(4.0, positives + negatives + 2.0))
    conflict = clamp(negatives / max(4.0, positives + negatives + 2.0))
    confidence = clamp(0.4 + 0.08 * positives - 0.07 * negatives)
    return confidence, support, conflict


@dataclass
class FeatureMatrix:
    feature_names: list[str]
    x: list[list[float]]
    y: list[int]
    record_ids: list[str]


class Standardizer:
    def __init__(self, means: list[float] | None = None, scales: list[float] | None = None):
        self.means = means or []
        self.scales = scales or []

    def fit(self, rows: list[list[float]]) -> "Standardizer":
        if not rows:
            self.means = []
            self.scales = []
            return self
        cols = list(zip(*rows))
        self.means = [mean(col) for col in cols]
        self.scales = [pstdev(col) or 1.0 for col in cols]
        return self

    def transform_row(self, row: list[float]) -> list[float]:
        return [(value - mu) / scale for value, mu, scale in zip(row, self.means, self.scales)]

    def transform(self, rows: list[list[float]]) -> list[list[float]]:
        return [self.transform_row(row) for row in rows]

    def to_json(self) -> dict[str, list[float]]:
        return {"means": list(self.means), "scales": list(self.scales)}

    @classmethod
    def from_json(cls, payload: dict[str, list[float]]) -> "Standardizer":
        return cls(means=[float(x) for x in payload["means"]], scales=[float(x) for x in payload["scales"]])


def feature_names_from_records(records: list[ActionRecord], preferred: list[str] | None = None) -> list[str]:
    observed = {name for record in records for name in record.features}
    ordered = [name for name in (preferred or DEFAULT_FEATURES) if name in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def matrix_from_records(records: list[ActionRecord], feature_names: list[str] | None = None) -> FeatureMatrix:
    names = feature_names or feature_names_from_records(records)
    rows: list[list[float]] = []
    labels: list[int] = []
    ids: list[str] = []
    for record in records:
        rows.append([float(record.features.get(name, 0.0)) for name in names])
        labels.append(record.label)
        ids.append(record.record_id)
    return FeatureMatrix(names, rows, labels, ids)

