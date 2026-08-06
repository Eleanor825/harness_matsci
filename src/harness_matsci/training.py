from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .calibration import threshold_for_selective_risk
from .features import Standardizer, matrix_from_records
from .metrics import binary_metrics, discovery_gain
from .models import LogisticGate
from .routing import route_record
from .schema import ActionRecord, GateDecision


@dataclass
class TrainedGate:
    model: LogisticGate
    standardizer: Standardizer
    threshold: float
    calibration: dict[str, float]

    def predict_proba(self, records: list[ActionRecord]) -> list[float]:
        matrix = matrix_from_records(records, self.model.feature_names)
        rows = self.standardizer.transform(matrix.x)
        return self.model.predict_proba(rows)

    def decisions(self, records: list[ActionRecord]) -> list[GateDecision]:
        return [route_record(record, prob, self.threshold) for record, prob in zip(records, self.predict_proba(records))]

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model.to_json(),
            "standardizer": self.standardizer.to_json(),
            "threshold": self.threshold,
            "calibration": dict(self.calibration),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "TrainedGate":
        return cls(
            model=LogisticGate.from_json(payload["model"]),
            standardizer=Standardizer.from_json(payload["standardizer"]),
            threshold=float(payload["threshold"]),
            calibration={str(k): float(v) for k, v in payload.get("calibration", {}).items()},
        )


def split_records(
    records: list[ActionRecord],
    *,
    seed: int = 0,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
) -> tuple[list[ActionRecord], list[ActionRecord], list[ActionRecord]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_fraction)
    n_val = int(len(shuffled) * val_fraction)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def train_gate(
    train_records: list[ActionRecord],
    val_records: list[ActionRecord],
    *,
    alpha: float = 0.1,
    epochs: int = 700,
    learning_rate: float = 0.08,
    l2: float = 0.001,
    lineage_weight: float = 0.0,
) -> TrainedGate:
    if not train_records:
        raise ValueError("train_records cannot be empty")
    if not val_records:
        raise ValueError("val_records cannot be empty")
    train_matrix = matrix_from_records(train_records)
    standardizer = Standardizer().fit(train_matrix.x)
    x_train = standardizer.transform(train_matrix.x)
    sample_weight = [1.0 + lineage_weight * record.features.get("lineage_impact", 0.0) for record in train_records]
    model = LogisticGate.fresh(train_matrix.feature_names)
    model.fit(
        x_train,
        train_matrix.y,
        sample_weight=sample_weight,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )

    val_matrix = matrix_from_records(val_records, train_matrix.feature_names)
    val_probs = model.predict_proba(standardizer.transform(val_matrix.x))
    calibration = threshold_for_selective_risk(val_matrix.y, val_probs, alpha=alpha)
    return TrainedGate(model, standardizer, calibration["threshold"], calibration)


def evaluate_gate(records: list[ActionRecord], gate: TrainedGate, *, budget_fraction: float = 0.1) -> dict[str, Any]:
    matrix = matrix_from_records(records, gate.model.feature_names)
    probabilities = gate.predict_proba(records)
    metrics = binary_metrics(matrix.y, probabilities, gate.threshold)
    budget = max(1, int(len(records) * budget_fraction)) if records else 0
    gain = discovery_gain(matrix.y, [record.utility for record in records], probabilities, budget)
    decisions = [decision.to_json() for decision in gate.decisions(records)]
    return {
        "metrics": metrics,
        "discovery_gain": gain,
        "threshold": gate.threshold,
        "n_records": len(records),
        "decisions": decisions,
    }


def verbal_confidence_baseline(records: list[ActionRecord], *, alpha: float = 0.1, budget_fraction: float = 0.1) -> dict[str, Any]:
    labels = [record.label for record in records]
    probabilities = [float(record.features.get("verbal_confidence", 0.5)) for record in records]
    calibration = threshold_for_selective_risk(labels, probabilities, alpha=alpha)
    return {
        "threshold": calibration["threshold"],
        "calibration": calibration,
        "metrics": binary_metrics(labels, probabilities, calibration["threshold"]),
        "discovery_gain": discovery_gain(labels, [record.utility for record in records], probabilities, max(1, int(len(records) * budget_fraction))),
    }


def evidence_heuristic_baseline(records: list[ActionRecord], *, alpha: float = 0.1, budget_fraction: float = 0.1) -> dict[str, Any]:
    labels = [record.label for record in records]
    probabilities = []
    for record in records:
        f = record.features
        score = (
            0.28 * f.get("evidence_support", 0.0)
            + 0.20 * f.get("perturbation_stability", 0.5)
            + 0.16 * f.get("tool_agreement", 0.5)
            + 0.14 * f.get("source_reliability", 0.5)
            + 0.12 * f.get("reversibility", 0.5)
            - 0.22 * f.get("evidence_conflict", 0.0)
            - 0.16 * f.get("ood_score", 0.0)
            - 0.10 * f.get("model_disagreement", 0.0)
        )
        probabilities.append(max(0.0, min(1.0, 0.42 + score)))
    calibration = threshold_for_selective_risk(labels, probabilities, alpha=alpha)
    return {
        "threshold": calibration["threshold"],
        "calibration": calibration,
        "metrics": binary_metrics(labels, probabilities, calibration["threshold"]),
        "discovery_gain": discovery_gain(labels, [record.utility for record in records], probabilities, max(1, int(len(records) * budget_fraction))),
    }
