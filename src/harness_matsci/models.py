from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .features import sigmoid


@dataclass
class LogisticGate:
    """Small transparent action-reliability model.

    This deliberately avoids heavyweight ML dependencies so experiments are
    reproducible in restricted environments.  The model is strong enough for
    calibrated action gating and simple enough to audit in a paper.
    """

    feature_names: list[str]
    weights: list[float]
    bias: float = 0.0

    @classmethod
    def fresh(cls, feature_names: list[str]) -> "LogisticGate":
        return cls(feature_names=list(feature_names), weights=[0.0 for _ in feature_names], bias=0.0)

    def predict_logit(self, row: list[float]) -> float:
        return self.bias + sum(weight * value for weight, value in zip(self.weights, row))

    def predict_proba_row(self, row: list[float]) -> float:
        return sigmoid(self.predict_logit(row))

    def predict_proba(self, rows: Iterable[list[float]]) -> list[float]:
        return [self.predict_proba_row(row) for row in rows]

    def fit(
        self,
        rows: list[list[float]],
        labels: list[int],
        *,
        sample_weight: list[float] | None = None,
        epochs: int = 700,
        learning_rate: float = 0.08,
        l2: float = 0.001,
    ) -> "LogisticGate":
        if len(rows) != len(labels):
            raise ValueError("rows and labels must have same length")
        if not rows:
            raise ValueError("cannot train on an empty dataset")
        weights = sample_weight or [1.0 for _ in labels]
        if len(weights) != len(labels):
            raise ValueError("sample_weight and labels must have same length")
        n = max(1e-12, sum(weights))
        for _ in range(epochs):
            grad_w = [0.0 for _ in self.weights]
            grad_b = 0.0
            for row, label, row_weight in zip(rows, labels, weights):
                pred = self.predict_proba_row(row)
                error = (pred - label) * row_weight
                grad_b += error
                for idx, value in enumerate(row):
                    grad_w[idx] += error * value
            for idx, grad in enumerate(grad_w):
                self.weights[idx] -= learning_rate * ((grad / n) + l2 * self.weights[idx])
            self.bias -= learning_rate * grad_b / n
        return self

    def to_json(self) -> dict[str, object]:
        return {
            "type": "logistic_gate",
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
            "bias": self.bias,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "LogisticGate":
        if payload.get("type") != "logistic_gate":
            raise ValueError("not a logistic_gate payload")
        return cls(
            feature_names=[str(x) for x in payload["feature_names"]],
            weights=[float(x) for x in payload["weights"]],
            bias=float(payload["bias"]),
        )


def brier_score(labels: list[int], probabilities: list[float]) -> float:
    if not labels:
        return 0.0
    return sum((prob - label) ** 2 for label, prob in zip(labels, probabilities)) / len(labels)


def log_loss(labels: list[int], probabilities: list[float]) -> float:
    if not labels:
        return 0.0
    eps = 1e-12
    total = 0.0
    for label, prob in zip(labels, probabilities):
        prob = min(1.0 - eps, max(eps, prob))
        total += -(label * math.log(prob) + (1 - label) * math.log(1 - prob))
    return total / len(labels)

