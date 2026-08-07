from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .calibration import threshold_for_grouped_selective_risk, threshold_for_selective_risk
from .features import Standardizer, matrix_from_records
from .features import text_confidence
from .metrics import binary_metrics, discovery_gain, fixed_coverage_metrics, scientific_discovery_metrics
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
    min_coverage: float = 0.0,
    balance_benchmarks: bool = False,
) -> TrainedGate:
    if not train_records:
        raise ValueError("train_records cannot be empty")
    if not val_records:
        raise ValueError("val_records cannot be empty")
    train_matrix = matrix_from_records(train_records)
    standardizer = Standardizer().fit(train_matrix.x)
    x_train = standardizer.transform(train_matrix.x)
    sample_weight = _sample_weights(
        train_records,
        lineage_weight=lineage_weight,
        balance_benchmarks=balance_benchmarks,
    )
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
    calibration = _calibrate_threshold(
        val_records,
        val_matrix.y,
        val_probs,
        alpha=alpha,
        min_coverage=min_coverage,
        balance_benchmarks=balance_benchmarks,
    )
    return TrainedGate(model, standardizer, calibration["threshold"], calibration)


def train_gate_with_features(
    train_records: list[ActionRecord],
    val_records: list[ActionRecord],
    *,
    feature_names: list[str],
    alpha: float = 0.1,
    epochs: int = 700,
    learning_rate: float = 0.08,
    l2: float = 0.001,
    lineage_weight: float = 0.0,
    min_coverage: float = 0.0,
    balance_benchmarks: bool = False,
) -> TrainedGate:
    if not train_records:
        raise ValueError("train_records cannot be empty")
    if not val_records:
        raise ValueError("val_records cannot be empty")
    train_matrix = matrix_from_records(train_records, feature_names)
    standardizer = Standardizer().fit(train_matrix.x)
    x_train = standardizer.transform(train_matrix.x)
    sample_weight = _sample_weights(
        train_records,
        lineage_weight=lineage_weight,
        balance_benchmarks=balance_benchmarks,
    )
    model = LogisticGate.fresh(feature_names)
    model.fit(
        x_train,
        train_matrix.y,
        sample_weight=sample_weight,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    val_matrix = matrix_from_records(val_records, feature_names)
    val_probs = model.predict_proba(standardizer.transform(val_matrix.x))
    calibration = _calibrate_threshold(
        val_records,
        val_matrix.y,
        val_probs,
        alpha=alpha,
        min_coverage=min_coverage,
        balance_benchmarks=balance_benchmarks,
    )
    return TrainedGate(model, standardizer, calibration["threshold"], calibration)


def evaluate_gate(
    records: list[ActionRecord],
    gate: TrainedGate,
    *,
    budget_fraction: float = 0.1,
    macro_by_benchmark: bool = False,
) -> dict[str, Any]:
    matrix = matrix_from_records(records, gate.model.feature_names)
    probabilities = gate.predict_proba(records)
    metrics = binary_metrics(matrix.y, probabilities, gate.threshold)
    budget = max(1, int(len(records) * budget_fraction)) if records else 0
    gain = discovery_gain(matrix.y, [record.utility for record in records], probabilities, budget)
    decisions = [decision.to_json() for decision in gate.decisions(records)]
    report = {
        "metrics": metrics,
        "risk_coverage": fixed_coverage_metrics(matrix.y, probabilities),
        "discovery_gain": gain,
        "scientific_discovery": scientific_discovery_metrics(records, probabilities, budget),
        "threshold": gate.threshold,
        "n_records": len(records),
        "decisions": decisions,
    }
    if macro_by_benchmark:
        benchmarks = sorted({record.benchmark for record in records})
        slices = {
            benchmark: evaluate_gate(
                [record for record in records if record.benchmark == benchmark],
                gate,
                budget_fraction=budget_fraction,
            )
            for benchmark in benchmarks
        }
        report["macro_metrics"] = _macro_dict([item["metrics"] for item in slices.values()])
        report["macro_discovery_gain"] = _macro_dict([item["discovery_gain"] for item in slices.values()])
        report["by_benchmark"] = slices
    return report


def _sample_weights(
    records: list[ActionRecord],
    *,
    lineage_weight: float,
    balance_benchmarks: bool,
) -> list[float]:
    counts = Counter(record.benchmark for record in records)
    n_benchmarks = max(1, len(counts))
    n_records = len(records)
    weights: list[float] = []
    for record in records:
        task_weight = n_records / (n_benchmarks * counts[record.benchmark]) if balance_benchmarks else 1.0
        lineage = 1.0 + lineage_weight * record.features.get("lineage_impact", 0.0)
        weights.append(task_weight * lineage)
    return weights


def _macro_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    shared = set.intersection(*(set(row) for row in rows))
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in sorted(shared)}


def verbal_confidence_scores(records: list[ActionRecord]) -> list[float]:
    scores: list[float] = []
    for record in records:
        if "verbal_confidence" in record.features:
            scores.append(float(record.features["verbal_confidence"]))
        else:
            confidence, _, _ = text_confidence(f"{record.visible_context} {record.candidate_action}")
            scores.append(confidence)
    return scores


def evidence_heuristic_scores(records: list[ActionRecord]) -> list[float]:
    scores: list[float] = []
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
        scores.append(max(0.0, min(1.0, 0.42 + score)))
    return scores


def evaluate_probability_signal(
    validation_records: list[ActionRecord],
    test_records: list[ActionRecord],
    score_fn,
    *,
    alpha: float = 0.1,
    budget_fraction: float = 0.1,
    min_coverage: float = 0.0,
    balance_benchmarks: bool = False,
) -> dict[str, Any]:
    val_labels = [record.label for record in validation_records]
    val_probabilities = [float(prob) for prob in score_fn(validation_records)]
    calibration = _calibrate_threshold(
        validation_records,
        val_labels,
        val_probabilities,
        alpha=alpha,
        min_coverage=min_coverage,
        balance_benchmarks=balance_benchmarks,
    )
    test_labels = [record.label for record in test_records]
    test_probabilities = [float(prob) for prob in score_fn(test_records)]
    return {
        "threshold": calibration["threshold"],
        "calibration": calibration,
        "validation": {
            "metrics": binary_metrics(val_labels, val_probabilities, calibration["threshold"]),
            "discovery_gain": discovery_gain(
                val_labels,
                [record.utility for record in validation_records],
                val_probabilities,
                max(1, int(len(validation_records) * budget_fraction)),
            ),
            "scientific_discovery": scientific_discovery_metrics(
                validation_records,
                val_probabilities,
                max(1, int(len(validation_records) * budget_fraction)),
            ),
            "n_records": len(validation_records),
        },
        "test": {
            "metrics": binary_metrics(test_labels, test_probabilities, calibration["threshold"]),
            "risk_coverage": fixed_coverage_metrics(test_labels, test_probabilities),
            "discovery_gain": discovery_gain(
                test_labels,
                [record.utility for record in test_records],
                test_probabilities,
                max(1, int(len(test_records) * budget_fraction)),
            ),
            "scientific_discovery": scientific_discovery_metrics(
                test_records,
                test_probabilities,
                max(1, int(len(test_records) * budget_fraction)),
            ),
            "n_records": len(test_records),
        },
    }


def verbal_confidence_baseline(records: list[ActionRecord], *, alpha: float = 0.1, budget_fraction: float = 0.1) -> dict[str, Any]:
    report = evaluate_probability_signal(records, records, verbal_confidence_scores, alpha=alpha, budget_fraction=budget_fraction)
    return {
        "threshold": report["threshold"],
        "calibration": report["calibration"],
        "metrics": report["test"]["metrics"],
        "discovery_gain": report["test"]["discovery_gain"],
    }


def evidence_heuristic_baseline(records: list[ActionRecord], *, alpha: float = 0.1, budget_fraction: float = 0.1) -> dict[str, Any]:
    report = evaluate_probability_signal(records, records, evidence_heuristic_scores, alpha=alpha, budget_fraction=budget_fraction)
    return {
        "threshold": report["threshold"],
        "calibration": report["calibration"],
        "metrics": report["test"]["metrics"],
        "discovery_gain": report["test"]["discovery_gain"],
    }


def _calibrate_threshold(
    records: list[ActionRecord],
    labels: list[int],
    probabilities: list[float],
    *,
    alpha: float,
    min_coverage: float,
    balance_benchmarks: bool,
) -> dict[str, float]:
    if balance_benchmarks and len({record.benchmark for record in records}) > 1:
        return threshold_for_grouped_selective_risk(
            labels,
            probabilities,
            [record.benchmark for record in records],
            alpha=alpha,
            min_coverage=min_coverage,
        )
    return threshold_for_selective_risk(
        labels,
        probabilities,
        alpha=alpha,
        min_coverage=min_coverage,
    )
