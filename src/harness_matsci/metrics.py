from __future__ import annotations

from .calibration import expected_calibration_error
from .models import brier_score, log_loss


def binary_metrics(labels: list[int], probabilities: list[float], threshold: float) -> dict[str, float]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have same length")
    if not labels:
        return {
            "n": 0.0,
            "coverage": 0.0,
            "selective_accuracy": 0.0,
            "selective_risk": 0.0,
            "confidently_wrong_rate": 0.0,
            "accuracy_at_threshold": 0.0,
            "ece": 0.0,
            "brier": 0.0,
            "log_loss": 0.0,
        }
    selected = [idx for idx, prob in enumerate(probabilities) if prob >= threshold]
    predictions = [1 if prob >= threshold else 0 for prob in probabilities]
    accuracy = sum(int(pred == label) for pred, label in zip(predictions, labels)) / len(labels)
    if selected:
        correct = sum(labels[idx] for idx in selected)
        selected_accuracy = correct / len(selected)
        selected_risk = 1.0 - selected_accuracy
    else:
        selected_accuracy = 0.0
        selected_risk = 0.0
    return {
        "n": float(len(labels)),
        "coverage": len(selected) / len(labels),
        "selective_accuracy": selected_accuracy,
        "selective_risk": selected_risk,
        "confidently_wrong_rate": sum(1 - labels[idx] for idx in selected) / len(labels),
        "accuracy_at_threshold": accuracy,
        "ece": expected_calibration_error(labels, probabilities),
        "brier": brier_score(labels, probabilities),
        "log_loss": log_loss(labels, probabilities),
    }


def discovery_gain(labels: list[int], utilities: list[float], probabilities: list[float], budget: int) -> dict[str, float]:
    if not labels or budget <= 0:
        return {"budget": float(max(0, budget)), "hit_rate": 0.0, "mean_utility": 0.0, "best_utility": 0.0}
    order = sorted(range(len(probabilities)), key=lambda idx: probabilities[idx], reverse=True)[:budget]
    return {
        "budget": float(min(budget, len(labels))),
        "hit_rate": sum(labels[idx] for idx in order) / len(order),
        "mean_utility": sum(utilities[idx] for idx in order) / len(order),
        "best_utility": max(utilities[idx] for idx in order),
    }

