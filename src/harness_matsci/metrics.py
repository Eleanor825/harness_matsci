from __future__ import annotations

import math

from .calibration import expected_calibration_error
from .models import brier_score, log_loss
from .schema import ActionRecord


def action_worthiness_score(metrics: dict[str, float], gain: dict[str, float]) -> float:
    """A threshold-independent diagnostic score for method selection.

    The fixed-budget terms prevent a policy that abstains on every action from
    appearing optimal.  This is a reporting/selection score, not a claim that
    the three task utilities are interchangeable.
    """
    return (
        0.25 * float(metrics.get("aurc", 1.0))
        + 0.20 * float(metrics.get("brier", 1.0))
        + 0.15 * float(metrics.get("ece", 1.0))
        + 0.15 * float(metrics.get("log_loss", 1.0))
        + 0.15 * (1.0 - float(gain.get("hit_efficiency", 0.0)))
        + 0.10 * (1.0 - float(gain.get("utility_efficiency", 0.0)))
    )


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
            "aurc": 0.0,
            "coverage_at_risk_0.05": 0.0,
            "coverage_at_risk_0.10": 0.0,
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
    ranking = ranking_metrics(labels, probabilities)
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
        **ranking,
    }


def ranking_metrics(labels: list[int], probabilities: list[float]) -> dict[str, float]:
    """Summarize the full risk-coverage ranking independently of a threshold."""
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have same length")
    if not labels:
        return {"aurc": 0.0, "coverage_at_risk_0.05": 0.0, "coverage_at_risk_0.10": 0.0}
    order = sorted(range(len(probabilities)), key=lambda idx: probabilities[idx], reverse=True)
    cumulative_errors = 0
    risks: list[float] = []
    coverage_at_risk = {0.05: 0.0, 0.1: 0.0}
    for rank, index in enumerate(order, 1):
        cumulative_errors += 1 - labels[index]
        risk = cumulative_errors / rank
        coverage = rank / len(labels)
        risks.append(risk)
        for risk_limit in coverage_at_risk:
            if risk <= risk_limit:
                coverage_at_risk[risk_limit] = coverage
    return {
        "aurc": sum(risks) / len(risks),
        "coverage_at_risk_0.05": coverage_at_risk[0.05],
        "coverage_at_risk_0.10": coverage_at_risk[0.1],
    }


def fixed_coverage_metrics(
    labels: list[int],
    probabilities: list[float],
    *,
    coverages: tuple[float, ...] = (0.1, 0.25, 0.5),
) -> dict[str, dict[str, float]]:
    """Report risk at fixed coverage without selecting a test threshold."""
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have same length")
    if not labels:
        return {f"risk_at_{coverage:.2f}": {"coverage": 0.0, "selective_risk": 0.0} for coverage in coverages}
    order = sorted(range(len(probabilities)), key=lambda idx: probabilities[idx], reverse=True)
    result: dict[str, dict[str, float]] = {}
    for requested_coverage in coverages:
        if not 0 < requested_coverage <= 1:
            raise ValueError("fixed coverages must be in (0, 1]")
        n_selected = max(1, min(len(labels), math.ceil(len(labels) * requested_coverage)))
        selected = order[:n_selected]
        result[f"risk_at_{requested_coverage:.2f}"] = {
            "coverage": n_selected / len(labels),
            "selective_risk": sum(1 - labels[idx] for idx in selected) / n_selected,
        }
    return result


def discovery_gain(labels: list[int], utilities: list[float], probabilities: list[float], budget: int) -> dict[str, float]:
    if not labels or budget <= 0:
        return {
            "budget": float(max(0, budget)),
            "hit_rate": 0.0,
            "mean_utility": 0.0,
            "best_utility": 0.0,
            "random_hit_rate": 0.0,
            "random_mean_utility": 0.0,
            "hit_rate_lift": 0.0,
            "mean_utility_lift": 0.0,
            "normalized_hit_lift": 0.0,
            "normalized_utility_lift": 0.0,
            "oracle_hit_rate": 0.0,
            "oracle_mean_utility": 0.0,
            "hit_efficiency": 0.0,
            "utility_efficiency": 0.0,
        }
    order = sorted(range(len(probabilities)), key=lambda idx: probabilities[idx], reverse=True)[:budget]
    oracle_order = sorted(range(len(utilities)), key=lambda idx: utilities[idx], reverse=True)[:budget]
    random_hit_rate = sum(labels) / len(labels)
    random_mean_utility = sum(utilities) / len(utilities)
    selected_mean_utility = sum(utilities[idx] for idx in order) / len(order)
    selected_hit_rate = sum(labels[idx] for idx in order) / len(order)
    oracle_hit_rate = sum(labels[idx] for idx in oracle_order) / len(oracle_order)
    oracle_mean_utility = sum(utilities[idx] for idx in oracle_order) / len(oracle_order)
    hit_gap = oracle_hit_rate - random_hit_rate
    utility_gap = oracle_mean_utility - random_mean_utility
    return {
        "budget": float(min(budget, len(labels))),
        "hit_rate": selected_hit_rate,
        "mean_utility": selected_mean_utility,
        "best_utility": max(utilities[idx] for idx in order),
        "random_hit_rate": random_hit_rate,
        "random_mean_utility": random_mean_utility,
        "hit_rate_lift": selected_hit_rate - random_hit_rate,
        "mean_utility_lift": selected_mean_utility - random_mean_utility,
        "normalized_hit_lift": (selected_hit_rate - random_hit_rate) / hit_gap if hit_gap > 0 else 0.0,
        "normalized_utility_lift": (selected_mean_utility - random_mean_utility) / utility_gap if utility_gap > 0 else 0.0,
        "oracle_hit_rate": oracle_hit_rate,
        "oracle_mean_utility": oracle_mean_utility,
        "hit_efficiency": selected_hit_rate / oracle_hit_rate if oracle_hit_rate > 0 else 0.0,
        "utility_efficiency": selected_mean_utility / oracle_mean_utility if oracle_mean_utility > 0 else 0.0,
    }


def scientific_discovery_metrics(
    records: list[ActionRecord],
    probabilities: list[float],
    budget: int,
) -> dict[str, float | str]:
    if len(records) != len(probabilities):
        raise ValueError("records and probabilities must have same length")
    if not records or budget <= 0:
        return {
            "task": "empty",
            "positive_recall": 0.0,
            "simple_regret": 0.0,
            "regime_coverage": 0.0,
            "selected_regimes": 0.0,
        }
    selected = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)[:budget]
    positives = sum(record.label for record in records)
    selected_positives = sum(records[index].label for index in selected)
    all_regimes = {str(record.metadata.get("group_id", record.benchmark)) for record in records}
    selected_regimes = {str(records[index].metadata.get("group_id", records[index].benchmark)) for index in selected}
    oracle_best = max(record.utility for record in records)
    selected_best = max(records[index].utility for index in selected)
    task = records[0].benchmark if len({record.benchmark for record in records}) == 1 else "mixed"
    result: dict[str, float | str] = {
        "task": task,
        "positive_recall": selected_positives / positives if positives else 0.0,
        "simple_regret": max(0.0, oracle_best - selected_best),
        "regime_coverage": len(selected_regimes) / len(all_regimes),
        "selected_regimes": float(len(selected_regimes)),
    }
    if task == "preferential_bo":
        latent_values = [float(record.metadata.get("chosen_true_utility", record.utility)) for record in records]
        latent_oracle = max(latent_values)
        latent_selected = max(latent_values[index] for index in selected)
        result["pairwise_latent_regret"] = max(0.0, latent_oracle - latent_selected)
        result["preferential_hit_recall"] = result["positive_recall"]
    elif task == "discover_unique":
        result["unique_material_recall"] = result["positive_recall"]
        result["crystal_regime_coverage"] = result["regime_coverage"]
    elif task == "extreme_properties":
        result["extreme_hit_recall"] = result["positive_recall"]
        result["target_regime_coverage"] = result["regime_coverage"]
    return result
