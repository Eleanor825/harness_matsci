from __future__ import annotations

import math


def threshold_for_selective_risk(
    labels: list[int],
    probabilities: list[float],
    *,
    alpha: float = 0.1,
    min_coverage: float = 0.0,
) -> dict[str, float]:
    """Choose the lowest threshold that maximizes coverage under risk <= alpha."""
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have same length")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be in [0, 1]")
    if not labels:
        return {
            "threshold": 1.0,
            "coverage": 0.0,
            "selective_risk": 0.0,
            "selected": 0.0,
            "constraint_satisfied": 0.0,
        }

    candidates = sorted(set(probabilities), reverse=True)
    best: dict[str, float] | None = None
    n = len(labels)
    for threshold in candidates:
        selected = [idx for idx, prob in enumerate(probabilities) if prob >= threshold]
        if not selected:
            continue
        coverage = len(selected) / n
        risk = sum(1 - labels[idx] for idx in selected) / len(selected)
        if risk <= alpha and coverage >= min_coverage:
            if best is None or coverage > best["coverage"] or (coverage == best["coverage"] and threshold < best["threshold"]):
                best = {
                    "threshold": float(threshold),
                    "coverage": float(coverage),
                    "selective_risk": float(risk),
                    "selected": float(len(selected)),
                    "constraint_satisfied": 1.0,
                }
    if best is not None:
        return best

    # If no threshold satisfies both constraints, choose the smallest prefix
    # that meets the requested coverage.  This makes an infeasible risk
    # contract visible instead of allowing a one-example abstention policy to
    # look like a successful uncertainty guarantee.
    n_selected = max(1, math.ceil(len(labels) * min_coverage))
    order = sorted(range(len(probabilities)), key=lambda idx: probabilities[idx], reverse=True)
    threshold = probabilities[order[n_selected - 1]]
    selected = [idx for idx, probability in enumerate(probabilities) if probability >= threshold]
    risk = sum(1 - labels[idx] for idx in selected) / len(selected)
    return {
        "threshold": float(threshold),
        "coverage": len(selected) / n,
        "selective_risk": float(risk),
        "selected": float(len(selected)),
        "constraint_satisfied": 0.0,
    }


def threshold_for_grouped_selective_risk(
    labels: list[int],
    probabilities: list[float],
    groups: list[str],
    *,
    alpha: float = 0.1,
    min_coverage: float = 0.0,
) -> dict[str, float]:
    if len(labels) != len(probabilities) or len(labels) != len(groups):
        raise ValueError("labels, probabilities, and groups must have same length")
    if not labels:
        return threshold_for_selective_risk(labels, probabilities, alpha=alpha, min_coverage=min_coverage)
    group_names = sorted(set(groups))
    candidates = sorted(set(probabilities), reverse=True)
    best: dict[str, float] | None = None
    fallback_candidates: list[dict[str, float]] = []
    for threshold in candidates:
        group_coverages: list[float] = []
        group_risks: list[float] = []
        for group in group_names:
            members = [index for index, value in enumerate(groups) if value == group]
            selected = [index for index in members if probabilities[index] >= threshold]
            group_coverages.append(len(selected) / len(members))
            group_risks.append(sum(1 - labels[index] for index in selected) / len(selected) if selected else 1.0)
        macro_coverage = sum(group_coverages) / len(group_coverages)
        macro_risk = sum(group_risks) / len(group_risks)
        minimum_group_coverage = min(group_coverages)
        selected_count = sum(probability >= threshold for probability in probabilities)
        candidate = {
            "threshold": float(threshold),
            "coverage": float(macro_coverage),
            "minimum_group_coverage": float(minimum_group_coverage),
            "selective_risk": float(macro_risk),
            "selected": float(selected_count),
        }
        if minimum_group_coverage >= min_coverage:
            fallback_candidates.append(candidate)
        if macro_risk <= alpha and minimum_group_coverage >= min_coverage:
            if best is None or macro_coverage > best["coverage"]:
                best = {
                    **candidate,
                    "constraint_satisfied": 1.0,
                }
    if best is not None:
        return best
    if fallback_candidates:
        fallback = min(
            fallback_candidates,
            key=lambda item: (item["selective_risk"], item["coverage"]),
        )
        return {**fallback, "constraint_satisfied": 0.0}
    raise RuntimeError("grouped calibration could not satisfy minimum per-group coverage")


def expected_calibration_error(labels: list[int], probabilities: list[float], bins: int = 10) -> float:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have same length")
    if not labels:
        return 0.0
    total = 0.0
    n = len(labels)
    for bin_idx in range(bins):
        lo = bin_idx / bins
        hi = (bin_idx + 1) / bins
        members = [idx for idx, prob in enumerate(probabilities) if (lo <= prob < hi) or (bin_idx == bins - 1 and prob == 1.0)]
        if not members:
            continue
        conf = sum(probabilities[idx] for idx in members) / len(members)
        acc = sum(labels[idx] for idx in members) / len(members)
        total += len(members) / n * abs(acc - conf)
    return total


def risk_coverage_curve(labels: list[int], probabilities: list[float]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for threshold in sorted(set(probabilities), reverse=True):
        selected = [idx for idx, prob in enumerate(probabilities) if prob >= threshold]
        if not selected:
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": len(selected) / len(labels),
                "selective_risk": sum(1 - labels[idx] for idx in selected) / len(selected),
            }
        )
    return rows
