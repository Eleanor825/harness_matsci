from __future__ import annotations


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
    if not labels:
        return {"threshold": 1.0, "coverage": 0.0, "selective_risk": 0.0, "selected": 0.0}

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
                }
    if best is not None:
        return best

    # If no threshold satisfies the risk contract, choose the most reliable one
    # and report the violation rather than silently returning full coverage.
    threshold = max(probabilities)
    selected = [idx for idx, prob in enumerate(probabilities) if prob >= threshold]
    risk = sum(1 - labels[idx] for idx in selected) / len(selected)
    return {
        "threshold": float(threshold),
        "coverage": len(selected) / n,
        "selective_risk": float(risk),
        "selected": float(len(selected)),
    }


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

