from __future__ import annotations

import math
import random
from collections.abc import Callable

from harness_matsci.features import clamp, sigmoid
from harness_matsci.schema import ActionRecord

from .common import action_features, split_name


Objective = tuple[str, tuple[float, ...], tuple[float, ...], Callable[[tuple[float, ...]], float]]


def _forrester(point: tuple[float, ...]) -> float:
    x = point[0]
    return ((6.0 * x - 2.0) ** 2) * math.sin(12.0 * x - 4.0)


def _six_hump_camel(point: tuple[float, ...]) -> float:
    x, y = point
    return (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2


def _goldstein_price(point: tuple[float, ...]) -> float:
    x, y = point
    left = 1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    right = 30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    return math.log1p(abs(left * right))


def _levy(point: tuple[float, ...]) -> float:
    x, y = point
    w1 = 1 + (x - 1) / 4
    w2 = 1 + (y - 1) / 4
    return math.sin(math.pi * w1) ** 2 + (w1 - 1) ** 2 * (1 + 10 * math.sin(math.pi * w1 + 1) ** 2) + (w2 - 1) ** 2 * (1 + math.sin(2 * math.pi * w2) ** 2)


OBJECTIVES: list[Objective] = [
    ("forrester", (0.0,), (1.0,), _forrester),
    ("six_hump_camel", (-3.0, -2.0), (3.0, 2.0), _six_hump_camel),
    ("goldstein_price", (-2.0, -2.0), (2.0, 2.0), _goldstein_price),
    ("levy", (-10.0, -10.0), (10.0, 10.0), _levy),
]


def _sample_point(bounds_lo: tuple[float, ...], bounds_hi: tuple[float, ...], rng: random.Random) -> tuple[float, ...]:
    return tuple(rng.uniform(lo, hi) for lo, hi in zip(bounds_lo, bounds_hi))


def _center_ood(point: tuple[float, ...], bounds_lo: tuple[float, ...], bounds_hi: tuple[float, ...]) -> float:
    distances = []
    for value, lo, hi in zip(point, bounds_lo, bounds_hi):
        center = 0.5 * (lo + hi)
        radius = max(1e-12, 0.5 * (hi - lo))
        distances.append(abs(value - center) / radius)
    return clamp(sum(distances) / len(distances))


def _format_point(point: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{value:.3f}" for value in point) + ")"


def make_preferential_bo_records(n: int = 300, seed: int = 0) -> list[ActionRecord]:
    """Create pairwise 'A is better than B' action records.

    This mirrors preferential Bayesian optimization: the visible agent sees a
    noisy surrogate and proposes a pairwise choice; the hidden benchmark labels
    whether the chosen arm truly has lower objective value.
    """

    rng = random.Random(seed)
    records: list[ActionRecord] = []
    for idx in range(n):
        name, lo, hi, objective = rng.choice(OBJECTIVES)
        point_a = _sample_point(lo, hi, rng)
        point_b = _sample_point(lo, hi, rng)
        true_a = objective(point_a)
        true_b = objective(point_b)
        pair_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b))) / math.sqrt(len(point_a))
        ood = clamp(0.5 * _center_ood(point_a, lo, hi) + 0.5 * _center_ood(point_b, lo, hi))
        variance = clamp(0.12 + 0.26 * pair_distance + 0.45 * ood + rng.random() * 0.12)
        noise_scale = (0.10 + 0.70 * variance) * (1.0 + 0.08 * (abs(true_a) + abs(true_b)))
        pred_a = true_a + rng.gauss(0.0, noise_scale)
        pred_b = true_b + rng.gauss(0.0, noise_scale)
        chooses_a = pred_a <= pred_b
        chosen = "A" if chooses_a else "B"
        chosen_true = true_a if chooses_a else true_b
        other_true = true_b if chooses_a else true_a
        chosen_pred = pred_a if chooses_a else pred_b
        other_pred = pred_b if chooses_a else pred_a
        label = int(chosen_true <= other_true)
        true_margin = abs(true_a - true_b) / (1.0 + abs(true_a) + abs(true_b))
        predicted_margin = abs(pred_a - pred_b) / (1.0 + abs(pred_a) + abs(pred_b))
        reliability_signal = sigmoid(5.5 * predicted_margin - 2.0 * variance - 1.1 * ood)
        conflict = clamp(0.10 + 0.58 * variance + (0.22 if predicted_margin < 0.08 else 0.0) + rng.gauss(0.0, 0.06))
        cost = clamp(0.18 + 0.12 * len(point_a) + 0.20 * ood + rng.random() * 0.18)
        features = action_features(
            rng=rng,
            latent_quality=reliability_signal,
            novelty=0.5,
            cost=cost,
            ood=ood,
            variance=variance,
            conflict=conflict,
            lineage=clamp(true_margin + 0.3 * predicted_margin),
        )
        features["preference_margin"] = clamp(predicted_margin * 3.0)
        features["audited_ld"] = clamp(features["audited_ld"] + 0.14 * (1.0 - variance) + 0.10 * predicted_margin)
        utility = max(0.0, other_true - chosen_true) / (1.0 + abs(true_a) + abs(true_b))
        context = (
            f"Preferential BO task on {name}. Candidate A={_format_point(point_a)}, "
            f"candidate B={_format_point(point_b)}. Surrogate predicts y(A)={pred_a:.3f}, "
            f"y(B)={pred_b:.3f}; lower is better."
        )
        evidence = [
            f"surrogate_margin={predicted_margin:.3f}",
            f"posterior_variance={variance:.3f}",
            f"domain_edge_score={ood:.3f}",
        ]
        records.append(
            ActionRecord(
                record_id=f"preferential_bo-{seed}-{idx:05d}",
                benchmark="preferential_bo",
                split=split_name(idx, n),
                visible_context=context,
                candidate_action=f"choose candidate {chosen} as better because predicted objective {chosen_pred:.3f} < {other_pred:.3f}",
                action_type="choose",
                evidence=evidence,
                features=features,
                label=label,
                utility=utility,
                metadata={
                    "objective": name,
                    "point_a": point_a,
                    "point_b": point_b,
                    "true_a": true_a,
                    "true_b": true_b,
                    "chosen": chosen,
                    "true_margin": true_margin,
                },
            )
        )
    return records
