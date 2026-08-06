from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

from harness_matsci.features import clamp


ELEMENT_GROUPS = {
    "alkali": ["Li", "Na", "K", "Rb", "Cs"],
    "alkaline_earth": ["Be", "Mg", "Ca", "Sr", "Ba"],
    "transition": ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "W"],
    "post_transition": ["Al", "Ga", "In", "Sn", "Pb", "Bi"],
    "metalloid": ["B", "Si", "Ge", "As", "Sb", "Te"],
    "nonmetal": ["C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I"],
    "rare_earth": ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Yb"],
}

ELEMENT_TO_GROUP = {element: group for group, elements in ELEMENT_GROUPS.items() for element in elements}


@dataclass(frozen=True)
class SyntheticMaterial:
    formula: str
    elements: tuple[str, ...]
    property_value: float
    uncertainty: float
    novelty: float


def split_name(index: int, n: int) -> str:
    if index < int(0.6 * n):
        return "train"
    if index < int(0.8 * n):
        return "val"
    return "test"


def noisy(value: float, scale: float, rng: random.Random) -> float:
    return value + rng.gauss(0.0, scale)


def random_composition(rng: random.Random, min_elements: int = 2, max_elements: int = 4) -> tuple[str, ...]:
    groups = rng.sample(list(ELEMENT_GROUPS), k=rng.randint(min_elements, max_elements))
    elements = [rng.choice(ELEMENT_GROUPS[group]) for group in groups]
    return tuple(sorted(elements))


def formula_from_elements(elements: tuple[str, ...], rng: random.Random) -> str:
    parts = []
    for element in elements:
        count = rng.choice([1, 1, 1, 2, 2, 3, 4])
        parts.append(element if count == 1 else f"{element}{count}")
    return "".join(parts)


def composition_signature(elements: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(ELEMENT_TO_GROUP.get(element, "unknown") for element in elements))


def novelty_from_seen(elements: tuple[str, ...], seen_signatures: Counter[tuple[str, ...]]) -> float:
    signature = composition_signature(elements)
    count = seen_signatures[signature]
    return clamp(1.0 / math.sqrt(count + 1.0))


def material_property(elements: tuple[str, ...], rng: random.Random) -> float:
    group_bonus = {
        "transition": 34.0,
        "rare_earth": 24.0,
        "metalloid": 18.0,
        "post_transition": 12.0,
        "alkaline_earth": 9.0,
        "nonmetal": -5.0,
        "alkali": -12.0,
    }
    value = 80.0
    for element in elements:
        value += group_bonus.get(ELEMENT_TO_GROUP.get(element, "unknown"), 0.0)
        value += (sum(ord(char) for char in element) % 13) * 1.7
    if any(ELEMENT_TO_GROUP.get(element) == "transition" for element in elements) and any(
        ELEMENT_TO_GROUP.get(element) == "metalloid" for element in elements
    ):
        value += 45.0
    if any(ELEMENT_TO_GROUP.get(element) == "rare_earth" for element in elements) and any(
        ELEMENT_TO_GROUP.get(element) == "nonmetal" for element in elements
    ):
        value += 25.0
    return noisy(value, 12.0, rng)


def action_features(
    *,
    rng: random.Random,
    latent_quality: float,
    novelty: float,
    cost: float,
    ood: float,
    variance: float,
    conflict: float,
    lineage: float,
) -> dict[str, float]:
    observed_quality = clamp(noisy(latent_quality, 0.09 + 0.07 * variance, rng))
    support = clamp(0.25 + 0.58 * observed_quality + 0.10 * novelty - 0.18 * conflict + noisy(0.0, 0.05, rng))
    verbal = clamp(0.34 + 0.52 * observed_quality - 0.16 * conflict + noisy(0.0, 0.08, rng))
    stability = clamp(0.80 - 0.55 * variance - 0.20 * ood + noisy(0.0, 0.05, rng))
    tool_agreement = clamp(0.30 + 0.55 * observed_quality - 0.24 * conflict - 0.20 * variance + noisy(0.0, 0.05, rng))
    source = clamp(0.86 - 0.35 * ood - 0.14 * conflict + noisy(0.0, 0.05, rng))
    disagreement = clamp(0.14 + 0.60 * variance + 0.22 * ood + 0.18 * conflict + noisy(0.0, 0.05, rng))
    preference_margin = clamp(abs(observed_quality - 0.5) * 1.65 + noisy(0.0, 0.04, rng))
    audited_ld = clamp(0.45 + 0.36 * observed_quality + 0.12 * support - 0.20 * conflict - 0.10 * ood + noisy(0.0, 0.05, rng))
    reversibility = clamp(0.92 - 0.68 * cost + 0.08 * novelty + noisy(0.0, 0.04, rng))
    return {
        "audited_ld": audited_ld,
        "preference_margin": preference_margin,
        "verbal_confidence": verbal,
        "perturbation_stability": stability,
        "evidence_support": support,
        "evidence_conflict": clamp(conflict),
        "source_reliability": source,
        "tool_agreement": tool_agreement,
        "model_disagreement": disagreement,
        "ood_score": clamp(ood),
        "experimental_variance": clamp(variance),
        "cost": clamp(cost),
        "reversibility": reversibility,
        "lineage_impact": clamp(lineage),
    }
