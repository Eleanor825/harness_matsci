from __future__ import annotations

import random
from collections import Counter

from harness_matsci.features import clamp
from harness_matsci.schema import ActionRecord

from .common import action_features, composition_signature, formula_from_elements, material_property, novelty_from_seen, random_composition, split_name


def make_discover_unique_records(n: int = 300, seed: int = 0) -> list[ActionRecord]:
    """Create DiSCoVeR-style records for high-value and chemically unique materials.

    The synthetic generator stands in for Matbench/Materials Project tables when
    licenses or API keys are unavailable.  A proposed follow-up is positive only
    if it is both high-performing and not a duplicate of a common chemistry.
    """

    rng = random.Random(seed)
    raw = []
    seen_signatures: Counter[tuple[str, ...]] = Counter()
    for _ in range(n):
        elements = random_composition(rng)
        formula = formula_from_elements(elements, rng)
        novelty = novelty_from_seen(elements, seen_signatures)
        seen_signatures.update([composition_signature(elements)])
        property_value = material_property(elements, rng)
        raw.append((formula, elements, property_value, novelty))

    values = sorted(value for _, _, value, _ in raw)
    high_cutoff = values[int(0.72 * (len(values) - 1))] if values else 0.0
    records: list[ActionRecord] = []
    for idx, (formula, elements, property_value, novelty) in enumerate(raw):
        normalized_property = clamp((property_value - values[0]) / max(1e-12, values[-1] - values[0])) if values else 0.0
        ood = clamp(0.12 + 0.72 * novelty + rng.gauss(0.0, 0.08))
        variance = clamp(0.12 + 0.30 * novelty + 0.10 * len(elements) + rng.random() * 0.14)
        cost = clamp(0.24 + 0.12 * len(elements) + 0.18 * novelty + rng.random() * 0.20)
        conflict = clamp(0.12 + 0.25 * variance + 0.22 * (1.0 - normalized_property) + rng.gauss(0.0, 0.06))
        discover_score = 0.64 * normalized_property + 0.36 * novelty
        label = int(property_value >= high_cutoff and novelty >= 0.45)
        features = action_features(
            rng=rng,
            latent_quality=discover_score,
            novelty=novelty,
            cost=cost,
            ood=ood,
            variance=variance,
            conflict=conflict,
            lineage=clamp(novelty * normalized_property),
        )
        features["preference_margin"] = clamp(abs(normalized_property - 0.72) + 0.35 * novelty)
        utility = max(0.0, property_value - high_cutoff) / max(1.0, values[-1] - values[0]) * (0.5 + 0.5 * novelty)
        context = (
            f"DiSCoVeR-style screening. Candidate {formula} has predicted normalized property "
            f"{normalized_property:.3f}, chemistry novelty {novelty:.3f}, and uncertainty {variance:.3f}."
        )
        evidence = [
            f"predicted_property={normalized_property:.3f}",
            f"novelty_score={novelty:.3f}",
            f"synthesis_cost={cost:.3f}",
        ]
        records.append(
            ActionRecord(
                record_id=f"discover_unique-{seed}-{idx:05d}",
                benchmark="discover_unique",
                split=split_name(idx, n),
                visible_context=context,
                candidate_action=f"recommend {formula} for follow-up because it may combine high property and unique chemistry",
                action_type="recommend",
                evidence=evidence,
                features=features,
                label=label,
                utility=utility,
                metadata={
                    "formula": formula,
                    "elements": elements,
                    "property_value": property_value,
                    "novelty": novelty,
                    "high_cutoff": high_cutoff,
                },
            )
        )
    return records
