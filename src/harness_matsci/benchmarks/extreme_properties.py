from __future__ import annotations

import random

from harness_matsci.features import clamp
from harness_matsci.schema import ActionRecord

from .common import action_features, split_name


FRAGMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "phenyl", "amide", "ether", "pyridine", "sulfonyl", "morpholine"]


def _molecule_name(fragments: list[str]) -> str:
    return "-".join(fragments)


def _properties(fragments: list[str], rng: random.Random) -> dict[str, float]:
    counts = {fragment: fragments.count(fragment) for fragment in set(fragments)}
    heavy = len(fragments)
    halogens = counts.get("F", 0) + counts.get("Cl", 0) + counts.get("Br", 0)
    hetero = counts.get("N", 0) + counts.get("O", 0) + counts.get("S", 0) + counts.get("amide", 0) + counts.get("ether", 0)
    aromatic = counts.get("phenyl", 0) + counts.get("pyridine", 0)
    logp = 0.45 * heavy + 0.55 * halogens + 0.65 * aromatic - 0.35 * hetero + rng.gauss(0.0, 0.25)
    tpsa = 12.0 * hetero + 18.0 * counts.get("sulfonyl", 0) + 9.0 * counts.get("morpholine", 0) + rng.gauss(0.0, 4.0)
    qed = clamp(0.78 - 0.028 * abs(heavy - 7) - 0.035 * max(0, logp - 4.5) - 0.0025 * max(0, tpsa - 75) + rng.gauss(0.0, 0.04))
    drd2 = clamp(0.18 + 0.10 * aromatic + 0.08 * counts.get("N", 0) + 0.10 * counts.get("pyridine", 0) - 0.06 * counts.get("sulfonyl", 0) + rng.gauss(0.0, 0.08))
    return {"logP": logp, "TPSA": tpsa, "QED": qed, "DRD2": drd2, "MW_proxy": 42.0 * heavy + 18.0 * halogens + 35.0 * aromatic}


def _target_score(properties: dict[str, float], target: str) -> float:
    if target == "high_logP":
        return clamp((properties["logP"] - 2.0) / 4.0)
    if target == "low_TPSA":
        return clamp((80.0 - properties["TPSA"]) / 80.0)
    if target == "high_QED_DRD2":
        return clamp(0.55 * properties["QED"] + 0.45 * properties["DRD2"])
    if target == "high_TPSA":
        return clamp(properties["TPSA"] / 130.0)
    raise ValueError(f"unknown target: {target}")


def make_extreme_property_records(n: int = 300, seed: int = 0) -> list[ActionRecord]:
    """Create extreme-property discovery records over synthetic molecular edits."""

    rng = random.Random(seed)
    targets = ["high_logP", "low_TPSA", "high_QED_DRD2", "high_TPSA"]
    records: list[ActionRecord] = []
    for idx in range(n):
        base_len = rng.randint(3, 8)
        base = [rng.choice(FRAGMENTS[:9]) for _ in range(base_len)]
        edit = rng.choice(["add_halogen", "add_hetero", "add_aromatic", "replace_polar", "trim"])
        candidate = list(base)
        if edit == "add_halogen":
            candidate.append(rng.choice(["F", "Cl", "Br"]))
        elif edit == "add_hetero":
            candidate.append(rng.choice(["N", "O", "amide", "ether", "morpholine", "sulfonyl"]))
        elif edit == "add_aromatic":
            candidate.append(rng.choice(["phenyl", "pyridine"]))
        elif edit == "replace_polar" and candidate:
            candidate[rng.randrange(len(candidate))] = rng.choice(["phenyl", "Cl", "ether", "pyridine"])
        elif edit == "trim" and len(candidate) > 2:
            del candidate[rng.randrange(len(candidate))]

        target = rng.choice(targets)
        base_props = _properties(base, rng)
        candidate_props = _properties(candidate, rng)
        base_score = _target_score(base_props, target)
        candidate_score = _target_score(candidate_props, target)
        improvement = candidate_score - base_score
        ood = clamp(0.08 + 0.055 * len(candidate) + 0.12 * len(set(candidate)) / max(1, len(candidate)) + (0.20 if "Br" in candidate or "sulfonyl" in candidate else 0.0))
        variance = clamp(0.10 + 0.30 * ood + 0.22 * abs(len(candidate) - len(base)) + rng.random() * 0.12)
        cost = clamp(0.18 + 0.055 * len(candidate) + 0.16 * ("sulfonyl" in candidate or "Br" in candidate) + rng.random() * 0.16)
        conflict = clamp(0.12 + 0.24 * variance + 0.20 * (candidate_props["QED"] < 0.35) + rng.gauss(0.0, 0.06))
        label = int(candidate_score >= 0.72 and improvement > 0.025 and candidate_props["QED"] >= 0.28)
        features = action_features(
            rng=rng,
            latent_quality=clamp(0.52 + improvement + 0.35 * candidate_score),
            novelty=ood,
            cost=cost,
            ood=ood,
            variance=variance,
            conflict=conflict,
            lineage=clamp(max(0.0, improvement) + 0.3 * candidate_score),
        )
        features["preference_margin"] = clamp(abs(improvement) * 2.5)
        utility = max(0.0, improvement) * max(0.0, candidate_score)
        context = (
            f"Extreme-property task {target}. Base molecule {_molecule_name(base)} has score {base_score:.3f}. "
            f"Proposed edit {edit} yields candidate {_molecule_name(candidate)} with predicted score {candidate_score:.3f}."
        )
        evidence = [
            f"target={target}",
            f"predicted_improvement={improvement:.3f}",
            f"QED={candidate_props['QED']:.3f}",
            f"OOD={ood:.3f}",
        ]
        records.append(
            ActionRecord(
                record_id=f"extreme_properties-{seed}-{idx:05d}",
                benchmark="extreme_properties",
                split=split_name(idx, n),
                visible_context=context,
                candidate_action=f"apply edit {edit} to pursue {target} extreme-property candidate",
                action_type="recommend",
                evidence=evidence,
                features=features,
                label=label,
                utility=utility,
                metadata={
                    "target": target,
                    "base": base,
                    "candidate": candidate,
                    "base_properties": base_props,
                    "candidate_properties": candidate_props,
                    "base_score": base_score,
                    "candidate_score": candidate_score,
                    "edit": edit,
                },
            )
        )
    return records
