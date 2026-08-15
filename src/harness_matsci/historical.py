from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .features import clamp, text_confidence
from .schema import ActionRecord

HISTORICAL_TASK_FILES = {
    "matbench_pairwise": "matbench_pairwise_actions.jsonl",
    "preferential_bo": "pairwise_optimization_actions.jsonl",
    "discover_unique": "unique_materials_actions.jsonl",
    "extreme_properties": "extreme_properties_actions.jsonl",
}

MAIN_MATERIAL_TASKS = ("matbench_pairwise", "discover_unique", "extreme_properties")

ORACLE_SIGNAL_KEYS = frozenset(
    {
        "preference_strength",
        "latent_utility",
        "metric_value",
        "performance_score",
        "uniqueness_score",
        "discovery_score",
        "target_hit_score",
        "hit_fraction",
        "evidence_support",
        "evidence_conflict",
        "verbal_confidence",
        "tool_agreement",
        "consensus_spread",
        "ood_score",
        "reward",
        "reward_score",
        "mean_clipped_error",
        "five_hit",
    }
)

ORACLE_CONTEXT_KEYS = frozenset(
    {
        "performance_score",
        "uniqueness_score",
        "hit_fraction",
        "reward",
        "reward_score",
        "mean_clipped_error",
        "five_hit",
    }
)


def load_historical_task_records(data_dir: str | Path, task: str) -> list[ActionRecord]:
    try:
        filename = HISTORICAL_TASK_FILES[task]
    except KeyError as exc:
        raise ValueError(f"unknown historical task {task!r}") from exc
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(path)
    payloads: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError("record must be a JSON object")
                payloads.append(payload)
            except Exception as exc:
                raise ValueError(f"invalid historical record at {path}:{line_number}: {exc}") from exc
    if not payloads:
        raise ValueError(f"historical task file is empty: {path}")
    if task == "preferential_bo":
        return _convert_preferential_duels(payloads)
    return [_convert_historical_record(payload, task) for payload in payloads]


def load_historical_tasks(data_dir: str | Path, tasks: tuple[str, ...]) -> dict[str, list[ActionRecord]]:
    return {task: load_historical_task_records(data_dir, task) for task in tasks}


def grouped_four_way_split(
    records: list[ActionRecord],
    *,
    seed: int = 7,
    train_fraction: float = 0.6,
    feedback_fraction: float = 0.15,
    acceptance_fraction: float = 0.1,
    min_test_group_fraction: float = 0.05,
) -> dict[str, list[ActionRecord]]:
    """Create a regime-held-out test set and independent source partitions.

    The scientific regime boundary is ``group_id``.  We select a deterministic
    subset of complete groups for test, then split only the remaining source
    records into train/feedback/acceptance.  This avoids the invalid situation
    where a four-group benchmark puts one whole group into training and leaves
    almost no data for fitting, while still preventing group leakage into the
    final test set.  The independent source partitions are non-overlapping but
    can contain records from the same source regimes; this policy is explicit
    in the returned metadata and should be complemented by a group-robustness
    ablation in a final paper.
    """
    if not records:
        raise ValueError("records cannot be empty")
    if train_fraction <= 0 or feedback_fraction <= 0 or acceptance_fraction <= 0:
        raise ValueError("all non-test fractions must be positive")
    if train_fraction + feedback_fraction + acceptance_fraction >= 1:
        raise ValueError("fractions must leave positive test mass")
    if not 0 < min_test_group_fraction < 1:
        raise ValueError("min_test_group_fraction must be in (0, 1)")
    groups: dict[str, list[ActionRecord]] = defaultdict(list)
    for record in records:
        group_id = str(record.metadata.get("group_id", record.benchmark))
        groups[group_id].append(record)
    group_ids = sorted(groups, key=lambda group_id: _stable_hash(group_id, seed))
    total_records = len(records)
    target_test_records = total_records * (1.0 - train_fraction - feedback_fraction - acceptance_fraction)
    test_group_ids = _select_test_groups(
        groups,
        group_ids,
        target_test_records,
        min_group_records=max(2, math.ceil(total_records * min_test_group_fraction)),
    )
    source_group_ids = [group_id for group_id in group_ids if group_id not in test_group_ids]
    source_records = [record for group_id in source_group_ids for record in groups[group_id]]
    source_total_fraction = train_fraction + feedback_fraction + acceptance_fraction
    source_ratios = {
        "train": train_fraction / source_total_fraction,
        "feedback": feedback_fraction / source_total_fraction,
        "acceptance": acceptance_fraction / source_total_fraction,
    }
    output: dict[str, list[ActionRecord]] = {"train": [], "feedback": [], "acceptance": [], "test": []}
    for record in sorted(source_records, key=lambda item: _stable_hash(item.record_id, seed)):
        rank = _stable_hash(f"record|{record.record_id}", seed) / float(2**64)
        if rank < source_ratios["train"]:
            split = "train"
        elif rank < source_ratios["train"] + source_ratios["feedback"]:
            split = "feedback"
        else:
            split = "acceptance"
        output[split].append(_assign_split(record, split))
    output["test"] = [_assign_split(record, "test") for group_id in test_group_ids for record in groups[group_id]]
    if any(not output[name] for name in output):
        raise ValueError(f"four-way split produced an empty partition: { {name: len(rows) for name, rows in output.items()} }")
    return output


def random_four_way_split(
    records: list[ActionRecord],
    *,
    seed: int = 7,
    train_fraction: float = 0.6,
    feedback_fraction: float = 0.15,
    acceptance_fraction: float = 0.1,
) -> dict[str, list[ActionRecord]]:
    """Create disjoint random train, feedback, acceptance, and test splits.

    Synthetic records do not have scientifically meaningful groups, so a
    seeded record-level split is appropriate.  Unlike the legacy train/val
    helper, feedback and acceptance are separate partitions because feedback
    is consumed by the proposer while acceptance is used for model selection.
    """
    if not records:
        raise ValueError("records cannot be empty")
    fractions = (train_fraction, feedback_fraction, acceptance_fraction)
    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("all non-test fractions must be positive")
    if sum(fractions) >= 1:
        raise ValueError("fractions must leave positive test mass")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_records = len(shuffled)
    n_train = int(n_records * train_fraction)
    n_feedback = int(n_records * feedback_fraction)
    n_acceptance = int(n_records * acceptance_fraction)
    boundaries = (
        n_train,
        n_train + n_feedback,
        n_train + n_feedback + n_acceptance,
    )
    output = {
        "train": shuffled[: boundaries[0]],
        "feedback": shuffled[boundaries[0] : boundaries[1]],
        "acceptance": shuffled[boundaries[1] : boundaries[2]],
        "test": shuffled[boundaries[2] :],
    }
    if any(not output[name] for name in output):
        raise ValueError(f"four-way split produced an empty partition: { {name: len(rows) for name, rows in output.items()} }")
    return {name: [_assign_split(record, name) for record in rows] for name, rows in output.items()}


def _select_test_groups(
    groups: dict[str, list[ActionRecord]],
    ordered_group_ids: list[str],
    target_records: float,
    *,
    min_group_records: int = 1,
) -> list[str]:
    """Select a small group subset close to the target test mass.

    Exhaustive subset search is safe for the current benchmark sizes (at most
    ten groups) and lets us prefer a test slice containing both labels.
    """
    candidates: list[tuple[tuple[float, int, int], list[str]]] = []
    eligible_group_ids = [
        group_id for group_id in ordered_group_ids if len(groups[group_id]) >= min_group_records
    ]
    if not eligible_group_ids:
        eligible_group_ids = list(ordered_group_ids)
    for width in range(1, len(eligible_group_ids)):
        for subset in itertools.combinations(eligible_group_ids, width):
            subset_records = [record for group_id in subset for record in groups[group_id]]
            labels = {record.label for record in subset_records}
            if len(labels) < 2:
                label_penalty = 1
            else:
                label_penalty = 0
            size_error = abs(len(subset_records) - target_records) / max(1.0, target_records)
            tie_break = sum(ordered_group_ids.index(group_id) for group_id in subset)
            candidates.append(((label_penalty, size_error, width + tie_break / 10000.0), list(subset)))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def split_historical_tasks(
    records_by_task: dict[str, list[ActionRecord]],
    *,
    seed: int = 7,
    train_fraction: float = 0.6,
    feedback_fraction: float = 0.15,
    acceptance_fraction: float = 0.1,
) -> dict[str, dict[str, list[ActionRecord]]]:
    return {
        task: grouped_four_way_split(
            records,
            seed=seed,
            train_fraction=train_fraction,
            feedback_fraction=feedback_fraction,
            acceptance_fraction=acceptance_fraction,
        )
        for task, records in records_by_task.items()
    }


def _convert_historical_record(payload: dict[str, Any], task: str) -> ActionRecord:
    raw_uncertainty = payload.get("uncertainty_signals", {})
    raw_context = payload.get("context_features", {})
    excluded_features = sorted(
        {
            str(key)
            for key, value in raw_uncertainty.items()
            if _number(value) and str(key) in ORACLE_SIGNAL_KEYS
        }
        | {
            f"context_{key}"
            for key, value in raw_context.items()
            if _number(value) and str(key) in ORACLE_CONTEXT_KEYS
        }
    )
    uncertainty = {
        str(key): clamp(float(value))
        for key, value in raw_uncertainty.items()
        if _number(value) and str(key) not in ORACLE_SIGNAL_KEYS
    }
    features = dict(uncertainty)
    visible_context = _sanitize_visible_context(task, str(payload.get("visible_context", "")))
    candidate_action = str(payload.get("candidate_action", ""))
    text_probability, text_support, text_conflict = text_confidence(f"{visible_context} {candidate_action}")
    features["verbal_confidence"] = text_probability
    features["evidence_support"] = text_support
    features["evidence_conflict"] = text_conflict
    features["cost"] = _cost_score(payload.get("cost_level", "medium"))
    features["reversibility"] = _reversibility_score(payload.get("reversibility", "medium"))
    features["action_complexity"] = clamp(len(candidate_action) / 300.0)
    features["evidence_count"] = clamp(len(payload.get("evidence", [])) / 8.0)
    if "source_risk" in features:
        features["source_reliability"] = 1.0 - features["source_risk"]
    for key, value in raw_context.items():
        if _number(value) and str(key) not in ORACLE_CONTEXT_KEYS:
            features[f"context_{key}"] = _normalize_context_feature(str(key), float(value))
    features.update(_visible_candidate_features(task, visible_context, candidate_action, raw_context))
    outcome = payload.get("outcome_success")
    if outcome is None:
        raise ValueError("historical record lacks outcome_success")
    source = payload.get("source", {})
    tool_outputs = payload.get("tool_outputs", {})
    utility = tool_outputs.get("utility", payload.get("metric_value", 0.0)) if isinstance(tool_outputs, dict) else payload.get("metric_value", 0.0)
    metadata = {
        "group_id": str(payload.get("group_id", payload.get("trace_id", "ungrouped"))),
        "task": str(payload.get("task", task)),
        "domain": str(payload.get("domain", "materials_discovery")),
        "source": source,
        "label_source": payload.get("label_source", "historical_benchmark_proxy"),
        "trace_id": payload.get("trace_id", ""),
        "raw_record_id": payload.get("record_id", ""),
        "metric_direction": payload.get("metric_direction", "maximize"),
        "outcome_success": bool(outcome),
        "raw_uncertainty_keys": sorted(str(key) for key in raw_uncertainty if _number(raw_uncertainty[key])),
        "raw_context_keys": sorted(str(key) for key in raw_context if _number(raw_context[key])),
        "excluded_oracle_features": excluded_features,
    }
    evidence = _sanitize_evidence(task, payload.get("evidence", []))
    return ActionRecord(
        record_id=str(payload.get("record_id", "")),
        benchmark=task,
        split="unspecified",
        visible_context=visible_context,
        candidate_action=candidate_action,
        action_type=_safe_action_type(str(payload.get("action_type", "recommend"))),
        evidence=evidence,
        features=features,
        label=int(bool(outcome)),
        utility=float(utility),
        metadata=metadata,
    )


def _convert_preferential_duels(payloads: list[dict[str, Any]]) -> list[ActionRecord]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        group_id = str(payload.get("group_id", payload.get("trace_id", "ungrouped")))
        groups[group_id].append(payload)
    records: list[ActionRecord] = []
    for group_id, candidates in sorted(groups.items()):
        ordered = sorted(candidates, key=lambda payload: str(payload.get("record_id", "")))
        if len(ordered) < 2:
            raise ValueError(f"preferential group {group_id!r} needs at least two candidates")
        for index, left in enumerate(ordered):
            offset = 1 + _stable_hash(str(left.get("record_id", index)), 17) % (len(ordered) - 1)
            right = ordered[(index + offset) % len(ordered)]
            records.append(_make_preferential_duel(group_id, index, left, right))
    return records


def _make_preferential_duel(
    group_id: str,
    index: int,
    left: dict[str, Any],
    right: dict[str, Any],
) -> ActionRecord:
    left_utility = float(left.get("tool_outputs", {}).get("utility", left.get("metric_value", 0.0)))
    right_utility = float(right.get("tool_outputs", {}).get("utility", right.get("metric_value", 0.0)))
    left_ood = float(left.get("context_features", {}).get("domain_center_distance", 0.5))
    right_ood = float(right.get("context_features", {}).get("domain_center_distance", 0.5))
    average_ood = clamp((left_ood + right_ood) / 2.0)
    noise_scale = 0.06 + 0.18 * average_ood
    pair_id = f"{left.get('record_id')}|{right.get('record_id')}"
    rng = random.Random(_stable_hash(pair_id, 29))
    left_observed = _visible_surrogate(left, rng, noise_scale)
    right_observed = _visible_surrogate(right, rng, noise_scale)
    choose_left = left_observed >= right_observed
    chosen_name = "A" if choose_left else "B"
    chosen_true = left_utility if choose_left else right_utility
    other_true = right_utility if choose_left else left_utility
    label = int(chosen_true >= other_true)
    observed_margin = clamp(abs(left_observed - right_observed))
    true_margin = clamp(abs(left_utility - right_utility))
    left_action = str(left.get("candidate_action", "candidate A"))
    right_action = str(right.get("candidate_action", "candidate B"))
    left_coordinates = left_action.split(":", 1)[-1].strip()
    right_coordinates = right_action.split(":", 1)[-1].strip()
    dimension = max(
        float(left.get("context_features", {}).get("dimension", 1.0)),
        float(right.get("context_features", {}).get("dimension", 1.0)),
    )
    reliability = clamp(0.5 + 0.5 * observed_margin - 0.35 * noise_scale)
    features = {
        "preference_margin": observed_margin,
        "verbal_confidence": reliability,
        "evidence_support": observed_margin,
        "evidence_conflict": clamp(noise_scale + (1.0 - observed_margin) * 0.25),
        "perturbation_stability": clamp(1.0 - noise_scale),
        "tool_agreement": clamp(1.0 - 2.0 * noise_scale),
        "model_disagreement": clamp(noise_scale),
        "ood_score": average_ood,
        "cost": 0.2,
        "reversibility": 0.8,
        "action_complexity": clamp((len(left_action) + len(right_action)) / 300.0),
        "evidence_count": 0.375,
        "candidate_structure_complexity": clamp(dimension / 4.0),
        "candidate_composition_diversity": clamp(abs(left_ood - right_ood)),
        "candidate_domain_position": average_ood,
    }
    source = dict(left.get("source", {}))
    source["paired_record_id"] = right.get("record_id", "")
    return ActionRecord(
        record_id=f"pairwise-duel::{group_id.split('::')[-1]}::{index:05d}",
        benchmark="preferential_bo",
        split="unspecified",
        visible_context=(
            f"Preferential optimization duel on {group_id.split('::')[-1]}. "
            f"Candidate A: {left_coordinates}; candidate B: {right_coordinates}. "
            "The agent observes only a noisy surrogate ordering, not the latent objective."
        ),
        candidate_action=f"Choose candidate {chosen_name} for the next preferential comparison.",
        action_type="choose",
        evidence=[
            f"surrogate_margin={observed_margin:.4f}",
            f"estimated_noise={noise_scale:.4f}",
            f"domain_edge_score={average_ood:.4f}",
        ],
        features=features,
        label=label,
        utility=true_margin if label else 0.0,
        metadata={
            "group_id": group_id,
            "task": "pairwise_preference_optimization",
            "domain": "optimization_benchmark",
            "source": source,
            "label_source": "simulated_noisy_preferential_duel_from_published_objective",
            "left_record_id": left.get("record_id", ""),
            "right_record_id": right.get("record_id", ""),
            "chosen": chosen_name,
            "left_true_utility": left_utility,
            "right_true_utility": right_utility,
            "chosen_true_utility": chosen_true,
            "true_margin": true_margin,
            "excluded_oracle_features": sorted(
                set(left.get("uncertainty_signals", {})) & ORACLE_SIGNAL_KEYS
            ),
            "raw_uncertainty_keys": sorted(
                set(left.get("uncertainty_signals", {})) | set(right.get("uncertainty_signals", {}))
            ),
            "raw_context_keys": sorted(
                set(left.get("context_features", {})) | set(right.get("context_features", {}))
            ),
        },
    )


def _assign_split(record: ActionRecord, split: str) -> ActionRecord:
    payload = record.to_json()
    payload["split"] = split
    return ActionRecord.from_json(payload)


def _safe_action_type(action_type: str) -> str:
    if action_type in {"choose", "choose_candidate", "recommend", "recommend_experiment", "execute_tool", "experiment", "retrieve_more", "ask_more", "summarize_literature", "abstain"}:
        return action_type
    return "recommend"


def _cost_score(value: Any) -> float:
    return {"low": 0.2, "medium": 0.5, "high": 0.8}.get(str(value).lower(), 0.5)


def _reversibility_score(value: Any) -> float:
    return {"low": 0.2, "medium": 0.5, "high": 0.8}.get(str(value).lower(), 0.5)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _sanitize_visible_context(task: str, context: str) -> str:
    """Remove benchmark outcomes that were serialized into historical context."""
    if task in {"discover_unique", "matbench_pairwise"}:
        context = re.sub(r"\s*log10\(K_VRH\)\s*=\s*[-+]?\d+(?:\.\d+)?", "", context)
        context = re.sub(r"\s*log10_k_vrh\s*=\s*[-+]?\d+(?:\.\d+)?", "", context, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", context).strip()


def _sanitize_evidence(task: str, evidence: Any) -> list[str]:
    cleaned = [str(item) for item in evidence]
    if task in {"discover_unique", "matbench_pairwise"}:
        cleaned = [
            re.sub(r"log10\(K_VRH\)\s*=\s*[-+]?\d+(?:\.\d+)?", "property estimate", item)
            for item in cleaned
        ]
        cleaned = [
            re.sub(r"log10_k_vrh\s*=\s*[-+]?\d+(?:\.\d+)?", "property estimate", item, flags=re.IGNORECASE)
            for item in cleaned
        ]
    if task == "extreme_properties":
        cleaned = [
            re.sub(r"(?:hit_fraction|reward|all_hit)\s*=\s*[^,;\s]+", "outcome withheld", item)
            for item in cleaned
        ]
    return cleaned


def _visible_surrogate(payload: dict[str, Any], rng: random.Random, noise_scale: float) -> float:
    """Construct a reproducible prior score without reading oracle outcomes."""
    action = str(payload.get("candidate_action", ""))
    context = str(payload.get("visible_context", ""))
    coordinates = [float(value) for value in re.findall(r"x\d+=(-?\d+(?:\.\d+)?)", action)]
    smooth_prior = sum(math.sin(value) + 0.25 * math.cos(2.0 * value) for value in coordinates)
    digest = hashlib.sha256(f"{action}|{context}".encode()).hexdigest()
    hashed_prior = int(digest[:12], 16) / float(16**12 - 1)
    return 0.5 + 0.15 * math.tanh(smooth_prior) + 0.1 * (hashed_prior - 0.5) + rng.gauss(0.0, noise_scale)


def _normalize_context_feature(name: str, value: float) -> float:
    if name == "dimension":
        return clamp(value / 4.0)
    if name == "space_group":
        return clamp(value / 230.0)
    return clamp(value)


def _visible_candidate_features(
    task: str,
    visible_context: str,
    candidate_action: str,
    raw_context: dict[str, Any],
) -> dict[str, float]:
    if task == "matbench_pairwise":
        formulas = re.findall(r"Composition:\s*([^;]+)", visible_context)
        chosen_match = re.search(r"candidate\s+([AB])", candidate_action, flags=re.IGNORECASE)
        chosen_index = 0 if not chosen_match or chosen_match.group(1).upper() == "A" else 1
        chosen_formula = formulas[chosen_index] if len(formulas) > chosen_index else candidate_action
        elements = re.findall(r"[A-Z][a-z]?", chosen_formula)
        space_groups = [float(value) for value in re.findall(r"space group:\s*(\d+)", visible_context, flags=re.IGNORECASE)]
        chosen_space_group = space_groups[chosen_index] if len(space_groups) > chosen_index else 0.0
        same_system = float(raw_context.get("same_crystal_system", 0.0))
        surrogate_margin = float(raw_context.get("surrogate_margin", 0.0))
        uncertainty = float(raw_context.get("surrogate_uncertainty", 0.5))
        return {
            "candidate_structure_complexity": clamp(len(chosen_formula) / 40.0),
            "candidate_composition_diversity": clamp(len(set(elements)) / 6.0),
            "candidate_domain_position": clamp(chosen_space_group / 230.0),
            "pair_same_crystal_system": clamp(same_system),
            "pair_surrogate_margin": clamp(surrogate_margin),
            "pair_surrogate_uncertainty": clamp(uncertainty),
            "ood_score": clamp(0.2 + 0.25 * (1.0 - same_system) + 0.5 * uncertainty),
        }
    if task == "preferential_bo":
        coordinates = [float(value) for value in re.findall(r"x\d+=(-?\d+(?:\.\d+)?)", candidate_action)]
        dimension = float(raw_context.get("dimension", len(coordinates) or 1.0))
        domain_position = float(raw_context.get("domain_center_distance", 0.5))
        coordinate_spread = max(coordinates) - min(coordinates) if len(coordinates) > 1 else abs(coordinates[0]) if coordinates else 0.0
        return {
            "candidate_structure_complexity": clamp(dimension / 4.0),
            "candidate_composition_diversity": clamp(coordinate_spread / 20.0),
            "candidate_domain_position": clamp(domain_position),
            "ood_score": clamp(domain_position),
        }
    if task == "discover_unique":
        match = re.search(r"Composition:\s*([^;]+)", visible_context)
        formula = match.group(1).strip() if match else candidate_action
        elements = re.findall(r"[A-Z][a-z]?", formula)
        coefficients = [float(value) for value in re.findall(r"[A-Z][a-z]?(\d+(?:\.\d+)?)?", formula) if value]
        crystal_match = re.search(r"crystal system:\s*([^;\.]+)", visible_context, flags=re.IGNORECASE)
        crystal_system = crystal_match.group(1).strip().lower() if crystal_match else "unknown"
        space_group = float(raw_context.get("space_group", 0.0))
        features = {
            "candidate_structure_complexity": clamp(len(formula) / 40.0),
            "candidate_composition_diversity": clamp(len(set(elements)) / 6.0),
            "candidate_domain_position": clamp(space_group / 230.0),
            "composition_stoichiometry": clamp(sum(coefficients or [float(len(elements))]) / 16.0),
            "ood_score": clamp(0.15 + 0.15 * max(0, len(set(elements)) - 1) + len(formula) / 200.0),
        }
        for system in ("cubic", "hexagonal", "monoclinic", "orthorhombic", "tetragonal", "triclinic", "trigonal"):
            features[f"crystal_{system}"] = 1.0 if crystal_system == system else 0.0
        return features
    smiles = candidate_action.split(":", 1)[-1].strip()
    atom_tokens = re.findall(r"Br|Cl|[A-Z][a-z]?|[cnosp]", smiles)
    hetero_atoms = sum(token not in {"C", "c", "H"} for token in atom_tokens)
    ring_markers = sum(character.isdigit() for character in smiles)
    atom_count = max(1, len(atom_tokens))
    return {
        "candidate_structure_complexity": clamp(len(smiles) / 300.0),
        "candidate_composition_diversity": clamp(hetero_atoms / atom_count),
        "candidate_domain_position": clamp((ring_markers + smiles.count("(")) / 30.0),
        "smiles_atom_count": clamp(atom_count / 100.0),
        "smiles_ring_density": clamp(ring_markers / atom_count),
        "smiles_branch_density": clamp(smiles.count("(") / atom_count),
        "ood_score": clamp(0.2 + len(smiles) / 600.0 + hetero_atoms / (2.0 * atom_count)),
    }


def _stable_hash(group_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{group_id}".encode()).hexdigest()
    return int(digest[:16], 16)
