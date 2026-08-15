from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import math
import random
import urllib.request
from bisect import bisect_left
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .features import clamp


MATBENCH_LOG_KVRH_URL = "https://raw.githubusercontent.com/materialsproject/matbench/main/scripts/artifacts/matbench_log_kvrh.json.bz2"
MATBENCH_PAIRWISE_FILENAME = "matbench_pairwise_actions.jsonl"


@dataclass(frozen=True)
class MatbenchPairwiseConfig:
    out: str
    source_path: str | None = None
    cache_dir: str | None = None
    n_pairs: int = 8000
    seed: int = 1729
    same_system_fraction: float = 0.6
    close_pair_fraction: float = 0.75
    surrogate_noise_floor: float = 0.06
    surrogate_noise_scale: float = 0.30
    min_true_gap: float = 0.003
    update_summary: bool = False


def load_matbench_log_kvrh_rows(source_path: str | Path | None = None, cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    if source_path:
        payload = Path(source_path).read_bytes()
    else:
        payload = _download_with_cache(MATBENCH_LOG_KVRH_URL, cache_dir=cache_dir, filename="matbench_log_kvrh.json.bz2")
    data = json.loads(bz2.decompress(payload).decode("utf-8"))
    indices = sorted((int(key) for key in data["mbid"]), key=int)
    rows: list[dict[str, Any]] = []
    for index in indices:
        key = str(index)
        rows.append(
            {
                "mbid": str(data["mbid"][key]),
                "composition": str(data["composition"][key]),
                "log10_k_vrh": float(data["log10(K_VRH)"][key]),
                "spg_num": int(data["spg_num"][key]),
                "crys_sys": str(data["crys_sys"][key]).lower(),
            }
        )
    return rows


def build_matbench_pairwise_payloads(
    rows: list[dict[str, Any]],
    *,
    n_pairs: int = 8000,
    seed: int = 1729,
    same_system_fraction: float = 0.6,
    close_pair_fraction: float = 0.75,
    surrogate_noise_floor: float = 0.06,
    surrogate_noise_scale: float = 0.30,
    min_true_gap: float = 0.003,
) -> list[dict[str, Any]]:
    if len(rows) < 2:
        raise ValueError("Matbench pairwise construction needs at least two material rows")
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    rng = random.Random(seed)
    values = [float(row["log10_k_vrh"]) for row in rows]
    normalized = _normalize(values)
    indexed_rows = [dict(row, normalized_k_vrh=score) for row, score in zip(rows, normalized)]
    all_sorted = sorted(range(len(indexed_rows)), key=lambda index: indexed_rows[index]["normalized_k_vrh"])
    all_scores = [indexed_rows[index]["normalized_k_vrh"] for index in all_sorted]
    by_system: dict[str, list[int]] = {}
    for index, row in enumerate(indexed_rows):
        by_system.setdefault(str(row["crys_sys"]), []).append(index)
    sorted_by_system = {
        system: sorted(indices, key=lambda index: indexed_rows[index]["normalized_k_vrh"])
        for system, indices in by_system.items()
        if len(indices) >= 2
    }
    system_names = sorted(sorted_by_system)

    payloads: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int, int]] = set()
    attempts = 0
    max_attempts = max(1000, n_pairs * 20)
    while len(payloads) < n_pairs and attempts < max_attempts:
        attempts += 1
        left_index = rng.randrange(len(indexed_rows))
        left_row = indexed_rows[left_index]
        use_same_system = rng.random() < same_system_fraction and str(left_row["crys_sys"]) in sorted_by_system
        if use_same_system:
            pool_indices = sorted_by_system[str(left_row["crys_sys"])]
            pool_scores = [indexed_rows[index]["normalized_k_vrh"] for index in pool_indices]
        else:
            pool_indices = all_sorted
            pool_scores = all_scores
        right_index = _sample_partner_index(
            left_index,
            pool_indices,
            pool_scores,
            indexed_rows[left_index]["normalized_k_vrh"],
            rng,
            close_pair_fraction=close_pair_fraction,
        )
        if right_index == left_index:
            continue
        if abs(float(indexed_rows[left_index]["normalized_k_vrh"]) - float(indexed_rows[right_index]["normalized_k_vrh"])) < min_true_gap:
            continue
        pair_key = (min(left_index, right_index), max(left_index, right_index), len(payloads) % 2)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        payloads.append(
            _make_pairwise_payload(
                left_row,
                indexed_rows[right_index],
                event_seq=len(payloads),
                rng=rng,
                seed=seed,
                surrogate_noise_floor=surrogate_noise_floor,
                surrogate_noise_scale=surrogate_noise_scale,
            )
        )
    if len(payloads) < n_pairs:
        raise RuntimeError(f"only constructed {len(payloads)} pairwise records after {attempts} attempts")
    return payloads


def write_matbench_pairwise_dataset(config: MatbenchPairwiseConfig) -> dict[str, Any]:
    rows = load_matbench_log_kvrh_rows(source_path=config.source_path, cache_dir=config.cache_dir)
    payloads = build_matbench_pairwise_payloads(
        rows,
        n_pairs=config.n_pairs,
        seed=config.seed,
        same_system_fraction=config.same_system_fraction,
        close_pair_fraction=config.close_pair_fraction,
        surrogate_noise_floor=config.surrogate_noise_floor,
        surrogate_noise_scale=config.surrogate_noise_scale,
        min_true_gap=config.min_true_gap,
    )
    out_path = Path(config.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    summary = summarize_payloads(payloads)
    if config.update_summary:
        _update_summary_json(out_path.parent, summary)
    return {
        "schema": "matbench-pairwise-build-v1",
        "config": asdict(config),
        "source_rows": len(rows),
        "output_path": str(out_path),
        "summary": summary,
    }


def summarize_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {str(payload.get("group_id", "")) for payload in payloads}
    positives = sum(1 for payload in payloads if payload.get("outcome_success"))
    utilities = [float(payload.get("tool_outputs", {}).get("utility", 0.0)) for payload in payloads]
    return {
        "records": len(payloads),
        "positive_records": positives,
        "negative_records": len(payloads) - positives,
        "positive_rate": positives / len(payloads) if payloads else 0.0,
        "groups": len(groups),
        "metric_keys": ["matbench_log_kvrh_pairwise_gap"],
        "tasks": ["materials_pairwise_preference"],
        "domains": ["materials_project_elasticity"],
        "utility_mean": sum(utilities) / len(utilities) if utilities else 0.0,
        "utility_max": max(utilities) if utilities else 0.0,
    }


def _make_pairwise_payload(
    left_row: dict[str, Any],
    right_row: dict[str, Any],
    *,
    event_seq: int,
    rng: random.Random,
    seed: int,
    surrogate_noise_floor: float,
    surrogate_noise_scale: float,
) -> dict[str, Any]:
    left_true = float(left_row["normalized_k_vrh"])
    right_true = float(right_row["normalized_k_vrh"])
    true_gap = abs(left_true - right_true)
    same_system = str(left_row["crys_sys"]) == str(right_row["crys_sys"])
    system_penalty = 0.0 if same_system else 0.08
    noise = clamp(surrogate_noise_floor + surrogate_noise_scale * (1.0 - true_gap) + system_penalty + rng.random() * 0.04, 0.03, 0.45)
    left_surrogate = clamp(left_true + rng.gauss(0.0, noise))
    right_surrogate = clamp(right_true + rng.gauss(0.0, noise))
    choose_left = left_surrogate >= right_surrogate
    chosen_name = "A" if choose_left else "B"
    chosen_row = left_row if choose_left else right_row
    other_row = right_row if choose_left else left_row
    chosen_true = left_true if choose_left else right_true
    other_true = right_true if choose_left else left_true
    label = int(chosen_true >= other_true)
    utility = true_gap if label else 0.0
    surrogate_margin = abs(left_surrogate - right_surrogate)
    group_id = _group_id(left_row, right_row)
    digest = hashlib.sha1(
        f"{seed}|{event_seq}|{left_row['mbid']}|{right_row['mbid']}|{chosen_name}".encode("utf-8")
    ).hexdigest()[:12]
    return {
        "record_id": f"matbench-pairwise::{digest}",
        "group_id": group_id,
        "domain": "materials_project_elasticity",
        "task": "materials_pairwise_preference",
        "visible_context": (
            "Matbench elasticity pairwise follow-up. "
            f"Candidate A: Composition: {left_row['composition']}; crystal system: {left_row['crys_sys']}; "
            f"space group: {left_row['spg_num']}; surrogate bulk-modulus tier: {_tier(left_surrogate)}. "
            f"Candidate B: Composition: {right_row['composition']}; crystal system: {right_row['crys_sys']}; "
            f"space group: {right_row['spg_num']}; surrogate bulk-modulus tier: {_tier(right_surrogate)}. "
            "The true Materials Project elasticity target is withheld until evaluation."
        ),
        "candidate_action": f"Choose candidate {chosen_name} for high-bulk-modulus materials follow-up.",
        "action_type": "choose",
        "expected_outcome": "The chosen material has higher hidden Matbench bulk-modulus target than the alternative.",
        "metric_key": "matbench_log_kvrh_pairwise_gap",
        "metric_direction": "maximize",
        "metric_value": round(utility, 6),
        "outcome_success": bool(label),
        "agent_id": "matbench_pairwise_builder",
        "trace_id": group_id,
        "evidence": [
            f"A_surrogate_tier={_tier(left_surrogate)}",
            f"B_surrogate_tier={_tier(right_surrogate)}",
            f"surrogate_margin={surrogate_margin:.4f}",
            f"estimated_noise={noise:.4f}",
        ],
        "tool_outputs": {
            "candidate_a": _hidden_material_payload(left_row),
            "candidate_b": _hidden_material_payload(right_row),
            "chosen": chosen_name,
            "chosen_material_id": chosen_row["mbid"],
            "other_material_id": other_row["mbid"],
            "chosen_normalized_k_vrh": chosen_true,
            "other_normalized_k_vrh": other_true,
            "true_gap": true_gap,
            "utility": utility,
        },
        "perturbation_trials": [],
        "context_features": {
            "left_space_group_norm": clamp(float(left_row["spg_num"]) / 230.0),
            "right_space_group_norm": clamp(float(right_row["spg_num"]) / 230.0),
            "same_crystal_system": 1.0 if same_system else 0.0,
            "left_surrogate_score": round(left_surrogate, 6),
            "right_surrogate_score": round(right_surrogate, 6),
            "surrogate_margin": round(surrogate_margin, 6),
            "surrogate_uncertainty": round(noise, 6),
        },
        "uncertainty_signals": {
            "surrogate_margin": round(surrogate_margin, 6),
            "surrogate_uncertainty": round(noise, 6),
            "estimate_agreement": round(clamp(1.0 - noise + 0.25 * surrogate_margin), 6),
            "perturbation_stability": round(clamp(1.0 - noise), 6),
            "source_risk": 0.12,
            "extraction_confidence": 1.0,
        },
        "source": {
            "dataset": "matbench_log_kvrh",
            "dataset_url": MATBENCH_LOG_KVRH_URL,
            "source_note": "Materials Project elasticity data exposed through Matbench; labels compare hidden log10(K_VRH).",
        },
        "verbal_confidence": round(clamp(0.45 + 0.5 * surrogate_margin - 0.3 * noise), 6),
        "cost_level": "low",
        "reversibility": "high",
        "risk_level": "medium" if label else "high",
        "predicted_p_correct": None,
        "runtime_decision": None,
        "label_source": "matbench_log_kvrh_pairwise_property_comparison",
        "label_confidence": 1.0,
        "failure_mode": "lower_hidden_bulk_modulus_chosen" if not label else "",
        "created_online": False,
        "event_seq": event_seq,
        "notes": "Pairwise action built from true Matbench materials rows; true property values are withheld from visible fields.",
    }


def _sample_partner_index(
    left_index: int,
    pool_indices: list[int],
    pool_scores: list[float],
    left_score: float,
    rng: random.Random,
    *,
    close_pair_fraction: float,
) -> int:
    if len(pool_indices) < 2:
        return left_index
    if rng.random() >= close_pair_fraction:
        for _ in range(20):
            candidate_index = rng.choice(pool_indices)
            if candidate_index != left_index:
                return candidate_index
        return left_index
    center = bisect_left(pool_scores, left_score)
    window = max(4, min(len(pool_indices) - 1, int(math.sqrt(len(pool_indices)))))
    low = max(0, center - window)
    high = min(len(pool_indices) - 1, center + window)
    candidates = [pool_indices[index] for index in range(low, high + 1) if pool_indices[index] != left_index]
    if not candidates:
        candidates = [index for index in pool_indices if index != left_index]
    return rng.choice(candidates) if candidates else left_index


def _hidden_material_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "mbid": row["mbid"],
        "composition": row["composition"],
        "crys_sys": row["crys_sys"],
        "spg_num": row["spg_num"],
        "log10_k_vrh": row["log10_k_vrh"],
        "normalized_k_vrh": row["normalized_k_vrh"],
    }


def _group_id(left_row: dict[str, Any], right_row: dict[str, Any]) -> str:
    systems = sorted([str(left_row["crys_sys"]), str(right_row["crys_sys"])])
    relation = "same" if systems[0] == systems[1] else "cross"
    return f"matbench_pairwise::{relation}::{systems[0]}_vs_{systems[1]}"


def _tier(score: float) -> str:
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


def _normalize(values: list[float]) -> list[float]:
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [0.5 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _download_with_cache(url: str, cache_dir: str | Path | None, filename: str) -> bytes:
    if cache_dir:
        cache_path = Path(cache_dir) / filename
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path.read_bytes()
    last_error: Exception | None = None
    for _ in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "harness-matsci/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if cache_dir:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(payload)
            return payload
        except Exception as error:  # pragma: no cover - network-only fallback
            last_error = error
    raise RuntimeError(f"failed to download {url}: {last_error}")


def _update_summary_json(data_dir: Path, task_summary: dict[str, Any]) -> None:
    summary_path = data_dir / "summary.json"
    summary: dict[str, Any]
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {"source_note": "Historical proxy ActionRecord sources.", "tasks": {}, "sources": {}}
    summary.setdefault("tasks", {})["matbench_pairwise"] = {
        "records": task_summary["records"],
        "positive_records": task_summary["positive_records"],
        "negative_records": task_summary["negative_records"],
        "unlabeled_records": 0,
        "groups": task_summary["groups"],
        "metric_keys": task_summary["metric_keys"],
        "tasks": task_summary["tasks"],
        "domains": task_summary["domains"],
    }
    summary.setdefault("sources", {})["matbench_pairwise"] = {
        "paper": "Matbench log10(K_VRH), derived from Materials Project elasticity data",
        "url": MATBENCH_LOG_KVRH_URL,
        "task_type": "A better than B / pairwise material property preference",
        "data": "Pairs of real Matbench materials with composition, crystal system, and space group visible; true log10(K_VRH) is hidden.",
        "label": "The chosen candidate is correct if its hidden normalized log10(K_VRH) is at least the other candidate's value.",
    }
    summary["total_records"] = sum(int(item.get("records", 0)) for item in summary.get("tasks", {}).values())
    summary["total_positive"] = sum(int(item.get("positive_records", 0)) for item in summary.get("tasks", {}).values())
    summary["total_negative"] = sum(int(item.get("negative_records", 0)) for item in summary.get("tasks", {}).values())
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a real-material Matbench pairwise preference action dataset.")
    parser.add_argument("--out", required=True, help=f"Output JSONL path, usually .../{MATBENCH_PAIRWISE_FILENAME}")
    parser.add_argument("--source-path", help="Optional local matbench_log_kvrh.json.bz2 path")
    parser.add_argument("--cache-dir", help="Optional download cache directory")
    parser.add_argument("--n-pairs", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--same-system-fraction", type=float, default=0.6)
    parser.add_argument("--close-pair-fraction", type=float, default=0.75)
    parser.add_argument("--surrogate-noise-floor", type=float, default=0.06)
    parser.add_argument("--surrogate-noise-scale", type=float, default=0.30)
    parser.add_argument("--min-true-gap", type=float, default=0.003)
    parser.add_argument("--update-summary", action="store_true", help="Update summary.json in the output directory")
    args = parser.parse_args(argv)
    report = write_matbench_pairwise_dataset(
        MatbenchPairwiseConfig(
            out=args.out,
            source_path=args.source_path,
            cache_dir=args.cache_dir,
            n_pairs=args.n_pairs,
            seed=args.seed,
            same_system_fraction=args.same_system_fraction,
            close_pair_fraction=args.close_pair_fraction,
            surrogate_noise_floor=args.surrogate_noise_floor,
            surrogate_noise_scale=args.surrogate_noise_scale,
            min_true_gap=args.min_true_gap,
            update_summary=args.update_summary,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
