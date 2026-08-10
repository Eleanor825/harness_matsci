from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .historical import HISTORICAL_TASK_FILES, ORACLE_SIGNAL_KEYS, load_historical_task_records
from .schema import ActionRecord


FORBIDDEN_VISIBLE_PATTERNS = (
    "hidden objective value is",
    "log10(k_vrh)=",
    "hit_fraction=",
    "target_hit_score",
    "reward=",
    "all_hit=",
    "latent_utility",
    "discovery_score",
)

PAIRWISE_LABEL_SOURCE = "simulated_noisy_preferential_duel_from_published_objective"


@dataclass(frozen=True)
class LabelAuditConfig:
    data_dir: str
    tasks: tuple[str, ...] = tuple(HISTORICAL_TASK_FILES)
    sample_per_task: int = 12
    seed: int = 1729
    tolerance: float = 1e-8


def run_label_utility_audit(config: LabelAuditConfig) -> dict[str, Any]:
    task_summaries: dict[str, Any] = {}
    sample_manifest: list[dict[str, Any]] = []
    for task in config.tasks:
        raw_payloads = _read_raw_payloads(config.data_dir, task)
        records = load_historical_task_records(config.data_dir, task)
        summary = _audit_task(task, raw_payloads, records, config)
        task_summaries[task] = summary
        sample_manifest.extend(summary["sample_records"])

    aggregate = {
        "total_raw_records": sum(item["raw_records"] for item in task_summaries.values()),
        "total_action_records": sum(item["converted_records"] for item in task_summaries.values()),
        "all_label_consistency_passed": all(item["consistency"]["label_mismatch_count"] == 0 for item in task_summaries.values()),
        "all_utility_consistency_passed": all(item["consistency"]["utility_mismatch_count"] == 0 for item in task_summaries.values()),
        "all_visible_leakage_free": all(item["leakage"]["visible_text_forbidden_count"] == 0 for item in task_summaries.values()),
        "all_raw_oracle_keys_recorded_as_excluded": all(item["leakage"]["missing_excluded_oracle_key_count"] == 0 for item in task_summaries.values()),
    }
    return {
        "schema": "label-utility-audit-v1",
        "config": asdict(config),
        "aggregate": aggregate,
        "tasks": task_summaries,
        "sample_manifest": sample_manifest,
        "limitations": [
            "This audit proves internal proxy-label consistency and leakage sanitation; it is not a human expert annotation study.",
            "Preferential BO labels are simulated noisy duels derived from published latent objectives, not real online BO trajectories.",
            "Unique-material and extreme-property labels inherit the validity of the historical benchmark builders and should still be spot-checked by domain experts before a final paper claim.",
        ],
    }


def save_label_utility_audit(report: dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_target.write_text(render_label_utility_audit_markdown(report), encoding="utf-8")


def render_label_utility_audit_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# Label and Utility Audit",
        "",
        "This audit checks whether the offline proxy labels/utilities are internally consistent with their stored benchmark outcomes and whether hidden outcome fields are withheld from visible decision-time text.",
        "",
        "## Aggregate Verdict",
        "",
        f"- Raw records: `{aggregate['total_raw_records']}`.",
        f"- Converted action records: `{aggregate['total_action_records']}`.",
        f"- Label consistency passed: `{aggregate['all_label_consistency_passed']}`.",
        f"- Utility consistency passed: `{aggregate['all_utility_consistency_passed']}`.",
        f"- Visible leakage free: `{aggregate['all_visible_leakage_free']}`.",
        f"- Raw oracle keys recorded as excluded: `{aggregate['all_raw_oracle_keys_recorded_as_excluded']}`.",
        "",
        "## Task Summary",
        "",
        "| Task | Records | Groups | Positive rate | Utility mean | Label mismatches | Utility mismatches | Visible leaks | Raw oracle keys excluded |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, summary in report["tasks"].items():
        utility = summary["utility_distribution"]
        consistency = summary["consistency"]
        leakage = summary["leakage"]
        excluded_ratio = "n/a" if leakage["raw_oracle_key_count"] == 0 else f"{leakage['raw_oracle_key_count'] - leakage['missing_excluded_oracle_key_count']}/{leakage['raw_oracle_key_count']}"
        lines.append(
            f"| `{task}` | {summary['converted_records']} | {summary['groups']} | {summary['positive_rate']:.3f} | "
            f"{utility['mean']:.4f} | {consistency['label_mismatch_count']} | {consistency['utility_mismatch_count']} | "
            f"{leakage['visible_text_forbidden_count']} | {excluded_ratio} |"
        )
    lines.extend([
        "",
        "## What This Establishes",
        "",
        "- `label` is reproducible from stored benchmark outcome metadata for all converted action records.",
        "- `utility` is reproducible from raw benchmark utility fields for non-pairwise tasks and from true pairwise margins for preferential duels.",
        "- Hidden outcome strings such as exact objective values, `log10(K_VRH)`, `hit_fraction`, `reward`, and `all_hit` are absent from visible context/evidence after conversion.",
        "- Raw oracle-valued uncertainty fields are recorded in `excluded_oracle_features` so reviewers can audit what was withheld.",
        "",
        "## Claim Boundary",
        "",
        "This is a proxy-label consistency audit, not proof that live MatBot action labels are correct. A final paper should still add a small expert spot-check or online trajectory audit.",
    ])
    return "\n".join(lines) + "\n"


def _audit_task(task: str, raw_payloads: list[dict[str, Any]], records: list[ActionRecord], config: LabelAuditConfig) -> dict[str, Any]:
    raw_by_id = {str(payload.get("record_id", "")): payload for payload in raw_payloads}
    label_failures: list[dict[str, Any]] = []
    utility_failures: list[dict[str, Any]] = []
    tie_count = 0
    negative_positive_utility = 0
    positive_zero_utility = 0
    visible_leaks: list[dict[str, Any]] = []
    missing_excluded_keys: list[dict[str, Any]] = []
    raw_oracle_key_count = 0

    for record in records:
        label_expected, utility_expected, is_tie = _expected_label_and_utility(task, record, raw_by_id)
        if label_expected is None:
            label_failures.append({"record_id": record.record_id, "reason": "missing_expected_label"})
        elif int(label_expected) != record.label:
            label_failures.append({"record_id": record.record_id, "expected": int(label_expected), "observed": record.label})
        if utility_expected is None:
            utility_failures.append({"record_id": record.record_id, "reason": "missing_expected_utility"})
        elif not _close(float(utility_expected), float(record.utility), config.tolerance):
            utility_failures.append({"record_id": record.record_id, "expected": float(utility_expected), "observed": record.utility})
        tie_count += int(is_tie)
        negative_positive_utility += int(record.label == 0 and record.utility > config.tolerance)
        positive_zero_utility += int(record.label == 1 and record.utility <= config.tolerance)

        matched = _forbidden_visible_matches(record)
        if matched:
            visible_leaks.append({"record_id": record.record_id, "patterns": matched})

        if task != "preferential_bo":
            raw_keys = set(str(key) for key in record.metadata.get("raw_uncertainty_keys", [])) & set(ORACLE_SIGNAL_KEYS)
            raw_oracle_key_count += len(raw_keys)
            excluded = set(str(key) for key in record.metadata.get("excluded_oracle_features", []))
            missing = sorted(raw_keys - excluded)
            if missing:
                missing_excluded_keys.append({"record_id": record.record_id, "missing": missing})

    utilities = [record.utility for record in records]
    labels = [record.label for record in records]
    groups = {str(record.metadata.get("group_id", record.benchmark)) for record in records}
    return {
        "raw_records": len(raw_payloads),
        "converted_records": len(records),
        "groups": len(groups),
        "positive_rate": mean(labels) if labels else 0.0,
        "label_sources": sorted({str(record.metadata.get("label_source", "unknown")) for record in records}),
        "utility_distribution": _numeric_summary(utilities),
        "consistency": {
            "label_mismatch_count": len(label_failures),
            "utility_mismatch_count": len(utility_failures),
            "label_failures_preview": label_failures[:20],
            "utility_failures_preview": utility_failures[:20],
            "pairwise_tie_count": tie_count if task == "preferential_bo" else 0,
            "negative_records_with_positive_utility": negative_positive_utility,
            "positive_records_with_zero_utility": positive_zero_utility,
        },
        "leakage": {
            "visible_text_forbidden_count": len(visible_leaks),
            "visible_text_forbidden_preview": visible_leaks[:20],
            "raw_oracle_key_count": raw_oracle_key_count,
            "missing_excluded_oracle_key_count": len(missing_excluded_keys),
            "missing_excluded_oracle_key_preview": missing_excluded_keys[:20],
        },
        "sample_records": _sample_records(task, records, config.sample_per_task, config.seed),
    }


def _expected_label_and_utility(
    task: str,
    record: ActionRecord,
    raw_by_id: dict[str, dict[str, Any]],
) -> tuple[int | None, float | None, bool]:
    if task == "preferential_bo":
        metadata = record.metadata
        try:
            left = float(metadata["left_true_utility"])
            right = float(metadata["right_true_utility"])
            chosen = str(metadata["chosen"])
            chosen_true = left if chosen == "A" else right
            other_true = right if chosen == "A" else left
            expected_label = int(chosen_true >= other_true)
            true_margin = max(0.0, min(1.0, abs(left - right)))
            expected_utility = true_margin if expected_label else 0.0
            return expected_label, expected_utility, math.isclose(left, right, abs_tol=1e-12)
        except Exception:
            return None, None, False

    raw_id = str(record.metadata.get("raw_record_id", record.record_id))
    raw = raw_by_id.get(raw_id)
    expected_label = int(bool(record.metadata.get("outcome_success"))) if "outcome_success" in record.metadata else None
    if raw is None:
        return expected_label, None, False
    tool_outputs = raw.get("tool_outputs", {})
    expected_utility = tool_outputs.get("utility", raw.get("metric_value", 0.0)) if isinstance(tool_outputs, dict) else raw.get("metric_value", 0.0)
    return expected_label, float(expected_utility), False


def _read_raw_payloads(data_dir: str | Path, task: str) -> list[dict[str, Any]]:
    filename = HISTORICAL_TASK_FILES[task]
    path = Path(data_dir) / filename
    payloads: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payloads.append(json.loads(line))
    return payloads


def _forbidden_visible_matches(record: ActionRecord) -> list[str]:
    text = " ".join([record.visible_context, record.candidate_action, *record.evidence]).lower()
    return [pattern for pattern in FORBIDDEN_VISIBLE_PATTERNS if pattern in text]


def _numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def _sample_records(task: str, records: list[ActionRecord], sample_per_task: int, seed: int) -> list[dict[str, Any]]:
    positives = _deterministic_order([record for record in records if record.label == 1], seed)
    negatives = _deterministic_order([record for record in records if record.label == 0], seed)
    positive_quota = min(len(positives), sample_per_task // 2)
    negative_quota = min(len(negatives), sample_per_task - positive_quota)
    selected = positives[:positive_quota] + negatives[:negative_quota]
    if len(selected) < sample_per_task:
        seen = {record.record_id for record in selected}
        selected.extend(record for record in _deterministic_order(records, seed) if record.record_id not in seen)
    selected = selected[:sample_per_task]
    return [
        {
            "task": task,
            "record_id": record.record_id,
            "group_id": str(record.metadata.get("group_id", record.benchmark)),
            "label": record.label,
            "utility": record.utility,
            "label_source": str(record.metadata.get("label_source", "unknown")),
            "visible_context_excerpt": _excerpt(record.visible_context),
            "candidate_action_excerpt": _excerpt(record.candidate_action),
            "evidence_excerpt": [_excerpt(item) for item in record.evidence[:3]],
        }
        for record in selected
    ]


def _deterministic_order(records: list[ActionRecord], seed: int) -> list[ActionRecord]:
    return sorted(records, key=lambda record: hashlib.sha256(f"audit|{seed}|{record.record_id}".encode()).hexdigest())


def _excerpt(text: str, limit: int = 240) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def _close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance + tolerance * max(abs(left), abs(right), 1.0)
