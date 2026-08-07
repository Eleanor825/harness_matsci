from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .features import clamp
from .io import write_json, write_jsonl
from .schema import ActionRecord
from .training import TrainedGate, evidence_heuristic_baseline, evaluate_gate, train_gate_with_features, verbal_confidence_baseline


DEFAULT_PAPER_ACTIONS_PATH = Path("/Users/huan/matsci_uncertainty/data/raw/nature_electrolyte/training_actions_500.jsonl")

PAPER_V1_FEATURES = [
    "verbal_confidence",
    "evidence_support",
    "evidence_conflict",
    "tool_agreement",
    "cost",
    "reversibility",
]

PAPER_V2_FEATURES = [
    "audited_ld",
    "preference_margin",
    "verbal_confidence",
    "perturbation_stability",
    "evidence_support",
    "evidence_conflict",
    "source_reliability",
    "tool_agreement",
    "model_disagreement",
    "ood_score",
    "experimental_variance",
    "cost",
    "reversibility",
    "lineage_impact",
    "source_risk",
    "extraction_confidence",
    "consensus_spread",
    "segment_length",
    "title_length",
    "abstract_length",
    "has_pdf",
]

ACTION_TYPE_MAP = {
    "ask_more": "retrieve_more",
    "retrieve_more": "retrieve_more",
    "summarize_literature": "retrieve_more",
    "recommend_experiment": "recommend_experiment",
    "execute_tool": "execute_tool",
    "commit_decision": "commit_decision",
    "choose_candidate": "choose_candidate",
    "abstain": "abstain",
}

COST_LEVELS = {"low": 0.25, "medium": 0.60, "high": 0.90}
REVERSIBILITY_LEVELS = {"low": 0.25, "medium": 0.60, "high": 0.90}
RISK_LEVELS = {"low": 0.20, "medium": 0.55, "high": 0.90}


def load_paper_action_records(path: str | Path = DEFAULT_PAPER_ACTIONS_PATH) -> list[ActionRecord]:
    source = Path(path)
    records: list[ActionRecord] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                records.append(paper_json_to_action_record(raw))
            except Exception as exc:
                raise ValueError(f"invalid paper action record at {source}:{line_number}: {exc}") from exc
    return records


def paper_json_to_action_record(raw: dict[str, Any]) -> ActionRecord:
    signals = raw.get("uncertainty_signals", {}) or {}
    context_features = raw.get("context_features", {}) or {}

    evidence_support = _signal(signals, "evidence_support", 0.5)
    evidence_conflict = _signal(signals, "evidence_conflict", 0.0)
    verbal_confidence = _float(raw.get("verbal_confidence"), _signal(signals, "verbal_confidence", 0.5))
    source_risk = _signal(signals, "source_risk", _level(raw.get("risk_level"), RISK_LEVELS, 0.55))
    source_reliability = clamp(1.0 - source_risk)
    extraction_confidence = _signal(signals, "extraction_confidence", source_reliability)
    ood_score = _signal(signals, "ood_score", 0.5)
    consensus_spread = _signal(signals, "consensus_spread", 0.5)
    perturbation_stability = _signal(signals, "perturbation_stability", 0.5)
    tool_agreement = _signal(signals, "tool_agreement", 0.5)
    cost = _level(raw.get("cost_level"), COST_LEVELS, 0.6)
    reversibility = _level(raw.get("reversibility"), REVERSIBILITY_LEVELS, 0.6)
    risk = _level(raw.get("risk_level"), RISK_LEVELS, 0.55)

    experimental_variance = clamp(0.45 * consensus_spread + 0.35 * (1.0 - extraction_confidence) + 0.20 * ood_score)
    audited_ld = clamp(
        0.34 * evidence_support
        + 0.20 * tool_agreement
        + 0.18 * perturbation_stability
        + 0.18 * source_reliability
        + 0.10 * extraction_confidence
        - 0.24 * evidence_conflict
    )
    preference_margin = clamp(abs(verbal_confidence - 0.5) * 2.0 + 0.25 * abs(tool_agreement - consensus_spread))
    lineage_impact = clamp(0.45 * risk + 0.35 * cost + 0.20 * (1.0 - reversibility))

    features = {
        "audited_ld": audited_ld,
        "preference_margin": preference_margin,
        "verbal_confidence": clamp(verbal_confidence),
        "perturbation_stability": perturbation_stability,
        "evidence_support": evidence_support,
        "evidence_conflict": evidence_conflict,
        "source_reliability": source_reliability,
        "tool_agreement": tool_agreement,
        "model_disagreement": consensus_spread,
        "ood_score": ood_score,
        "experimental_variance": experimental_variance,
        "cost": cost,
        "reversibility": reversibility,
        "lineage_impact": lineage_impact,
        "source_risk": source_risk,
        "extraction_confidence": extraction_confidence,
        "consensus_spread": consensus_spread,
        "segment_length": _float(context_features.get("segment_length"), 0.0),
        "title_length": _float(context_features.get("title_length"), 0.0),
        "abstract_length": _float(context_features.get("abstract_length"), 0.0),
        "has_pdf": 1.0 if context_features.get("has_pdf") else 0.0,
    }

    raw_action_type = str(raw.get("action_type", "recommend"))
    action_type = ACTION_TYPE_MAP.get(raw_action_type, raw_action_type)
    evidence = raw.get("evidence", []) or []
    metadata = {
        "group_id": raw.get("group_id"),
        "trace_id": raw.get("trace_id"),
        "domain": raw.get("domain"),
        "task": raw.get("task"),
        "raw_action_type": raw_action_type,
        "expected_outcome": raw.get("expected_outcome"),
        "metric_key": raw.get("metric_key"),
        "metric_value": _float(raw.get("metric_value"), 0.0),
        "metric_direction": raw.get("metric_direction"),
        "runtime_decision": raw.get("runtime_decision"),
        "label_source": raw.get("label_source"),
        "label_confidence": _float(raw.get("label_confidence"), 0.0),
        "failure_mode": raw.get("failure_mode"),
        "created_online": bool(raw.get("created_online", False)),
        "event_seq": raw.get("event_seq"),
        "source": raw.get("source"),
        "context_features": context_features,
        "uncertainty_signals": signals,
    }
    return ActionRecord(
        record_id=str(raw["record_id"]),
        benchmark="nature_electrolyte_paper_bootstrap",
        split="unsplit",
        visible_context=str(raw.get("visible_context", "")),
        candidate_action=str(raw.get("candidate_action", "")),
        action_type=action_type,
        evidence=[str(item) for item in evidence],
        features=features,
        label=1 if bool(raw.get("outcome_success")) else 0,
        utility=_float(raw.get("metric_value"), 0.0),
        metadata=metadata,
    )


def split_records_by_group(
    records: list[ActionRecord],
    *,
    seed: int = 7,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    group_key: str = "group_id",
) -> tuple[list[ActionRecord], list[ActionRecord], list[ActionRecord], dict[str, Any]]:
    if not records:
        return [], [], [], {"groups": 0, "train_groups": 0, "val_groups": 0, "test_groups": 0}
    groups: dict[str, list[ActionRecord]] = defaultdict(list)
    for record in records:
        group_id = str(record.metadata.get(group_key) or record.metadata.get("trace_id") or record.record_id)
        groups[group_id].append(record)
    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    n_train = int(len(group_ids) * train_fraction)
    n_val = int(len(group_ids) * val_fraction)
    train_ids = set(group_ids[:n_train])
    val_ids = set(group_ids[n_train : n_train + n_val])
    test_ids = set(group_ids[n_train + n_val :])

    def assign_split(record: ActionRecord, split: str) -> ActionRecord:
        payload = record.to_json()
        payload["split"] = split
        return ActionRecord.from_json(payload)

    train_records = [assign_split(record, "train") for group in train_ids for record in groups[group]]
    val_records = [assign_split(record, "val") for group in val_ids for record in groups[group]]
    test_records = [assign_split(record, "test") for group in test_ids for record in groups[group]]
    summary = {
        "groups": len(group_ids),
        "train_groups": len(train_ids),
        "val_groups": len(val_ids),
        "test_groups": len(test_ids),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "test_records": len(test_records),
    }
    return train_records, val_records, test_records, summary


def run_paper_bootstrap_experiment(
    data_path: str | Path = DEFAULT_PAPER_ACTIONS_PATH,
    *,
    workdir: str | Path | None = None,
    seed: int = 7,
    alpha: float = 0.1,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    epochs: int = 700,
    learning_rate: float = 0.08,
    l2: float = 0.001,
    budget_fraction: float = 0.1,
) -> dict[str, Any]:
    records = load_paper_action_records(data_path)
    train_records, val_records, test_records, split_summary = split_records_by_group(
        records,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    versions = {
        "v1_compact_uncertainty_contract": PAPER_V1_FEATURES,
        "v2_extended_paper_harness": PAPER_V2_FEATURES,
    }
    version_reports: dict[str, Any] = {}
    for name, feature_names in versions.items():
        gate = train_gate_with_features(
            train_records,
            val_records,
            feature_names=feature_names,
            alpha=alpha,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        version_reports[name] = {
            "feature_names": list(feature_names),
            "feature_count": len(feature_names),
            "gate": gate.to_json(),
            "val": evaluate_gate(val_records, gate, budget_fraction=budget_fraction),
            "test": evaluate_gate(test_records, gate, budget_fraction=budget_fraction),
        }

    selected_version = _select_version(version_reports)
    report = {
        "experiment": {
            "name": "paper_bootstrap_v1",
            "objective": "train a runtime action-value uncertainty harness from historical electrolyte papers",
            "data_path": str(data_path),
            "seed": seed,
            "alpha": alpha,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "budget_fraction": budget_fraction,
        },
        "dataset": summarize_paper_records(records),
        "split": split_summary,
        "versions": version_reports,
        "selected_version": selected_version,
        "baselines": {
            "verbal_confidence": verbal_confidence_baseline(test_records, alpha=alpha, budget_fraction=budget_fraction),
            "evidence_heuristic": evidence_heuristic_baseline(test_records, alpha=alpha, budget_fraction=budget_fraction),
        },
        "notes": [
            "Records are reconstructed from historical papers and use weak post-hoc labels.",
            "Splits are grouped by paper group_id to reduce paper-level leakage.",
            "This is a bootstrap substitute for online MatBot trajectory logging, not a replacement for final runtime data.",
        ],
    }

    if workdir is not None:
        target = Path(workdir)
        target.mkdir(parents=True, exist_ok=True)
        split_records = train_records + val_records + test_records
        write_jsonl(split_records, target / "normalized_actions.jsonl")
        write_json(report, target / "summary.json")
        write_json(version_reports[selected_version]["gate"], target / "selected_gate.json")
    return report


def summarize_paper_records(records: list[ActionRecord]) -> dict[str, Any]:
    action_types = Counter(record.metadata.get("raw_action_type", record.action_type) for record in records)
    groups = {str(record.metadata.get("group_id") or record.record_id) for record in records}
    labels = Counter(record.label for record in records)
    domains = Counter(str(record.metadata.get("domain", "unknown")) for record in records)
    return {
        "records": len(records),
        "groups": len(groups),
        "positive": int(labels[1]),
        "negative": int(labels[0]),
        "action_types": dict(sorted(action_types.items())),
        "domains": dict(sorted(domains.items())),
    }


def _select_version(version_reports: dict[str, Any]) -> str:
    def key(item: tuple[str, Any]) -> tuple[float, float, float]:
        metrics = item[1]["val"]["metrics"]
        return (
            float(metrics["selective_risk"]),
            float(metrics["brier"]),
            -float(metrics["coverage"]),
        )

    return min(version_reports.items(), key=key)[0]


def _float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _signal(signals: dict[str, Any], key: str, default: float) -> float:
    return clamp(_float(signals.get(key), default))


def _level(value: Any, mapping: dict[str, float], default: float) -> float:
    if value is None:
        return float(default)
    return float(mapping.get(str(value).lower(), default))
