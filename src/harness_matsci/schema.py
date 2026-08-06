from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ALLOWED_ACTION_TYPES = {
    "choose",
    "choose_candidate",
    "commit_decision",
    "execute_tool",
    "recommend_experiment",
    "recommend",
    "ask_more",
    "retrieve_more",
    "simulate",
    "ask_expert",
    "experiment",
    "summarize_literature",
    "abstain",
    "self_modify",
}

ALLOWED_ROUTES = {
    "proceed",
    "retrieve_more",
    "simulate",
    "ask_expert",
    "experiment",
    "abstain",
}


@dataclass(frozen=True)
class ActionRecord:
    """One supervised scientific action decision.

    The record is intentionally action-level rather than answer-level.  The
    base scientific agent proposes ``candidate_action`` from ``visible_context``;
    the harness learns whether this action is worth executing without seeing
    hidden outcomes at decision time.
    """

    record_id: str
    benchmark: str
    split: str
    visible_context: str
    candidate_action: str
    action_type: str
    evidence: list[str]
    features: dict[str, float]
    label: int
    utility: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_type not in ALLOWED_ACTION_TYPES:
            raise ValueError(f"unknown action_type: {self.action_type}")
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        for key, value in self.features.items():
            if not isinstance(key, str):
                raise TypeError("feature names must be strings")
            if not isinstance(value, (int, float)):
                raise TypeError(f"feature {key!r} must be numeric")

    def to_json(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "benchmark": self.benchmark,
            "split": self.split,
            "visible_context": self.visible_context,
            "candidate_action": self.candidate_action,
            "action_type": self.action_type,
            "evidence": list(self.evidence),
            "features": dict(self.features),
            "label": self.label,
            "utility": self.utility,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ActionRecord":
        return cls(
            record_id=str(payload["record_id"]),
            benchmark=str(payload["benchmark"]),
            split=str(payload.get("split", "unspecified")),
            visible_context=str(payload["visible_context"]),
            candidate_action=str(payload["candidate_action"]),
            action_type=str(payload["action_type"]),
            evidence=[str(x) for x in payload.get("evidence", [])],
            features={str(k): float(v) for k, v in payload.get("features", {}).items()},
            label=int(payload["label"]),
            utility=float(payload.get("utility", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class GateDecision:
    record_id: str
    reliability: float
    threshold: float
    selected: bool
    route: str
    rationale: str

    def __post_init__(self) -> None:
        if self.route not in ALLOWED_ROUTES:
            raise ValueError(f"unknown route: {self.route}")

    def to_json(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "reliability": self.reliability,
            "threshold": self.threshold,
            "selected": self.selected,
            "route": self.route,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class HarnessSpec:
    """Declarative runtime decision harness contract."""

    name: str
    required_features: list[str]
    proceed_routes: list[str]
    fallback_routes: list[str]
    target_selective_risk: float

    def validate_record(self, record: ActionRecord) -> None:
        missing = [name for name in self.required_features if name not in record.features]
        if missing:
            raise ValueError(f"record {record.record_id} missing features: {missing}")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "required_features": list(self.required_features),
            "proceed_routes": list(self.proceed_routes),
            "fallback_routes": list(self.fallback_routes),
            "target_selective_risk": self.target_selective_risk,
        }
