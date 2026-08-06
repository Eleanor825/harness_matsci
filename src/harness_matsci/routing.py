from __future__ import annotations

from .schema import ActionRecord, GateDecision


def route_record(record: ActionRecord, reliability: float, threshold: float) -> GateDecision:
    selected = reliability >= threshold
    if selected:
        if record.action_type in {"experiment", "recommend_experiment", "execute_tool"}:
            route = "experiment"
            rationale = "calibrated reliability clears threshold for experiment action"
        elif record.action_type in {"retrieve_more", "ask_more", "summarize_literature"}:
            route = "retrieve_more"
            rationale = "calibrated reliability clears threshold for retrieval-oriented action"
        else:
            route = "proceed"
            rationale = "calibrated reliability clears proceed threshold"
    else:
        features = record.features
        if features.get("evidence_conflict", 0.0) >= 0.45 or features.get("source_reliability", 1.0) <= 0.35:
            route = "retrieve_more"
            rationale = "reliability below threshold with evidence conflict or weak source reliability"
        elif features.get("model_disagreement", 0.0) >= 0.45 or features.get("tool_agreement", 1.0) <= 0.45:
            route = "simulate"
            rationale = "reliability below threshold with tool/model disagreement"
        elif features.get("cost", 0.0) >= 0.65 or features.get("reversibility", 1.0) <= 0.35:
            route = "ask_expert"
            rationale = "reliability below threshold for high-cost or low-reversibility action"
        else:
            route = "abstain"
            rationale = "reliability below threshold and no clearly reducible uncertainty route dominates"
    return GateDecision(
        record_id=record.record_id,
        reliability=float(reliability),
        threshold=float(threshold),
        selected=selected,
        route=route,
        rationale=rationale,
    )
