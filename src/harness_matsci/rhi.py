from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .harnesses import DEFAULT_HARNESS, deterministic_fallback, harness_text, validate_harness
from .schema import ActionRecord
from .training import TrainedGate, evaluate_gate, split_records


RHI_SEED_FEATURES = [
    "verbal_confidence",
    "evidence_support",
    "evidence_conflict",
    "ood_score",
]

RHI_SEED_HARNESS = {
    **copy.deepcopy(DEFAULT_HARNESS),
    "name": "H0_rhi_matsci",
    "required_features": list(RHI_SEED_FEATURES),
    "hops": [
        {"from": "orchestrator", "to": "evidence_auditor", "purpose": "audit visible evidence"},
        {"from": "evidence_auditor", "to": "uncertainty_gate", "purpose": "estimate action worthiness"},
        {"from": "uncertainty_gate", "to": "fallback_router", "purpose": "select proceed or fallback route"},
    ],
}


@dataclass(frozen=True)
class TrajectoryFeedback:
    """Feedback extracted from action-level trajectories for one RHI round."""

    round_index: int
    harness_name: str
    n_records: int
    failure_counts: dict[str, int]
    metric_snapshot: dict[str, float]
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HarnessProposal:
    """A candidate harness and the evidence that motivated its mutation."""

    candidate: dict[str, Any]
    proposer: str
    rationale: list[str]
    feedback: TrajectoryFeedback

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "proposer": self.proposer,
            "rationale": list(self.rationale),
            "feedback": self.feedback.to_json(),
        }


class HarnessProposer(Protocol):
    def propose(
        self,
        previous: dict[str, Any],
        feedback: TrajectoryFeedback,
        *,
        iteration: int,
    ) -> HarnessProposal:
        ...


class DeterministicTrajectoryProposer:
    """Offline proposer used when no external LLM is configured.

    The proposer is deliberately trajectory-conditioned: it counts concrete
    failure modes and mutates the contracts and hops that address those
    failures.  This keeps the RHI loop reproducible while exposing the same
    interface as an LLM proposer.
    """

    def propose(
        self,
        previous: dict[str, Any],
        feedback: TrajectoryFeedback,
        *,
        iteration: int,
    ) -> HarnessProposal:
        candidate = copy.deepcopy(previous)
        candidate["name"] = f"H{iteration}_trajectory_feedback"
        required_features = list(candidate.get("required_features", []))
        gates = list(candidate.get("gates", []))
        roles = copy.deepcopy(candidate.get("roles", []))
        rationale: list[str] = []
        counts = feedback.failure_counts

        def add_feature(name: str, reason: str) -> None:
            if name not in required_features:
                required_features.append(name)
                rationale.append(reason)

        def add_gate(text: str) -> None:
            if text not in gates:
                gates.append(text)

        if counts.get("confidently_wrong", 0) > 0 or counts.get("evidence_conflict", 0) > 0:
            rationale.append("Strengthen evidence auditing after confidently wrong or conflicting trajectories.")
            add_feature("evidence_conflict", "Add conflict-sensitive evidence checking after confidently wrong actions.")
            add_feature("source_reliability", "Require source reliability before promoting a scientific action.")
            add_gate("Evidence conflict or weak source reliability must trigger retrieval before proceed.")
            _upsert_role(
                roles,
                "evidence_auditor",
                "Before a costly action, identify conflicting evidence and downgrade unsupported claims.",
                ["evidence_conflict", "source_reliability", "evidence_support"],
            )

        if counts.get("overconfident", 0) > 0 or counts.get("calibration_error", 0) > 0:
            rationale.append("Add a calibration review contract for overconfident trajectories.")
            add_feature("verbal_confidence", "Compare verbal confidence with evidence-backed reliability.")
            add_feature("model_disagreement", "Expose disagreement as a separate uncertainty signal.")
            add_gate("Verbal confidence cannot override disagreement or missing evidence.")
            _upsert_role(
                roles,
                "calibration_reviewer",
                "Check whether confidence tracks observed action reliability on the current discovery regime.",
                ["verbal_confidence", "model_disagreement", "calibrated_probability"],
                kind="reviewer",
            )

        if counts.get("high_ood", 0) > 0 or counts.get("high_cost", 0) > 0:
            rationale.append("Add cost, reversibility, and OOD-aware verification hops.")
            add_feature("ood_score", "Route out-of-distribution candidates to verification before execution.")
            add_feature("cost", "Use action cost in the promotion decision.")
            add_feature("reversibility", "Use reversibility to separate cheap probes from irreversible experiments.")
            add_gate("High OOD, high cost, or low reversibility requires an explicit verification hop.")
            _upsert_role(
                roles,
                "risk_router",
                "Select verification, simulation, expert review, or abstention based on OOD and action cost.",
                ["ood_score", "cost", "reversibility", "route_reason"],
                kind="reviewer",
            )

        if counts.get("over_abstention", 0) > 0:
            rationale.append("Recover useful coverage with independent tool agreement.")
            add_feature("tool_agreement", "Recover coverage when independent tools agree on a low-risk action.")
            add_gate("Do not abstain when evidence and tools agree and the action is cheap and reversible.")

        candidate["required_features"] = required_features
        candidate["gates"] = gates[:20]
        candidate["roles"] = roles
        candidate["hops"] = _build_hops(candidate, counts)
        candidate["target_selective_risk"] = _updated_alpha(candidate, feedback)
        if not any(counts.values()):
            rationale.append("No new failure slice exceeded the mutation threshold; preserve the current contract.")
            candidate = deterministic_fallback(previous, iteration)
        return HarnessProposal(candidate, "deterministic_trajectory_proposer", rationale, feedback)


class JSONLLMHarnessProposer:
    """Optional OpenAI-compatible proposer driven by trajectory feedback.

    The model is only responsible for proposing a declarative harness.  The
    caller still validates the schema and the held-out gate decides whether
    the proposal is accepted.  No provider dependency is required unless this
    class is explicitly used.
    """

    def __init__(self, client: Any, *, model: str) -> None:
        self.client = client
        self.model = model

    def propose(
        self,
        previous: dict[str, Any],
        feedback: TrajectoryFeedback,
        *,
        iteration: int,
    ) -> HarnessProposal:
        prompt = _proposal_prompt(previous, feedback, iteration)
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0.2,
        )
        text = getattr(response, "output_text", "")
        if not text:
            raise ValueError("LLM proposer returned no output_text")
        candidate = _extract_json(text)
        return HarnessProposal(
            candidate=candidate,
            proposer=f"openai_responses:{self.model}",
            rationale=["Candidate generated from trajectory-local failure feedback by an external proposer."],
            feedback=feedback,
        )


def train_rhi(
    records: list[ActionRecord],
    *,
    iterations: int = 3,
    seed: int = 7,
    alpha: float = 0.1,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    epochs: int = 700,
    learning_rate: float = 0.08,
    l2: float = 0.001,
    budget_fraction: float = 0.1,
    epsilon: float = 0.01,
    proposer: HarnessProposer | None = None,
) -> dict[str, Any]:
    """Run recursive, trajectory-feedback-conditioned harness improvement.

    Each candidate is trained from the same training split, compared only to
    its immediate predecessor on the same held-out validation split, and
    accepted only when its composite action-worthiness score improves without
    violating the coverage guard.  The test split is touched only at the end.
    """
    if not records:
        raise ValueError("records cannot be empty")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    train_records, val_records, test_records = split_records(
        records,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    if not train_records or not val_records:
        raise ValueError("RHI requires non-empty train and validation splits")

    active_proposer = proposer or DeterministicTrajectoryProposer()
    current_harness = copy.deepcopy(RHI_SEED_HARNESS)
    current_gate = _train_with_features(
        train_records,
        val_records,
        feature_names=list(current_harness["required_features"]),
        alpha=alpha,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    current_eval = evaluate_gate(val_records, current_gate, budget_fraction=budget_fraction)
    versions: list[dict[str, Any]] = [_version_report(0, current_harness, current_eval, current_gate)]
    proposals: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    for iteration in range(1, iterations + 1):
        feedback = summarize_trajectory_feedback(
            val_records,
            current_gate,
            current_harness,
            round_index=iteration,
            budget_fraction=budget_fraction,
        )
        proposal = active_proposer.propose(current_harness, feedback, iteration=iteration)
        candidate, validation_status = validate_harness(proposal.candidate, current_harness, iteration)
        proposal_payload = proposal.to_json()
        proposal_payload["validation_status"] = validation_status
        proposal_payload["candidate"] = candidate
        proposals.append(proposal_payload)

        feature_names = list(candidate.get("required_features", current_gate.model.feature_names))
        candidate_gate = _train_with_features(
            train_records,
            val_records,
            feature_names=feature_names,
            alpha=float(candidate.get("target_selective_risk", alpha)),
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        candidate_eval = evaluate_gate(val_records, candidate_gate, budget_fraction=budget_fraction)
        comparison = compare_harnesses(
            current_eval,
            candidate_eval,
            epsilon=epsilon,
            iteration=iteration,
            previous_name=str(current_harness.get("name")),
            candidate_name=str(candidate.get("name")),
        )
        comparisons.append(comparison)
        versions.append(_version_report(iteration, candidate, candidate_eval, candidate_gate))
        if comparison["winner"] != "candidate":
            continue
        current_harness = candidate
        current_gate = candidate_gate
        current_eval = candidate_eval

    final_test = evaluate_gate(test_records, current_gate, budget_fraction=budget_fraction)
    return {
        "method": {
            "name": "RHI-MatSci",
            "paper": "Recursive Harness Self-Improvement, arXiv:2607.15524",
            "loop": [
                "solve/observe action trajectories",
                "summarize held-out failure feedback",
                "propose prompt, contract, and hop mutation",
                "validate declarative harness schema",
                "compare consecutive versions on validation",
                "accept candidate or retain predecessor",
            ],
            "proposer": type(active_proposer).__name__,
            "test_is_used_only_after_selection": True,
        },
        "config": {
            "iterations": iterations,
            "seed": seed,
            "alpha": alpha,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "budget_fraction": budget_fraction,
            "epsilon": epsilon,
        },
        "sizes": {"train": len(train_records), "val": len(val_records), "test": len(test_records)},
        "final_harness": current_harness,
        "versions": versions,
        "proposals": proposals,
        "comparisons": comparisons,
        "validation": current_eval,
        "test": final_test,
    }


def summarize_trajectory_feedback(
    records: list[ActionRecord],
    gate: TrainedGate,
    harness: dict[str, Any],
    *,
    round_index: int,
    budget_fraction: float,
) -> TrajectoryFeedback:
    probabilities = gate.predict_proba(records)
    failure_counts = {
        "confidently_wrong": 0,
        "overconfident": 0,
        "calibration_error": 0,
        "evidence_conflict": 0,
        "high_ood": 0,
        "high_cost": 0,
        "over_abstention": 0,
    }
    examples: list[dict[str, Any]] = []
    for record, probability in zip(records, probabilities):
        predicted = int(probability >= gate.threshold)
        wrong = predicted != record.label
        feature = record.features
        if wrong and probability >= max(gate.threshold, 0.7):
            failure_counts["confidently_wrong"] += 1
            if len(examples) < 12:
                examples.append(_feedback_example(record, probability, "confidently_wrong"))
        if probability >= 0.8 and abs(probability - record.label) >= 0.3:
            failure_counts["overconfident"] += 1
        if abs(probability - record.label) >= 0.35:
            failure_counts["calibration_error"] += 1
        if feature.get("evidence_conflict", 0.0) >= 0.45:
            failure_counts["evidence_conflict"] += 1
        if feature.get("ood_score", 0.0) >= 0.65:
            failure_counts["high_ood"] += 1
        if feature.get("cost", 0.0) >= 0.65:
            failure_counts["high_cost"] += 1
        if predicted == 0 and record.label == 1:
            failure_counts["over_abstention"] += 1

    evaluated = evaluate_gate(records, gate, budget_fraction=budget_fraction)
    return TrajectoryFeedback(
        round_index=round_index,
        harness_name=str(harness.get("name", "unknown")),
        n_records=len(records),
        failure_counts=failure_counts,
        metric_snapshot={str(k): float(v) for k, v in evaluated["metrics"].items()},
        examples=examples,
    )


def compare_harnesses(
    previous: dict[str, Any],
    candidate: dict[str, Any],
    *,
    epsilon: float,
    iteration: int,
    previous_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    previous_metrics = previous["metrics"]
    candidate_metrics = candidate["metrics"]
    score_previous = _score(previous)
    score_candidate = _score(candidate)
    coverage_gain = float(candidate_metrics["coverage"]) - float(previous_metrics["coverage"])
    risk_gain = float(previous_metrics["selective_risk"]) - float(candidate_metrics["selective_risk"])
    brier_gain = float(previous_metrics["brier"]) - float(candidate_metrics["brier"])
    candidate_not_worse = coverage_gain >= -0.05 and risk_gain >= -0.01 and brier_gain >= -0.01
    winner = "candidate" if candidate_not_worse and score_candidate + epsilon < score_previous else "previous"
    return {
        "iteration": iteration,
        "previous": previous_name,
        "candidate": candidate_name,
        "winner": winner,
        "previous_score": score_previous,
        "candidate_score": score_candidate,
        "score_gain": score_previous - score_candidate,
        "coverage_gain": coverage_gain,
        "selective_risk_gain": risk_gain,
        "brier_gain": brier_gain,
        "rationale": "candidate accepted only if composite uncertainty/action score improves while coverage and risk guards hold",
    }


def _score(report: dict[str, Any]) -> float:
    metrics = report["metrics"]
    return (
        float(metrics.get("selective_risk", 1.0))
        + float(metrics.get("brier", 1.0))
        + float(metrics.get("ece", 1.0))
        + float(metrics.get("log_loss", 1.0))
        + 0.25 * (1.0 - float(metrics.get("coverage", 0.0)))
    )


def _feedback_example(record: ActionRecord, probability: float, failure: str) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "benchmark": record.benchmark,
        "failure": failure,
        "predicted_probability": round(float(probability), 6),
        "label": record.label,
        "action_type": record.action_type,
        "features": dict(record.features),
        "candidate_action": record.candidate_action[:500],
    }


def _version_report(version: int, harness: dict[str, Any], evaluation: dict[str, Any], gate: TrainedGate) -> dict[str, Any]:
    return {
        "version": version,
        "harness": copy.deepcopy(harness),
        "gate": gate.to_json(),
        "validation": evaluation,
    }


def _train_with_features(
    train_records: list[ActionRecord],
    val_records: list[ActionRecord],
    *,
    feature_names: list[str],
    alpha: float,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> TrainedGate:
    from .paper_bootstrap import train_gate_with_features

    return train_gate_with_features(
        train_records,
        val_records,
        feature_names=feature_names,
        alpha=alpha,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )


def _upsert_role(
    roles: list[dict[str, Any]],
    role_id: str,
    instruction: str,
    contract: list[str],
    *,
    kind: str = "adviser",
) -> None:
    for role in roles:
        if role.get("id") == role_id:
            role["instruction"] = instruction
            role["contract"] = list(dict.fromkeys(role.get("contract", []) + contract))
            return
    roles.append({"id": role_id, "kind": kind, "instruction": instruction, "contract": contract})


def _build_hops(harness: dict[str, Any], failure_counts: dict[str, int]) -> list[dict[str, Any]]:
    hops = [
        {"from": "orchestrator", "to": "evidence_auditor", "purpose": "audit evidence before action promotion"},
        {"from": "evidence_auditor", "to": "uncertainty_gate", "purpose": "pass structured uncertainty contract"},
        {"from": "uncertainty_gate", "to": "fallback_router", "purpose": "route action by calibrated reliability and cost"},
    ]
    if failure_counts.get("evidence_conflict", 0) or failure_counts.get("high_ood", 0):
        hops.append({"from": "fallback_router", "to": "evidence_auditor", "purpose": "retrieve and re-audit conflicting or OOD evidence"})
    if failure_counts.get("high_cost", 0):
        hops.append({"from": "fallback_router", "to": "ask_expert", "purpose": "review irreversible or expensive action"})
    if failure_counts.get("over_abstention", 0):
        hops.append({"from": "fallback_router", "to": "uncertainty_gate", "purpose": "recalibrate cheap reversible actions before abstention"})
    return hops


def _updated_alpha(harness: dict[str, Any], feedback: TrajectoryFeedback) -> float:
    current = float(harness.get("target_selective_risk", 0.1))
    if feedback.failure_counts.get("confidently_wrong", 0) > 0:
        return max(0.03, current - 0.02)
    if feedback.failure_counts.get("over_abstention", 0) > feedback.n_records * 0.3:
        return min(0.2, current + 0.02)
    return current


def _proposal_prompt(previous: dict[str, Any], feedback: TrajectoryFeedback, iteration: int) -> str:
    return "\n".join(
        [
            "You are the RHI proposer for a materials-science action-worthiness harness.",
            "Return only a JSON object that preserves the harness schema.",
            "Modify roles, required_features, gates, target_selective_risk, and hops only when feedback justifies it.",
            f"Iteration: {iteration}",
            "Previous harness:",
            harness_text(previous),
            "Trajectory feedback:",
            json.dumps(feedback.to_json(), sort_keys=True, indent=2),
        ]
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM proposer output does not contain a JSON object")
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM proposer output must be a JSON object")
    return payload
