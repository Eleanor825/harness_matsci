from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import asdict, dataclass, field
from statistics import mean, pstdev
from typing import Any

from .calibration import threshold_for_selective_risk
from .features import Standardizer, feature_names_from_records
from .models import LogisticGate
from .schema import ActionRecord
from .training import TrainedGate, evaluate_gate, train_gate_with_features


VOI_FEATURES = [
    "verbal_confidence",
    "evidence_support",
    "evidence_conflict",
    "source_reliability",
    "tool_agreement",
    "model_disagreement",
    "perturbation_stability",
    "ood_score",
    "cost",
    "reversibility",
    "action_complexity",
    "evidence_count",
]


VOI_SEED_HARNESS: dict[str, Any] = {
    "name": "H0_reliability_only",
    "roles": [
        {"id": "evidence_auditor", "kind": "adviser", "contract": ["evidence_support", "evidence_conflict"]},
        {"id": "uncertainty_gate", "kind": "builder", "contract": ["p_success", "threshold"]},
        {"id": "fallback_router", "kind": "reviewer", "contract": ["route_reason"]},
    ],
    "required_features": list(VOI_FEATURES),
    "utility_features": list(VOI_FEATURES),
    "gates": ["Only proceed when calibrated reliability exceeds threshold."],
    "hops": [
        {"from": "orchestrator", "to": "evidence_auditor", "purpose": "audit visible evidence"},
        {"from": "evidence_auditor", "to": "uncertainty_gate", "purpose": "estimate reliability"},
        {"from": "uncertainty_gate", "to": "fallback_router", "purpose": "select route"},
    ],
    "execute_cost_weight": 0.0,
    "failure_cost_weight": 0.0,
    "epistemic_weight": 0.0,
    "verification_cost_weight": 0.0,
    "min_execute_reliability": 0.0,
    "verification_uncertainty_floor": 1.0,
    "verification_support_weight": 0.0,
    "allow_verification": True,
    "use_cost_signal": True,
    "use_uncertainty_signal": True,
    "decision_mode": "reliability",
}


@dataclass(frozen=True)
class UtilityNormalizer:
    minimum: float
    maximum: float

    def encode(self, value: float) -> float:
        if self.maximum <= self.minimum:
            return 0.5
        return max(0.0, min(1.0, (float(value) - self.minimum) / (self.maximum - self.minimum)))

    def to_json(self) -> dict[str, float]:
        return {"minimum": self.minimum, "maximum": self.maximum}


@dataclass
class LinearUtilityModel:
    feature_names: list[str]
    weights: list[float]
    bias: float

    def predict_row(self, row: list[float]) -> float:
        value = self.bias + sum(weight * value for weight, value in zip(self.weights, row))
        return max(0.0, min(1.0, value))

    def predict(self, rows: list[list[float]]) -> list[float]:
        return [self.predict_row(row) for row in rows]

    def to_json(self) -> dict[str, Any]:
        return {"feature_names": list(self.feature_names), "weights": list(self.weights), "bias": self.bias}


@dataclass(frozen=True)
class VoIPrediction:
    p_success: float
    expected_utility: float
    epistemic_uncertainty: float
    execute_value: float
    verify_value: float
    action_score: float
    decision: str
    route: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VoIModel:
    harness: dict[str, Any]
    reliability: TrainedGate
    utility_models: list[LinearUtilityModel]
    standardizer: Standardizer
    normalizer: UtilityNormalizer

    def predict(self, records: list[ActionRecord]) -> list[VoIPrediction]:
        names = self.utility_models[0].feature_names
        rows = self.standardizer.transform(
            [[float(record.features.get(name, 0.0)) for name in names] for record in records]
        )
        probabilities = self.reliability.predict_proba(records)
        ensemble = [model.predict(rows) for model in self.utility_models]
        predictions: list[VoIPrediction] = []
        for index, record in enumerate(records):
            utilities = [values[index] for values in ensemble]
            expected = sum(utilities) / len(utilities)
            epistemic = pstdev(utilities) if len(utilities) > 1 else 0.0
            features = record.features
            use_cost_signal = bool(self.harness.get("use_cost_signal", True))
            use_uncertainty_signal = bool(self.harness.get("use_uncertainty_signal", True))
            allow_verification = bool(self.harness.get("allow_verification", True))
            cost = max(0.0, min(1.0, float(features.get("cost", 0.5)))) if use_cost_signal else 0.0
            support = max(0.0, min(1.0, float(features.get("evidence_support", 0.5))))
            conflict = max(0.0, min(1.0, float(features.get("evidence_conflict", 0.0))))
            ood = max(0.0, min(1.0, float(features.get("ood_score", 0.0))))
            p_success = max(0.0, min(1.0, probabilities[index]))
            failure = 1.0 - p_success
            decision_mode = str(self.harness.get("decision_mode", "reliability"))
            execute_value = expected - float(self.harness.get("execute_cost_weight", 0.0)) * cost
            execute_value -= float(self.harness.get("failure_cost_weight", 0.0)) * failure
            if not use_uncertainty_signal:
                epistemic = 0.0
            execute_value -= float(self.harness.get("epistemic_weight", 0.0)) * epistemic
            reducibility = max(
                0.0,
                min(
                    1.0,
                    0.5 * conflict
                    + 0.3 * ood
                    + float(self.harness.get("verification_support_weight", 0.0)) * (1.0 - support),
                ),
            )
            verify_value = (
                float(self.harness.get("epistemic_weight", 0.0)) * epistemic * (0.5 + 0.5 * reducibility)
                - float(self.harness.get("verification_cost_weight", 0.0)) * cost
            )
            if decision_mode == "reliability":
                score = p_success
            elif decision_mode == "utility":
                score = execute_value
            else:
                score = max(execute_value, verify_value)
            min_reliability = max(
                float(self.harness.get("min_execute_reliability", 0.0)),
                float(self.reliability.threshold),
            )
            uncertainty_floor = float(self.harness.get("verification_uncertainty_floor", 1.0))
            if decision_mode == "reliability":
                decision = "execute" if p_success >= min_reliability else "stop"
                route = "experiment" if record.action_type in {"experiment", "recommend_experiment"} else "proceed"
            elif execute_value > 0.0 and p_success >= min_reliability:
                decision = "execute"
                route = "experiment" if record.action_type in {"experiment", "recommend_experiment"} else "proceed"
            elif allow_verification and verify_value > max(0.0, execute_value) and epistemic >= uncertainty_floor:
                decision = "verify"
                route = _verification_route(record, cost)
            else:
                decision = "stop"
                route = "abstain"
            predictions.append(
                VoIPrediction(
                    p_success=p_success,
                    expected_utility=expected,
                    epistemic_uncertainty=epistemic,
                    execute_value=execute_value,
                    verify_value=verify_value,
                    action_score=score,
                    decision=decision,
                    route=route,
                )
            )
        return predictions

    def to_json(self) -> dict[str, Any]:
        return {
            "harness": copy.deepcopy(self.harness),
            "reliability": self.reliability.to_json(),
            "utility_models": [model.to_json() for model in self.utility_models],
            "standardizer": self.standardizer.to_json(),
            "normalizer": self.normalizer.to_json(),
        }


def fit_voi_model(
    train_records: list[ActionRecord],
    feedback_records: list[ActionRecord],
    harness: dict[str, Any],
    *,
    alpha: float = 0.1,
    epochs: int = 160,
    learning_rate: float = 0.08,
    l2: float = 0.01,
    ensemble_size: int = 3,
) -> VoIModel:
    if not train_records or not feedback_records:
        raise ValueError("VoI fitting requires non-empty train and feedback records")
    reliability_features = list(harness.get("required_features", VOI_FEATURES))
    utility_features = list(harness.get("utility_features", reliability_features))
    reliability = train_gate_with_features(
        train_records,
        feedback_records,
        feature_names=reliability_features,
        alpha=alpha,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        min_coverage=0.1,
    )
    normalizer = UtilityNormalizer(
        minimum=min(record.utility for record in train_records),
        maximum=max(record.utility for record in train_records),
    )
    standardizer = Standardizer().fit(
        [[float(record.features.get(name, 0.0)) for name in utility_features] for record in train_records]
    )
    x_train = standardizer.transform(
        [[float(record.features.get(name, 0.0)) for name in utility_features] for record in train_records]
    )
    targets = [normalizer.encode(record.utility) for record in train_records]
    utility_models = []
    for member in range(max(1, ensemble_size)):
        selected = [
            index
            for index in range(len(train_records))
            if int(hashlib.sha256(f"voi|{member}|{train_records[index].record_id}".encode()).hexdigest()[:8], 16) % max(2, ensemble_size) != member - 1
        ]
        if len(selected) < max(4, len(train_records) // 3):
            selected = list(range(len(train_records)))
        utility_models.append(
            _fit_ridge_linear(
                utility_features,
                [x_train[index] for index in selected],
                [targets[index] for index in selected],
                l2=l2,
            )
        )
    return VoIModel(harness=copy.deepcopy(harness), reliability=reliability, utility_models=utility_models, standardizer=standardizer, normalizer=normalizer)


def evaluate_voi(
    records: list[ActionRecord],
    model: VoIModel,
    *,
    budget_fraction: float = 0.1,
    evaluation_cost_weight: float = 0.15,
) -> dict[str, Any]:
    predictions = model.predict(records)
    scores = [prediction.action_score for prediction in predictions]
    probabilities = [prediction.p_success for prediction in predictions]
    selected = sorted(range(len(records)), key=lambda index: scores[index], reverse=True)[: max(1, int(len(records) * budget_fraction))]
    actual_utility = [model.normalizer.encode(record.utility) for record in records]
    outcome_conditioned_utility = [value * float(record.label) for value, record in zip(actual_utility, records)]
    net_utility = [
        utility - evaluation_cost_weight * max(0.0, min(1.0, record.features.get("cost", 0.5)))
        for utility, record in zip(actual_utility, records)
    ]
    outcome_conditioned_net_utility = [
        utility - evaluation_cost_weight * max(0.0, min(1.0, record.features.get("cost", 0.5)))
        for utility, record in zip(outcome_conditioned_utility, records)
    ]
    oracle = sorted(range(len(records)), key=lambda index: net_utility[index], reverse=True)[: len(selected)]
    selected_net = mean(net_utility[index] for index in selected) if selected else 0.0
    oracle_net = mean(net_utility[index] for index in oracle) if oracle else 0.0
    outcome_selected = mean(outcome_conditioned_net_utility[index] for index in selected) if selected else 0.0
    outcome_oracle_order = sorted(range(len(records)), key=lambda index: outcome_conditioned_net_utility[index], reverse=True)[: len(selected)]
    outcome_oracle = mean(outcome_conditioned_net_utility[index] for index in outcome_oracle_order) if outcome_oracle_order else 0.0
    selected_labels = [records[index].label for index in selected]
    selected_regimes = {str(records[index].metadata.get("group_id", records[index].benchmark)) for index in selected}
    all_regimes = {str(record.metadata.get("group_id", record.benchmark)) for record in records}
    executed = [index for index, prediction in enumerate(predictions) if prediction.decision == "execute"]
    executed_wrong = sum(1 - records[index].label for index in executed)
    return {
        "n_records": len(records),
        "budget": len(selected),
        "scores": scores,
        "predictions": [prediction.to_json() for prediction in predictions],
        "selected_indices": selected,
        "selected_record_ids": [records[index].record_id for index in selected],
        "selected_utility": selected_net,
        "oracle_utility": oracle_net,
        "oracle_normalized_net_utility": selected_net / oracle_net if oracle_net > 0 else 0.0,
        "outcome_conditioned_oracle_normalized_net_utility": outcome_selected / outcome_oracle if outcome_oracle > 0 else 0.0,
        "hit_rate": mean(selected_labels) if selected_labels else 0.0,
        "coverage": len(executed) / len(records) if records else 0.0,
        "execute_selective_risk": executed_wrong / len(executed) if executed else 0.0,
        "confidently_wrong_execute_rate": executed_wrong / len(records) if records else 0.0,
        "verify_rate": sum(prediction.decision == "verify" for prediction in predictions) / len(records) if records else 0.0,
        "stop_rate": sum(prediction.decision == "stop" for prediction in predictions) / len(records) if records else 0.0,
        "regime_coverage": len(selected_regimes) / len(all_regimes) if all_regimes else 0.0,
        "mean_epistemic_uncertainty": mean(prediction.epistemic_uncertainty for prediction in predictions) if predictions else 0.0,
        "reliability": evaluate_gate(records, model.reliability, budget_fraction=budget_fraction),
    }


def summarize_voi_feedback(records: list[ActionRecord], model: VoIModel, *, budget_fraction: float) -> dict[str, Any]:
    evaluation = evaluate_voi(records, model, budget_fraction=budget_fraction)
    predictions = evaluation["predictions"]
    selected = set(evaluation["selected_indices"])
    failure_counts = {
        "confidently_wrong": sum(
            1 for index, prediction in enumerate(predictions) if prediction["decision"] == "execute" and records[index].label == 0
        ),
        "low_utility_selected": sum(
            1 for index in selected if records[index].utility < model.normalizer.minimum
        ),
        "high_epistemic": sum(
            1 for prediction in predictions if prediction["epistemic_uncertainty"] > 0.05
        ),
        "verification": sum(prediction["decision"] == "verify" for prediction in predictions),
        "over_abstention": sum(
            1 for index, prediction in enumerate(predictions) if prediction["decision"] == "stop" and records[index].label == 1
        ),
    }
    return {
        "n_records": len(records),
        "failure_counts": failure_counts,
        "metric_snapshot": {
            "oracle_normalized_net_utility": float(evaluation["oracle_normalized_net_utility"]),
            "execute_selective_risk": float(evaluation["execute_selective_risk"]),
            "coverage": float(evaluation["coverage"]),
            "verify_rate": float(evaluation["verify_rate"]),
        },
    }


def propose_voi_harness(
    previous: dict[str, Any],
    feedback: dict[str, Any],
    *,
    iteration: int,
    component: str = "full",
    available_features: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Produce a deterministic, trajectory-conditioned executable mutation.

    Unlike the original offline RHI proposer, every mutation below changes a
    quantity consumed by ``VoIModel.predict``.  The JSON roles, contracts, and
    hops are retained as the auditable agent-facing specification.
    """
    if component not in {"full", "utility", "uncertainty", "routing", "features"}:
        raise ValueError("unknown VoI mutation component")
    candidate = copy.deepcopy(previous)
    candidate["name"] = f"H{iteration}_scivoi_{component}"
    counts = feedback.get("failure_counts", {})
    rationale: list[str] = []
    can_utility = component in {"full", "utility"}
    can_uncertainty = component in {"full", "uncertainty"}
    can_routing = component in {"full", "routing"}
    can_features = component in {"full", "features"}
    observed = list(available_features or [])
    if can_utility:
        candidate["decision_mode"] = "utility"
        if observed:
            candidate["utility_features"] = list(dict.fromkeys(list(candidate.get("utility_features", [])) + observed))
        candidate["execute_cost_weight"] = max(0.15, float(previous.get("execute_cost_weight", 0.0)))
        candidate["failure_cost_weight"] = max(
            0.20 if counts.get("confidently_wrong", 0) else 0.12,
            float(previous.get("failure_cost_weight", 0.0)),
        )
        rationale.append("Replace confidence-only ranking with a cost- and failure-aware expected utility contract.")
    if can_uncertainty:
        candidate["decision_mode"] = "voi"
        candidate["epistemic_weight"] = max(0.20, float(previous.get("epistemic_weight", 0.0)))
        candidate["verification_uncertainty_floor"] = min(
            0.20, max(0.005, float(previous.get("verification_uncertainty_floor", 1.0)) * 0.25)
        )
        rationale.append("Use ensemble disagreement as epistemic uncertainty and reserve verification for reducible uncertainty.")
    if can_routing:
        candidate["verification_cost_weight"] = max(0.05, float(previous.get("verification_cost_weight", 0.0)))
        candidate["verification_support_weight"] = max(0.15, float(previous.get("verification_support_weight", 0.0)))
        candidate["min_execute_reliability"] = max(0.50, float(previous.get("min_execute_reliability", 0.0)))
        rationale.append("Make the fallback router executable: retrieve, simulate, or ask according to cost and action type.")
    if can_features:
        required = list(candidate.get("required_features", VOI_FEATURES))
        for feature_name in (
            "source_reliability",
            "tool_agreement",
            "model_disagreement",
            "cost",
            "reversibility",
            *observed,
        ):
            if feature_name not in required:
                required.append(feature_name)
        candidate["required_features"] = required
        candidate["utility_features"] = list(dict.fromkeys(list(candidate.get("utility_features", [])) + required))
        rationale.append("Expose provenance, agreement, cost, and reversibility in the action contract.")
    if can_utility or can_features:
        _upsert_role(
            candidate,
            "utility_estimator",
            "Estimate continuous scientific utility before treating an action as worth executing.",
            ["expected_utility", "cost", "failure_penalty"],
            kind="builder",
        )
    if can_uncertainty or can_routing:
        _upsert_role(
            candidate,
            "verification_router",
            "Estimate reducible epistemic uncertainty and route verification before an irreversible action.",
            ["epistemic_uncertainty", "verification_value", "route_reason"],
            kind="reviewer",
        )
        candidate["hops"] = list(candidate.get("hops", [])) + [
            {"from": "uncertainty_gate", "to": "verification_router", "purpose": "estimate value of information"},
            {"from": "verification_router", "to": "fallback_router", "purpose": "execute or verify based on net value"},
        ]
    candidate["gates"] = list(dict.fromkeys(list(candidate.get("gates", [])) + [
        "Do not execute a costly action when conservative net value is non-positive.",
        "Use verification only when epistemic uncertainty is reducible and verification has positive value.",
    ]))
    return candidate, rationale


def train_voi_rhi(
    train_records: list[ActionRecord],
    feedback_records: list[ActionRecord],
    acceptance_records: list[ActionRecord],
    test_records: list[ActionRecord],
    *,
    iterations: int = 3,
    seed: int = 7,
    alpha: float = 0.1,
    budget_fraction: float = 0.1,
    epochs: int = 160,
    learning_rate: float = 0.08,
    l2: float = 0.01,
    epsilon: float = 0.005,
    component: str = "full",
    acceptance_policy: str = "robust_guarded",
    initial_harness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if acceptance_policy not in {"robust_guarded", "mean_guarded", "always_accept", "never_accept"}:
        raise ValueError("unknown acceptance policy")
    if not train_records or not feedback_records or not acceptance_records or not test_records:
        raise ValueError("all VoI-RHI partitions must be non-empty")
    seed_harness = copy.deepcopy(initial_harness or VOI_SEED_HARNESS)
    current_harness = copy.deepcopy(seed_harness)
    available_features = feature_names_from_records(train_records + feedback_records)
    current_model = fit_voi_model(
        train_records, feedback_records, current_harness, alpha=alpha, epochs=epochs, learning_rate=learning_rate, l2=l2
    )
    current_eval = evaluate_voi(acceptance_records, current_model, budget_fraction=budget_fraction)
    feedback = summarize_voi_feedback(feedback_records, current_model, budget_fraction=budget_fraction)
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    accepted_versions = [current_harness["name"]]
    shards = _group_shards(acceptance_records, iterations, seed)
    for iteration in range(1, iterations + 1):
        candidate_harness, rationale = propose_voi_harness(
            current_harness,
            feedback,
            iteration=iteration,
            component=component,
            available_features=available_features,
        )
        candidate_model = fit_voi_model(
            train_records, feedback_records, candidate_harness, alpha=alpha, epochs=epochs, learning_rate=learning_rate, l2=l2
        )
        candidate_shard = shards[iteration - 1]
        predecessor_shard_eval = evaluate_voi(candidate_shard, current_model, budget_fraction=budget_fraction)
        candidate_shard_eval = evaluate_voi(candidate_shard, candidate_model, budget_fraction=budget_fraction)
        comparison = compare_voi_harnesses(
            current_model,
            candidate_model,
            predecessor_shard_eval,
            candidate_shard_eval,
            epsilon=epsilon,
            robust=acceptance_policy == "robust_guarded",
            all_acceptance=acceptance_records,
            budget_fraction=budget_fraction,
        )
        accepted = acceptance_policy == "always_accept" or (acceptance_policy != "never_accept" and comparison["winner"] == "candidate")
        if accepted:
            current_harness = candidate_harness
            current_model = candidate_model
            current_eval = evaluate_voi(acceptance_records, current_model, budget_fraction=budget_fraction)
            accepted_versions.append(current_harness["name"])
        feedback = summarize_voi_feedback(feedback_records, current_model, budget_fraction=budget_fraction)
        history.append({
            "iteration": iteration,
            "candidate_harness": candidate_harness,
            "rationale": rationale,
            "feedback": feedback,
            "comparison": comparison,
            "accepted": accepted,
            "acceptance_record_ids": [record.record_id for record in candidate_shard],
        })
        checkpoints.append({"round": iteration, "harness": copy.deepcopy(current_harness), "model": current_model.to_json(), "accepted": accepted})
    final_eval = evaluate_voi(test_records, current_model, budget_fraction=budget_fraction)
    initial_model = fit_voi_model(train_records, feedback_records, seed_harness, alpha=alpha, epochs=epochs, learning_rate=learning_rate, l2=l2)
    initial_test = evaluate_voi(test_records, initial_model, budget_fraction=budget_fraction)
    for checkpoint in checkpoints:
        checkpoint_model = fit_model_from_json(checkpoint["model"])
        checkpoint["test"] = evaluate_voi(test_records, checkpoint_model, budget_fraction=budget_fraction)
    return {
        "method": "Sci-VoI-RHI",
        "component": component,
        "acceptance_policy": acceptance_policy,
        "initial_harness": seed_harness,
        "final_harness": current_harness,
        "accepted_versions": accepted_versions,
        "history": history,
        "checkpoints": checkpoints,
        "initial_test": initial_test,
        "test": final_eval,
        "initial_model": initial_model.to_json(),
        "final_model": current_model.to_json(),
    }


def compare_voi_harnesses(
    previous_model: VoIModel,
    candidate_model: VoIModel,
    previous: dict[str, Any],
    candidate: dict[str, Any],
    *,
    epsilon: float,
    robust: bool,
    all_acceptance: list[ActionRecord],
    budget_fraction: float,
) -> dict[str, Any]:
    current = _comparison_values(previous, candidate)
    if robust:
        source_prev = _group_evaluations(
            all_acceptance,
            lambda subset: evaluate_voi(subset, previous_model, budget_fraction=budget_fraction),
        )
        source_candidate = _group_evaluations(
            all_acceptance,
            lambda subset: evaluate_voi(subset, candidate_model, budget_fraction=budget_fraction),
        )
        current["source_mean_gain"] = mean(
            source_candidate[group]["oracle_normalized_net_utility"] - source_prev[group]["oracle_normalized_net_utility"]
            for group in source_prev
        )
        current["source_worst_gain"] = min(
            source_candidate[group]["oracle_normalized_net_utility"] - source_prev[group]["oracle_normalized_net_utility"]
            for group in source_prev
        )
        current["source_regime_gains"] = {
            group: source_candidate[group]["oracle_normalized_net_utility"] - source_prev[group]["oracle_normalized_net_utility"]
            for group in source_prev
        }
        source_gains = list(current["source_regime_gains"].values())
        gain_mean = mean(source_gains) if source_gains else 0.0
        gain_std = pstdev(source_gains) if len(source_gains) > 1 else 0.0
        current["source_lcb_gain"] = gain_mean - 0.5 * gain_std / math.sqrt(max(1, len(source_gains)))
        current["source_loss_rate"] = sum(gain < -0.10 for gain in source_gains) / len(source_gains) if source_gains else 0.0
        safe = current["source_lcb_gain"] >= epsilon and current["source_loss_rate"] <= 0.35 and current["risk_gain"] >= -0.05
    else:
        safe = current["risk_gain"] >= -0.03
    winner = "candidate" if safe and current["utility_gain"] > -0.05 else "previous"
    current.update({
        "winner": winner,
        "candidate_group_count": len(source_candidate) if robust else 0,
        "previous_group_count": len(source_prev) if robust else 0,
        "rationale": "accept only positive utility gain with reliability and worst-regime guards",
    })
    return current


def fit_model_from_json(payload: dict[str, Any]) -> VoIModel:
    reliability = TrainedGate.from_json(payload["reliability"])
    utility_models = [
        LinearUtilityModel(
            feature_names=[str(name) for name in item["feature_names"]],
            weights=[float(value) for value in item["weights"]],
            bias=float(item["bias"]),
        )
        for item in payload["utility_models"]
    ]
    normalizer = UtilityNormalizer(float(payload["normalizer"]["minimum"]), float(payload["normalizer"]["maximum"]))
    return VoIModel(
        harness=copy.deepcopy(payload["harness"]),
        reliability=reliability,
        utility_models=utility_models,
        standardizer=Standardizer.from_json(payload["standardizer"]),
        normalizer=normalizer,
    )


def _comparison_values(previous: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    return {
        "utility_gain": float(candidate["oracle_normalized_net_utility"]) - float(previous["oracle_normalized_net_utility"]),
        "risk_gain": float(previous["execute_selective_risk"]) - float(candidate["execute_selective_risk"]),
        "coverage_gain": float(candidate["coverage"]) - float(previous["coverage"]),
        "hit_rate_gain": float(candidate["hit_rate"]) - float(previous["hit_rate"]),
    }


def _group_evaluations(records: list[ActionRecord], model_or_eval: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    groups = sorted({str(record.metadata.get("group_id", record.benchmark)) for record in records})
    for group in groups:
        subset = [record for record in records if str(record.metadata.get("group_id", record.benchmark)) == group]
        result[group] = model_or_eval(subset) if callable(model_or_eval) else model_or_eval
    return result


def _group_shards(records: list[ActionRecord], iterations: int, seed: int) -> list[list[ActionRecord]]:
    groups: dict[str, list[ActionRecord]] = {}
    for record in records:
        group = str(record.metadata.get("group_id", record.benchmark))
        groups.setdefault(group, []).append(record)
    group_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    if not group_ids:
        return [records for _ in range(max(1, iterations))]
    return [[record for group in group_ids[index::max(1, iterations)] for record in groups[group]] for index in range(max(1, iterations))]



def _fit_ridge_linear(feature_names: list[str], rows: list[list[float]], targets: list[float], *, l2: float) -> LinearUtilityModel:
    dimension = len(feature_names) + 1
    matrix = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    vector = [0.0 for _ in range(dimension)]
    for row, target in zip(rows, targets):
        values = [1.0] + list(row)
        for i in range(dimension):
            vector[i] += values[i] * target
            for j in range(dimension):
                matrix[i][j] += values[i] * values[j]
    for i in range(1, dimension):
        matrix[i][i] += l2
    solution = _solve_linear_system(matrix, vector)
    return LinearUtilityModel(feature_names=list(feature_names), weights=solution[1:], bias=solution[0])


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [list(row) + [value] for row, value in zip(matrix, vector)]
    n = len(vector)
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-10:
            augmented[pivot][column] = 1e-10
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [left - factor * right for left, right in zip(augmented[row], augmented[column])]
    return [augmented[index][-1] for index in range(n)]


def _verification_route(record: ActionRecord, cost: float) -> str:
    if cost >= 0.7:
        return "ask_expert"
    if record.action_type in {"experiment", "recommend_experiment"}:
        return "simulate"
    return "retrieve_more"


def _upsert_role(
    harness: dict[str, Any],
    role_id: str,
    instruction: str,
    contract: list[str],
    *,
    kind: str,
) -> None:
    roles = list(harness.get("roles", []))
    for role in roles:
        if role.get("id") == role_id:
            role["instruction"] = instruction
            role["contract"] = list(dict.fromkeys(list(role.get("contract", [])) + contract))
            harness["roles"] = roles
            return
    roles.append({"id": role_id, "kind": kind, "instruction": instruction, "contract": list(contract)})
    harness["roles"] = roles
