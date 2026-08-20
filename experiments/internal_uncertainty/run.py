from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from harness_matsci.benchmarks import make_records
from harness_matsci.logit_uncertainty import CausalLMLogitJudge, calibrated_confidence, execute_probability
from harness_matsci.metrics import discovery_gain
from harness_matsci.metrics import expected_calibration_error
from harness_matsci.voi import VOI_FEATURES, VOI_SEED_HARNESS, evaluate_voi, fit_voi_model, train_voi_rhi


def make_prompt(record) -> str:
    evidence = "; ".join(record.evidence) if record.evidence else "none"
    return (
        "You are judging whether a proposed scientific action is worth executing now. "
        "A means EXECUTE the proposed action. B means DEFER or STOP. "
        "Use only the visible context, evidence, and candidate action. "
        "Choose the safer scientific decision.\n\n"
        f"Context: {record.visible_context}\n"
        f"Evidence: {evidence}\n"
        f"Candidate action: {record.candidate_action}\n"
        "Decision:"
    )


def calibration_metrics(records, margins, temperature: float) -> dict[str, float]:
    labels = [record.label for record in records]
    predictions = [int(margin >= 0.0) for margin in margins]
    confidence = [calibrated_confidence(margin, temperature) for margin in margins]
    correctness = [int(prediction == label) for prediction, label in zip(predictions, labels)]
    brier = mean((score - target) ** 2 for score, target in zip(confidence, correctness))
    ece = expected_calibration_error(correctness, confidence)
    risk_order = sorted(range(len(records)), key=lambda index: confidence[index], reverse=True)
    budget = max(1, int(len(records) * 0.1))
    selected = risk_order[:budget]
    return {
        "accuracy": mean(predictions[index] == labels[index] for index in range(len(labels))),
        "mean_confidence": mean(confidence),
        "ece_on_correctness": ece,
        "brier_on_correctness": brier,
        "risk_at_10_percent": mean(1 - correctness[index] for index in selected),
        "coverage_at_10_percent": len(selected) / len(records),
        "temperature": temperature,
    }


def ranking_metrics(records, margins, temperature: float) -> dict[str, float]:
    scores = [execute_probability(margin, temperature) for margin in margins]
    labels = [record.label for record in records]
    utilities = [record.utility for record in records]
    budget = max(1, int(len(records) * 0.1))
    return discovery_gain(labels, utilities, scores, budget)


def fit_temperature(records, margins) -> float:
    best_temperature = 1.0
    best_brier = float("inf")
    labels = [int((margin >= 0.0) == bool(record.label)) for record, margin in zip(records, margins)]
    for step in range(1, 401):
        temperature = 0.05 + step * 0.025
        scores = [calibrated_confidence(margin, temperature) for margin in margins]
        brier = mean((score - label) ** 2 for score, label in zip(scores, labels))
        if brier < best_brier:
            best_brier = brier
            best_temperature = temperature
    return best_temperature


def split_validation(records, seed: int = 1729):
    validation = [record for record in records if record.split == "val"]
    rng = random.Random(seed)
    shuffled = list(validation)
    rng.shuffle(shuffled)
    midpoint = max(1, len(shuffled) // 2)
    return shuffled[:midpoint], shuffled[midpoint:]


def permute_internal_features(records, seed: int):
    rng = random.Random(seed)
    groups = {}
    for record in records:
        groups.setdefault(record.benchmark, []).append(record)
    output = []
    for group in groups.values():
        values = [
            (
                record.features["internal_logit_margin"],
                record.features["llm_execute_probability"],
                record.features["llm_decision_confidence"],
                record.features["llm_decision_uncertainty"],
            )
            for record in group
        ]
        rng.shuffle(values)
        for record, value in zip(group, values):
            features = dict(record.features)
            features["internal_logit_margin"] = value[0]
            features["llm_execute_probability"] = value[1]
            features["llm_decision_confidence"] = value[2]
            features["llm_decision_uncertainty"] = value[3]
            features["internal_logit_confidence"] = value[2]
            features["internal_logit_uncertainty"] = value[3]
            output.append(replace(record, features=features))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--n-per-task", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out", default="runs/internal_uncertainty_v1/summary.json")
    args = parser.parse_args()

    records = []
    for benchmark in ("preferential_bo", "discover_unique", "extreme_properties"):
        records.extend(make_records(benchmark, n=args.n_per_task, seed=args.seed))

    judge = CausalLMLogitJudge(args.model, temperature=1.0)
    margins = judge.score([make_prompt(record) for record in records], batch_size=4)
    margins_by_id = {record.record_id: margin for record, margin in zip(records, margins)}
    train_records = [record for record in records if record.split == "train"]
    feedback_records, acceptance_records = split_validation(records, args.seed)
    test_records = [record for record in records if record.split == "test"]
    validation_records = feedback_records + acceptance_records
    calibration_temperature = fit_temperature(
        feedback_records,
        [margins_by_id[record.record_id].margin for record in feedback_records],
    )

    augmented = []
    for record in records:
        margin = margins_by_id[record.record_id]
        features = dict(record.features)
        features.update(
            {
                "internal_logit_margin": margin.margin,
                "llm_execute_probability": execute_probability(margin.margin, calibration_temperature),
                "llm_decision_confidence": calibrated_confidence(margin.margin, calibration_temperature),
                "llm_decision_uncertainty": 1.0 - calibrated_confidence(margin.margin, calibration_temperature),
                "internal_logit_confidence": calibrated_confidence(margin.margin, calibration_temperature),
                "internal_logit_uncertainty": 1.0 - calibrated_confidence(margin.margin, calibration_temperature),
            }
        )
        augmented.append(replace(record, features=features))
    by_id = {record.record_id: record for record in augmented}
    train_records = [by_id[record.record_id] for record in train_records]
    feedback_records = [by_id[record.record_id] for record in feedback_records]
    acceptance_records = [by_id[record.record_id] for record in acceptance_records]
    test_records = [by_id[record.record_id] for record in test_records]

    internal_metrics = calibration_metrics(
        test_records,
        [margins_by_id[record.record_id].margin for record in test_records],
        calibration_temperature,
    )
    results = {
        "protocol": {
            "model": args.model,
            "task_families": ["preferential_bo", "discover_unique", "extreme_properties"],
            "records": len(records),
            "train": len(train_records),
            "feedback": len(feedback_records),
            "acceptance": len(acceptance_records),
            "test": len(test_records),
            "candidate_tokens": {"A": "EXECUTE", "B": "DEFER_OR_STOP"},
            "temperature_fit": "feedback split only",
            "test_access": "candidate-token logits only; no hidden labels or utilities",
        },
        "internal_logit_uncertainty": internal_metrics,
        "direct_llm_signal": ranking_metrics(
            test_records,
            [margins_by_id[record.record_id].margin for record in test_records],
            calibration_temperature,
        ),
    }

    baseline_harness = copy.deepcopy(VOI_SEED_HARNESS)
    baseline_model = fit_voi_model(train_records, feedback_records, baseline_harness, epochs=80)
    results["local_voi"] = evaluate_voi(test_records, baseline_model, budget_fraction=0.1)
    results["local_voi_rhi"] = train_voi_rhi(
        train_records,
        feedback_records,
        acceptance_records,
        test_records,
        iterations=3,
        seed=args.seed,
        epochs=80,
        component="full",
        acceptance_policy="robust_guarded",
        initial_harness=baseline_harness,
    )

    logit_harness = copy.deepcopy(VOI_SEED_HARNESS)
    for feature in (
        "llm_execute_probability",
        "llm_decision_confidence",
        "llm_decision_uncertainty",
        "internal_logit_confidence",
        "internal_logit_uncertainty",
        "internal_logit_margin",
    ):
        if feature not in logit_harness["required_features"]:
            logit_harness["required_features"].append(feature)
        if feature not in logit_harness["utility_features"]:
            logit_harness["utility_features"].append(feature)
    logit_harness["name"] = "H0_internal_logit_voi"
    logit_harness["decision_mode"] = "voi"
    logit_harness["epistemic_weight"] = 0.35
    logit_harness["failure_cost_weight"] = 0.25
    logit_harness["execute_cost_weight"] = 0.15
    logit_harness["verification_cost_weight"] = 0.10
    logit_harness["verification_uncertainty_floor"] = 0.25
    logit_model = fit_voi_model(train_records, feedback_records, logit_harness, epochs=80)
    results["llm_harness"] = evaluate_voi(test_records, logit_model, budget_fraction=0.1)

    permuted = permute_internal_features(train_records + feedback_records + acceptance_records + test_records, args.seed + 99)
    permuted_by_id = {record.record_id: record for record in permuted}
    permuted_model = fit_voi_model(
        [permuted_by_id[record.record_id] for record in train_records],
        [permuted_by_id[record.record_id] for record in feedback_records],
        logit_harness,
        epochs=80,
    )
    results["logit_voi_within_task_permutation"] = evaluate_voi(
        [permuted_by_id[record.record_id] for record in test_records],
        permuted_model,
        budget_fraction=0.1,
    )

    results["logit_voi_rhi"] = train_voi_rhi(
        train_records,
        feedback_records,
        acceptance_records,
        test_records,
        iterations=3,
        seed=args.seed,
        epochs=80,
        component="full",
        acceptance_policy="robust_guarded",
        initial_harness=logit_harness,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "model": args.model,
        "records": len(records),
        "internal_logit_uncertainty": internal_metrics,
        "local_voi_score": results["local_voi"]["oracle_normalized_net_utility"],
        "local_voi_rhi_score": results["local_voi_rhi"]["test"]["oracle_normalized_net_utility"],
        "llm_harness_score": results["llm_harness"]["oracle_normalized_net_utility"],
        "permuted_logit_voi_score": results["logit_voi_within_task_permutation"]["oracle_normalized_net_utility"],
        "logit_voi_rhi_score": results["logit_voi_rhi"]["test"]["oracle_normalized_net_utility"],
        "out": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
