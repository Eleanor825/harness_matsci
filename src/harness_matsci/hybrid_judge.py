from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .io import read_jsonl, write_json
from .metrics import action_worthiness_score
from .schema import ActionRecord
from .training import evaluate_probability_signal, evidence_heuristic_scores, verbal_confidence_scores
from .voi import VOI_SEED_HARNESS, fit_voi_model


@dataclass(frozen=True)
class HybridJudgeConfig:
    records_path: str
    judge_cache_path: str
    out_dir: str
    seed: int = 1729
    local_train_fraction: float = 0.68
    alpha: float = 0.1
    budget_fraction: float = 0.1
    epochs: int = 80
    learning_rate: float = 0.08
    l2: float = 0.01
    blend_step: float = 0.05
    adaptive_call_fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def run_hybrid_judge_experiment(config: HybridJudgeConfig) -> dict[str, Any]:
    records = read_jsonl(config.records_path)
    scores = _load_direct_judge_scores(config.judge_cache_path)
    missing = [record.record_id for record in records if record.record_id not in scores]
    if missing:
        raise ValueError(f"direct judge cache is missing {len(missing)} scores; first missing={missing[0]}")

    validation_records = [record for record in records if record.split == "validation"]
    test_records = [record for record in records if record.split == "test"]
    if not validation_records or not test_records:
        raise ValueError("records must contain non-empty validation and test splits")
    local_train, local_calibration = _stratified_split(
        validation_records,
        train_fraction=config.local_train_fraction,
        seed=config.seed,
    )

    harness = _hybrid_ready_harness()
    local_model = fit_voi_model(
        local_train,
        local_calibration,
        harness,
        alpha=config.alpha,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        l2=config.l2,
    )
    scored_groups = _score_groups(local_model, scores, local_calibration, test_records)
    all_records = list(local_calibration) + list(test_records)

    methods: dict[str, dict[str, Any]] = {}
    methods["llm_direct_judge"] = _evaluate_method(
        local_calibration,
        test_records,
        scored_groups["llm"],
        alpha=config.alpha,
        budget_fraction=config.budget_fraction,
    )
    methods["llm_direct_judge"]["test_llm_call_rate"] = 1.0
    methods["local_voi_harness"] = _evaluate_method(
        local_calibration,
        test_records,
        scored_groups["local"],
        alpha=config.alpha,
        budget_fraction=config.budget_fraction,
    )
    methods["local_voi_harness"]["test_llm_call_rate"] = 0.0
    methods["verbal_confidence"] = _evaluate_score_fn(
        local_calibration,
        test_records,
        verbal_confidence_scores,
        alpha=config.alpha,
        budget_fraction=config.budget_fraction,
    )
    methods["verbal_confidence"]["test_llm_call_rate"] = 0.0
    methods["evidence_heuristic"] = _evaluate_score_fn(
        local_calibration,
        test_records,
        evidence_heuristic_scores,
        alpha=config.alpha,
        budget_fraction=config.budget_fraction,
    )
    methods["evidence_heuristic"]["test_llm_call_rate"] = 0.0

    blend_selection = _select_static_blend(
        local_calibration,
        scored_groups["local"],
        scored_groups["llm"],
        alpha=config.alpha,
        budget_fraction=config.budget_fraction,
        step=config.blend_step,
    )
    static_blend_scores = _static_blend_scores(scored_groups["local"], scored_groups["llm"], blend_selection["weight"])
    methods["hybrid_static_blend"] = _evaluate_method(
        local_calibration,
        test_records,
        static_blend_scores,
        alpha=config.alpha,
        budget_fraction=config.budget_fraction,
    )
    methods["hybrid_static_blend"]["selection"] = blend_selection
    methods["hybrid_static_blend"]["test_llm_call_rate"] = 1.0

    guarded_selection = _select_guarded_blend(
        local_calibration,
        scored_groups["local"],
        scored_groups["llm"],
        llm_floor=float(methods["llm_direct_judge"]["threshold"]),
        alpha=config.alpha,
        budget_fraction=config.budget_fraction,
        step=config.blend_step,
    )
    guarded_scores = _guarded_blend_scores(
        scored_groups["local"],
        scored_groups["llm"],
        weight=guarded_selection["weight"],
        llm_floor=guarded_selection["llm_floor"],
    )
    methods["hybrid_llm_guarded_blend"] = _evaluate_method(
        local_calibration,
        test_records,
        guarded_scores,
        alpha=config.alpha,
        budget_fraction=config.budget_fraction,
    )
    methods["hybrid_llm_guarded_blend"]["selection"] = guarded_selection
    methods["hybrid_llm_guarded_blend"]["test_llm_call_rate"] = 1.0

    adaptive_selection = _select_adaptive_router(
        local_calibration,
        scored_groups["local"],
        scored_groups["llm"],
        scored_groups["need_llm"],
        alpha=config.alpha,
        budget_fraction=config.budget_fraction,
        call_fractions=config.adaptive_call_fractions,
        step=config.blend_step,
    )
    adaptive_scores, call_rate = _adaptive_router_scores(
        all_records,
        scored_groups["local"],
        scored_groups["llm"],
        scored_groups["need_llm"],
        weight=adaptive_selection["weight"],
        threshold=adaptive_selection["need_llm_threshold"],
    )
    methods["hybrid_adaptive_router"] = _evaluate_method(
        local_calibration,
        test_records,
        adaptive_scores,
        alpha=config.alpha,
        budget_fraction=config.budget_fraction,
    )
    methods["hybrid_adaptive_router"]["selection"] = adaptive_selection
    test_call_rate = _adaptive_router_scores(
        test_records,
        scored_groups["local"],
        scored_groups["llm"],
        scored_groups["need_llm"],
        weight=adaptive_selection["weight"],
        threshold=adaptive_selection["need_llm_threshold"],
    )[1]
    methods["hybrid_adaptive_router"]["test_llm_call_rate"] = test_call_rate

    budgeted_router_selections: dict[str, dict[str, float]] = {}
    for call_fraction in (0.3, 0.5):
        name = f"hybrid_adaptive_{int(call_fraction * 100)}pct_llm"
        selection = _select_adaptive_router(
            local_calibration,
            scored_groups["local"],
            scored_groups["llm"],
            scored_groups["need_llm"],
            alpha=config.alpha,
            budget_fraction=config.budget_fraction,
            call_fractions=(call_fraction,),
            step=config.blend_step,
        )
        scores_for_budget, _ = _adaptive_router_scores(
            all_records,
            scored_groups["local"],
            scored_groups["llm"],
            scored_groups["need_llm"],
            weight=selection["weight"],
            threshold=selection["need_llm_threshold"],
        )
        report_for_budget = _evaluate_method(
            local_calibration,
            test_records,
            scores_for_budget,
            alpha=config.alpha,
            budget_fraction=config.budget_fraction,
        )
        budget_call_rate = _adaptive_router_scores(
            test_records,
            scored_groups["local"],
            scored_groups["llm"],
            scored_groups["need_llm"],
            weight=selection["weight"],
            threshold=selection["need_llm_threshold"],
        )[1]
        report_for_budget["selection"] = selection
        report_for_budget["test_llm_call_rate"] = budget_call_rate
        methods[name] = report_for_budget
        budgeted_router_selections[name] = selection

    return {
        "method_summary": _compact_methods(methods),
        "config": asdict(config),
        "split": {
            "records_total": len(records),
            "validation": len(validation_records),
            "test": len(test_records),
            "local_train": len(local_train),
            "local_calibration": len(local_calibration),
        },
        "method_details": methods,
        "selected_parameters": {
            "static_blend": blend_selection,
            "llm_guarded_blend": guarded_selection,
            "adaptive_router": adaptive_selection,
            "budgeted_adaptive_routers": budgeted_router_selections,
        },
        "interpretation": {
            "hybrid_static_blend": "Convexly blends the local VoI harness score and direct LLM judge score; the blend weight is selected on local_calibration only.",
            "hybrid_llm_guarded_blend": "Uses the LLM judge as a calibrated safety floor, then uses the VoI score to improve ranking within the LLM-approved set.",
            "hybrid_adaptive_router": "Uses the local VoI harness by default and blends in the LLM judge only for high-uncertainty/high-risk actions, estimating the fraction of runtime LLM calls.",
            "budgeted_adaptive_routers": "Fixed-call-budget variants show the cost-quality trade-off when the LLM judge cannot be called on every action.",
            "caveat": "This is a 500-record cached GPT-5.5 subset experiment, not the full 15,717-record held-out-regime protocol.",
        },
    }


def save_hybrid_judge_experiment(report: dict[str, Any], out_dir: str | Path) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    write_json(report, path / "summary.json")
    (path / "README.md").write_text(_markdown_report(report), encoding="utf-8")


def _hybrid_ready_harness() -> dict[str, Any]:
    harness = copy.deepcopy(VOI_SEED_HARNESS)
    harness.update(
        {
            "name": "H_hybrid_ready_voi",
            "decision_mode": "voi",
            "execute_cost_weight": 0.15,
            "failure_cost_weight": 0.20,
            "epistemic_weight": 0.20,
            "verification_cost_weight": 0.05,
            "verification_uncertainty_floor": 0.20,
            "verification_support_weight": 0.15,
            "min_execute_reliability": 0.50,
            "allow_verification": True,
        }
    )
    return harness


def _load_direct_judge_scores(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("scores", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, dict):
        raise ValueError("direct judge cache must be a JSON object or contain a 'scores' object")
    return {str(key): float(value) for key, value in values.items()}


def _stratified_split(
    records: list[ActionRecord],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[list[ActionRecord], list[ActionRecord]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    groups: dict[tuple[str, int], list[ActionRecord]] = {}
    for record in records:
        groups.setdefault((record.benchmark, record.label), []).append(record)
    rng = random.Random(seed)
    train: list[ActionRecord] = []
    calibration: list[ActionRecord] = []
    for key in sorted(groups):
        rows = list(groups[key])
        rng.shuffle(rows)
        if len(rows) == 1:
            train.extend(rows)
            continue
        n_train = max(1, min(len(rows) - 1, round(len(rows) * train_fraction)))
        train.extend(rows[:n_train])
        calibration.extend(rows[n_train:])
    rng.shuffle(train)
    rng.shuffle(calibration)
    if not train or not calibration:
        raise ValueError("stratified split produced an empty train or calibration set")
    return train, calibration


def _score_groups(
    local_model: Any,
    llm_scores: dict[str, float],
    calibration_records: list[ActionRecord],
    test_records: list[ActionRecord],
) -> dict[str, dict[str, float]]:
    all_records = list(calibration_records) + list(test_records)
    local_predictions = local_model.predict(all_records)
    raw_local = {record.record_id: prediction.action_score for record, prediction in zip(all_records, local_predictions)}
    normalizer = _score_normalizer([raw_local[record.record_id] for record in calibration_records])
    local = {record_id: normalizer(score) for record_id, score in raw_local.items()}
    epistemic_raw = {record.record_id: prediction.epistemic_uncertainty for record, prediction in zip(all_records, local_predictions)}
    epistemic_norm = _score_normalizer([epistemic_raw[record.record_id] for record in calibration_records])
    need_llm: dict[str, float] = {}
    for record in all_records:
        local_score = local[record.record_id]
        confidence_gap = abs(local_score - 0.5) * 2.0
        feature_risk = max(
            float(record.features.get("ood_score", 0.0)),
            float(record.features.get("model_disagreement", 0.0)),
            float(record.features.get("evidence_conflict", 0.0)),
        )
        need_llm[record.record_id] = _clamp(
            0.45 * epistemic_norm(epistemic_raw[record.record_id])
            + 0.35 * (1.0 - confidence_gap)
            + 0.20 * feature_risk
        )
    return {
        "local": local,
        "llm": {record.record_id: _clamp(llm_scores[record.record_id]) for record in all_records},
        "need_llm": need_llm,
    }


def _score_normalizer(values: list[float]):
    lo = min(values) if values else 0.0
    hi = max(values) if values else 1.0
    if hi <= lo:
        return lambda value: 0.5
    return lambda value: _clamp((float(value) - lo) / (hi - lo))


def _select_static_blend(
    calibration_records: list[ActionRecord],
    local_scores: dict[str, float],
    llm_scores: dict[str, float],
    *,
    alpha: float,
    budget_fraction: float,
    step: float,
) -> dict[str, float]:
    best: dict[str, float] | None = None
    for weight in _weight_grid(step):
        scores = _static_blend_scores(local_scores, llm_scores, weight)
        report = _evaluate_method(calibration_records, calibration_records, scores, alpha=alpha, budget_fraction=budget_fraction)
        candidate = {
            "weight": weight,
            "calibration_score": float(report["score"]),
            "calibration_risk_at_10": float(report["risk_at_10"]),
            "calibration_hit_rate": float(report["hit_rate"]),
            "calibration_utility_efficiency": float(report["utility_efficiency"]),
        }
        if best is None or candidate["calibration_score"] < best["calibration_score"]:
            best = candidate
    if best is None:
        raise ValueError("failed to select a blend weight")
    return best


def _select_guarded_blend(
    calibration_records: list[ActionRecord],
    local_scores: dict[str, float],
    llm_scores: dict[str, float],
    *,
    llm_floor: float,
    alpha: float,
    budget_fraction: float,
    step: float,
) -> dict[str, float]:
    best: dict[str, float] | None = None
    for weight in _weight_grid(step):
        scores = _guarded_blend_scores(local_scores, llm_scores, weight=weight, llm_floor=llm_floor)
        report = _evaluate_method(calibration_records, calibration_records, scores, alpha=alpha, budget_fraction=budget_fraction)
        candidate = {
            "weight": weight,
            "llm_floor": llm_floor,
            "calibration_score": float(report["score"]),
            "calibration_risk_at_10": float(report["risk_at_10"]),
            "calibration_hit_rate": float(report["hit_rate"]),
            "calibration_utility_efficiency": float(report["utility_efficiency"]),
        }
        if best is None or candidate["calibration_score"] < best["calibration_score"]:
            best = candidate
    if best is None:
        raise ValueError("failed to select a guarded blend weight")
    return best



def _select_adaptive_router(
    calibration_records: list[ActionRecord],
    local_scores: dict[str, float],
    llm_scores: dict[str, float],
    need_llm_scores: dict[str, float],
    *,
    alpha: float,
    budget_fraction: float,
    call_fractions: tuple[float, ...],
    step: float,
) -> dict[str, float]:
    best: dict[str, float] | None = None
    for call_fraction in call_fractions:
        threshold = _threshold_for_call_fraction(calibration_records, need_llm_scores, call_fraction)
        for weight in _weight_grid(step):
            scores, call_rate = _adaptive_router_scores(
                calibration_records,
                local_scores,
                llm_scores,
                need_llm_scores,
                weight=weight,
                threshold=threshold,
            )
            report = _evaluate_method(calibration_records, calibration_records, scores, alpha=alpha, budget_fraction=budget_fraction)
            candidate = {
                "weight": weight,
                "target_llm_call_fraction": call_fraction,
                "need_llm_threshold": threshold,
                "calibration_llm_call_rate": call_rate,
                "calibration_score": float(report["score"]),
                "calibration_risk_at_10": float(report["risk_at_10"]),
                "calibration_hit_rate": float(report["hit_rate"]),
                "calibration_utility_efficiency": float(report["utility_efficiency"]),
            }
            if best is None or candidate["calibration_score"] < best["calibration_score"]:
                best = candidate
    if best is None:
        raise ValueError("failed to select adaptive router parameters")
    return best


def _evaluate_method(
    calibration_records: list[ActionRecord],
    test_records: list[ActionRecord],
    scores: dict[str, float],
    *,
    alpha: float,
    budget_fraction: float,
) -> dict[str, Any]:
    return _evaluate_score_fn(
        calibration_records,
        test_records,
        lambda records: [scores[record.record_id] for record in records],
        alpha=alpha,
        budget_fraction=budget_fraction,
    )


def _evaluate_score_fn(
    calibration_records: list[ActionRecord],
    test_records: list[ActionRecord],
    score_fn,
    *,
    alpha: float,
    budget_fraction: float,
) -> dict[str, Any]:
    report = evaluate_probability_signal(
        calibration_records,
        test_records,
        score_fn,
        alpha=alpha,
        min_coverage=0.1,
        budget_fraction=budget_fraction,
        balance_benchmarks=True,
    )
    test = report["test"]
    metrics = test["metrics"]
    gain = test["discovery_gain"]
    risk_at_10 = test["risk_coverage"]["risk_at_0.10"]["selective_risk"]
    return {
        "score": action_worthiness_score(metrics, gain),
        "selective_risk": metrics["selective_risk"],
        "coverage": metrics["coverage"],
        "risk_at_10": risk_at_10,
        "hit_rate": gain["hit_rate"],
        "mean_utility": gain["mean_utility"],
        "utility_efficiency": gain["utility_efficiency"],
        "ece": metrics["ece"],
        "brier": metrics["brier"],
        "threshold": report["threshold"],
        "raw_report": report,
    }


def _static_blend_scores(local_scores: dict[str, float], llm_scores: dict[str, float], weight: float) -> dict[str, float]:
    return {
        record_id: _clamp((1.0 - weight) * local_scores[record_id] + weight * llm_scores[record_id])
        for record_id in local_scores
    }


def _guarded_blend_scores(
    local_scores: dict[str, float],
    llm_scores: dict[str, float],
    *,
    weight: float,
    llm_floor: float,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for record_id in local_scores:
        blend = _clamp((1.0 - weight) * local_scores[record_id] + weight * llm_scores[record_id])
        scores[record_id] = blend if llm_scores[record_id] >= llm_floor else 0.0
    return scores


def _adaptive_router_scores(
    records: list[ActionRecord],
    local_scores: dict[str, float],
    llm_scores: dict[str, float],
    need_llm_scores: dict[str, float],
    *,
    weight: float,
    threshold: float,
) -> tuple[dict[str, float], float]:
    scores: dict[str, float] = {}
    calls = 0
    for record in records:
        record_id = record.record_id
        if need_llm_scores[record_id] >= threshold:
            calls += 1
            scores[record_id] = _clamp((1.0 - weight) * local_scores[record_id] + weight * llm_scores[record_id])
        else:
            scores[record_id] = local_scores[record_id]
    return scores, calls / len(records) if records else 0.0


def _threshold_for_call_fraction(records: list[ActionRecord], need_llm_scores: dict[str, float], call_fraction: float) -> float:
    if not 0 < call_fraction <= 1:
        raise ValueError("call_fraction must be in (0, 1]")
    values = sorted((need_llm_scores[record.record_id] for record in records), reverse=True)
    index = max(0, min(len(values) - 1, math.ceil(len(values) * call_fraction) - 1))
    return values[index]


def _weight_grid(step: float) -> list[float]:
    if not 0 < step <= 1:
        raise ValueError("step must be in (0, 1]")
    n = round(1.0 / step)
    return [round(min(1.0, index * step), 10) for index in range(n + 1)]


def _compact_methods(methods: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = [
        "score",
        "risk_at_10",
        "hit_rate",
        "mean_utility",
        "utility_efficiency",
        "ece",
        "brier",
        "coverage",
        "selective_risk",
    ]
    result: dict[str, dict[str, float]] = {}
    for name, report in methods.items():
        result[name] = {key: float(report.get(key, 0.0)) for key in keys}
        if "test_llm_call_rate" in report:
            result[name]["test_llm_call_rate"] = float(report["test_llm_call_rate"])
    return result


def _markdown_report(report: dict[str, Any]) -> str:
    rows = report["method_summary"]
    ordered = sorted(rows, key=lambda item: rows[item]["score"])
    lines = [
        "# Hybrid LLM + VoI Judge Subset Experiment",
        "",
        "This run evaluates whether a direct LLM judge can be combined with the local Sci-VoI harness at decision time.",
        "The result uses the cached GPT-5.5 direct judge scores from the 500-record subset; no new LLM calls are made by this run.",
        "",
        "## Protocol",
        "",
        f"- Records: `{report['split']['records_total']}` total; test `{report['split']['test']}`.",
        f"- Local VoI train/calibration split: `{report['split']['local_train']}` / `{report['split']['local_calibration']}` from the validation records.",
        f"- Fixed action budget: top `{report['config']['budget_fraction']:.0%}` actions.",
        f"- Target selective-risk alpha: `{report['config']['alpha']}`.",
        "- Score is the existing action-worthiness diagnostic; lower is better.",
        "",
        "## Results",
        "",
        "| Method | Score ↓ | Risk@10% ↓ | Hit ↑ | Mean utility ↑ | Utility eff. ↑ | ECE ↓ | LLM call rate ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ordered:
        row = rows[name]
        call_rate = row.get("test_llm_call_rate", 0.0)
        lines.append(
            f"| `{name}` | {row['score']:.4f} | {row['risk_at_10']:.4f} | {row['hit_rate']:.4f} | "
            f"{row['mean_utility']:.4f} | {row['utility_efficiency']:.4f} | {row['ece']:.4f} | {call_rate:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Selected Hybrid Parameters",
            "",
            f"- Static blend LLM weight: `{report['selected_parameters']['static_blend']['weight']:.2f}`.",
            f"- LLM-guarded blend weight: `{report['selected_parameters']['llm_guarded_blend']['weight']:.2f}` with floor `{report['selected_parameters']['llm_guarded_blend']['llm_floor']:.4f}`.",
            f"- Adaptive router LLM weight: `{report['selected_parameters']['adaptive_router']['weight']:.2f}`.",
            f"- Adaptive router target LLM call fraction: `{report['selected_parameters']['adaptive_router']['target_llm_call_fraction']:.2f}`.",
            f"- Adaptive router observed test LLM call rate: `{report['method_summary']['hybrid_adaptive_router'].get('test_llm_call_rate', 0.0):.4f}`.",
            "",
            "## Interpretation",
            "",
            "- `llm_direct_judge` is a one-shot GPT-5.5 action-worthiness score.",
            "- `local_voi_harness` is the local reliability/utility/uncertainty harness without LLM semantics.",
            "- `hybrid_static_blend` always combines both signals.",
            "- `hybrid_llm_guarded_blend` treats the LLM judge as a safety floor and lets VoI rerank actions that pass the floor.",
            "- `hybrid_adaptive_router` uses the local harness by default and consults the LLM judge only on high-uncertainty/high-risk actions.",
            "- `hybrid_adaptive_30pct_llm` and `hybrid_adaptive_50pct_llm` force an approximate LLM-call budget to expose the cost-quality trade-off.",
            "- This is a subset experiment; the next step is to run the same hybrid protocol on the full held-out-regime benchmark and with Claude-family judges.",
            "",
        ]
    )
    return "\n".join(lines)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run hybrid LLM + local VoI judge experiment on a cached direct-judge subset.")
    parser.add_argument("--records", required=True, help="Action records JSONL with validation/test splits")
    parser.add_argument("--judge-cache", required=True, help="Direct judge cache JSON containing record_id -> score")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--local-train-fraction", type=float, default=0.68)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--budget-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.01)
    args = parser.parse_args(argv)
    config = HybridJudgeConfig(
        records_path=args.records,
        judge_cache_path=args.judge_cache,
        out_dir=args.out_dir,
        seed=args.seed,
        local_train_fraction=args.local_train_fraction,
        alpha=args.alpha,
        budget_fraction=args.budget_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    report = run_hybrid_judge_experiment(config)
    save_hybrid_judge_experiment(report, args.out_dir)
    best = min(report["method_summary"], key=lambda key: report["method_summary"][key]["score"])
    print(f"wrote hybrid judge experiment to {args.out_dir}; best={best}; score={report['method_summary'][best]['score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
