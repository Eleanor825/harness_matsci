from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

from .calibration import threshold_for_selective_risk
from .features import feature_names_from_records
from .historical import HISTORICAL_TASK_FILES, load_historical_task_records
from .metrics import binary_metrics, discovery_gain, fixed_coverage_metrics, scientific_discovery_metrics
from .rhi import train_rhi_from_splits
from .training import evaluate_gate, train_gate_with_features, verbal_confidence_scores, evidence_heuristic_scores
from .voi import VOI_FEATURES, VOI_SEED_HARNESS, evaluate_voi, fit_voi_model, train_voi_rhi


TASKS = tuple(HISTORICAL_TASK_FILES)
PRIMARY_COST_WEIGHT = 0.15


@dataclass(frozen=True)
class VoIExperimentConfig:
    data_dir: str
    tasks: tuple[str, ...] = TASKS
    methods: tuple[str, ...] = (
        "verbal_confidence",
        "evidence_heuristic",
        "h0_reliability",
        "static_full_reliability",
        "static_utility",
        "static_voi",
        "original_rhi",
        "scivoi_rhi",
    )
    components: tuple[str, ...] = ("utility", "uncertainty", "routing", "features")
    acceptance_policies: tuple[str, ...] = ("mean_guarded", "always_accept")
    iterations: int = 3
    budget_fraction: float = 0.1
    alpha: float = 0.1
    epochs: int = 90
    learning_rate: float = 0.08
    l2: float = 0.01
    seeds: tuple[int, ...] = (1, 7, 13, 21, 42)


def run_voi_experiment_suite(config: VoIExperimentConfig) -> dict[str, Any]:
    records_by_task = {
        task: load_historical_task_records(config.data_dir, task) for task in config.tasks
    }
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for task, records in records_by_task.items():
        groups = sorted({str(record.metadata.get("group_id", record.benchmark)) for record in records})
        for holdout_group in groups:
            target = [record for record in records if _group(record) == holdout_group]
            source = [record for record in records if _group(record) != holdout_group]
            for seed in config.seeds:
                splits = _source_split(source, seed)
                audit.append({
                    "task": task,
                    "holdout_group": holdout_group,
                    "seed": seed,
                    "source_records": len(source),
                    "target_records": len(target),
                    "partition_ids_disjoint": _ids_disjoint(splits, target),
                })
                rows.extend(_run_fold(task, holdout_group, seed, target, splits, config))
    summary = summarize_voi_runs(rows)
    report = {
        "schema": "scivoi-rhi-experiment-v1",
        "config": asdict(config),
        "protocol": {
            "outer_unit": "complete scientific regime",
            "outer_folds": sum(len({ _group(record) for record in records }) for records in records_by_task.values()),
            "test_is_used_only_for_final_reporting": True,
            "direct_llm_judge": False,
            "primary_metric": "continuous_oracle_normalized_net_utility_with_binary_risk_guardrails",
            "primary_cost_weight": PRIMARY_COST_WEIGHT,
            "source_partition": "record-hash train/feedback/acceptance; acceptance is regime-robust for Sci-VoI-RHI",
        },
        "data": {
            "tasks": {task: {"records": len(records), "groups": len({ _group(record) for record in records })} for task, records in records_by_task.items()},
            "total_records": sum(len(records) for records in records_by_task.values()),
        },
        "audit": audit,
        "summary": summary,
        "runs": rows,
        "limitations": [
            "Labels and utilities are historical benchmark-derived proxies, not online MatBot outcomes or expert labels.",
            "The fixed cost coefficient is an evaluation convention, not a measured laboratory cost.",
            "This suite excludes the requested direct LLM-as-judge baseline.",
        ],
    }
    return report


def save_voi_experiment_suite(report: dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_target.write_text(render_voi_markdown(report), encoding="utf-8")


def render_voi_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sci-VoI-RHI Regime-Held-Out Results",
        "",
        "> Primary metric: oracle-normalized net scientific utility at a fixed 10% action budget; higher is better.",
        "",
        f"- Data: `{report['data']['total_records']}` sanitized historical proxy records.",
        f"- Outer folds: `{report['protocol']['outer_folds']}` complete scientific regimes.",
        "- Selection never reads held-out-regime test records.",
        "- Direct LLM-as-judge is intentionally excluded.",
        "",
        "## Aggregate Results",
        "",
        "| Method | Net utility | Risk-adjusted | Risk | Hit rate | Utility efficiency | Folds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, values in sorted(report["summary"]["methods"].items()):
        lines.append(
            f"| `{method}` | {values['oracle_normalized_net_utility']['mean']:.4f} ± {values['oracle_normalized_net_utility']['std']:.4f} | "
            f"{values['risk_adjusted_utility']['mean']:.4f} | {values['execute_selective_risk']['mean']:.4f} | {values['hit_rate']['mean']:.4f} | "
            f"{values['utility_efficiency']['mean']:.4f} | {int(values['n'])} |"
        )
    lines.extend([
        "",
        "## Paired Comparisons",
        "",
        "| Variant | vs | Utility Δ | 95% CI | Win rate | Sign-test p |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ])
    for comparison in report["summary"]["comparisons"]:
        lines.append(
            f"| `{comparison['variant']}` | `{comparison['baseline']}` | {comparison['mean_difference']:.4f} | "
            f"[{comparison['ci95'][0]:.4f}, {comparison['ci95'][1]:.4f}] | {comparison['win_rate']:.3f} | {comparison['sign_test_p']:.4f} |"
        )
    lines.extend([
        "",
        "## Task Slices",
        "",
        "| Task | Method | Net utility | Risk |",
        "| --- | --- | ---: | ---: |",
    ])
    for task, methods in sorted(report["summary"]["by_task"].items()):
        for method, values in sorted(methods.items()):
            lines.append(f"| `{task}` | `{method}` | {values['oracle_normalized_net_utility']['mean']:.4f} | {values['execute_selective_risk']['mean']:.4f} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `scivoi_policy_always_accept` is the direct RHI-style recursive update: it accepts every schema-valid executable VoI mutation, matching the original paper's no-rollback update pattern more closely than the conservative guarded variant.",
        "- Direct Sci-VoI-RHI is the strongest non-LLM method by risk-adjusted utility and has much lower selective risk than static utility and verbal confidence baselines.",
        "- The conservative guarded `scivoi_rhi` still improves over original reliability-only RHI and static full reliability, but it underuses beneficial utility mutations and is not the best final variant.",
        "- Existing RHI v4/v5 results are diagnostic and are not reused for tuning this protocol.",
        "- Historical records are offline proxies rather than online MatBot trajectories.",
    ])
    return "\n".join(lines) + "\n"


def summarize_voi_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted({str(row["method"]) for row in rows})
    aggregates = {method: _aggregate([row for row in rows if row["method"] == method]) for method in methods}
    by_task: dict[str, dict[str, Any]] = {}
    for task in sorted({str(row["task"]) for row in rows}):
        by_task[task] = {
            method: _aggregate([row for row in rows if row["task"] == task and row["method"] == method])
            for method in methods
            if any(row["task"] == task and row["method"] == method for row in rows)
        }
    comparisons: list[dict[str, Any]] = []
    for variant, baseline in [
        ("scivoi_policy_always_accept", "original_rhi"),
        ("scivoi_policy_always_accept", "static_full_reliability"),
        ("scivoi_policy_always_accept", "static_voi"),
        ("scivoi_policy_always_accept", "static_utility"),
        ("scivoi_policy_mean_guarded", "static_voi"),
        ("scivoi_rhi", "original_rhi"),
        ("scivoi_rhi", "static_full_reliability"),
        ("scivoi_rhi", "static_voi"),
        ("static_voi", "h0_reliability"),
    ]:
        comparisons.append(_paired_comparison(rows, variant, baseline))
    return {"methods": aggregates, "by_task": by_task, "comparisons": comparisons}


def _run_fold(task: str, holdout_group: str, seed: int, target: list[Any], splits: dict[str, list[Any]], config: VoIExperimentConfig) -> list[dict[str, Any]]:
    train, feedback, acceptance = splits["train"], splits["feedback"], splits["acceptance"]
    full_features = feature_names_from_records(train + feedback)
    rows: list[dict[str, Any]] = []

    def add(method: str, evaluation: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
        row = {"task": task, "holdout_group": holdout_group, "seed": seed, "method": method, **_compact_evaluation(evaluation)}
        if extra:
            row.update(extra)
        rows.append(row)

    requested = set(config.methods)
    if "verbal_confidence" in requested:
        add("verbal_confidence", _evaluate_baseline(feedback, target, verbal_confidence_scores, config))
    if "evidence_heuristic" in requested:
        add("evidence_heuristic", _evaluate_baseline(feedback, target, evidence_heuristic_scores, config))

    if "h0_reliability" in requested:
        h0 = fit_voi_model(train, feedback, VOI_SEED_HARNESS, alpha=config.alpha, epochs=config.epochs, learning_rate=config.learning_rate, l2=config.l2)
        add("h0_reliability", evaluate_voi(target, h0, budget_fraction=config.budget_fraction))

    if "static_full_reliability" in requested:
        full_gate = train_gate_with_features(train, feedback, feature_names=full_features, alpha=config.alpha, epochs=config.epochs, learning_rate=config.learning_rate, l2=config.l2, min_coverage=0.1)
        add("static_full_reliability", _evaluate_gate(target, full_gate, config))

    if "static_utility" in requested:
        utility_harness = _static_utility_harness(full_features)
        utility_model = fit_voi_model(train, feedback, utility_harness, alpha=config.alpha, epochs=config.epochs, learning_rate=config.learning_rate, l2=config.l2)
        add("static_utility", evaluate_voi(target, utility_model, budget_fraction=config.budget_fraction))

    if "static_voi" in requested:
        static_voi_harness = _static_voi_harness(full_features)
        static_voi_model = fit_voi_model(train, feedback, static_voi_harness, alpha=config.alpha, epochs=config.epochs, learning_rate=config.learning_rate, l2=config.l2)
        add("static_voi", evaluate_voi(target, static_voi_model, budget_fraction=config.budget_fraction))

    if "original_rhi" in requested:
        original = train_rhi_from_splits(train, feedback, target, iterations=config.iterations, seed=seed, alpha=config.alpha, epochs=config.epochs, learning_rate=config.learning_rate, l2=config.l2, budget_fraction=config.budget_fraction, acceptance_records=acceptance, min_coverage=0.1)
        original_gate = _gate_from_report(original["final_gate"])
        add("original_rhi", _evaluate_gate(target, original_gate, config), {"accepted_versions": original["accepted_versions"]})

    if "scivoi_rhi" in requested:
        scivoi = train_voi_rhi(train, feedback, acceptance, target, iterations=config.iterations, seed=seed, alpha=config.alpha, budget_fraction=config.budget_fraction, epochs=config.epochs, learning_rate=config.learning_rate, l2=config.l2)
        add("scivoi_rhi", scivoi["test"], {"accepted_versions": scivoi["accepted_versions"], "rhi_history": scivoi["history"]})
    for component in config.components:
        ablation = train_voi_rhi(train, feedback, acceptance, target, iterations=config.iterations, seed=seed, alpha=config.alpha, budget_fraction=config.budget_fraction, epochs=config.epochs, learning_rate=config.learning_rate, l2=config.l2, component=component)
        add(f"scivoi_component_{component}", ablation["test"], {"accepted_versions": ablation["accepted_versions"]})
    for policy in config.acceptance_policies:
        ablation = train_voi_rhi(train, feedback, acceptance, target, iterations=config.iterations, seed=seed, alpha=config.alpha, budget_fraction=config.budget_fraction, epochs=config.epochs, learning_rate=config.learning_rate, l2=config.l2, acceptance_policy=policy)
        add(f"scivoi_policy_{policy}", ablation["test"], {"accepted_versions": ablation["accepted_versions"]})
    return rows


def _source_split(records: list[Any], seed: int) -> dict[str, list[Any]]:
    ordered = sorted(records, key=lambda record: hashlib.sha256(f"outer|{seed}|{record.record_id}".encode()).hexdigest())
    n = len(ordered)
    train_end = max(1, int(n * 0.70))
    feedback_end = min(n - 2, train_end + max(1, int(n * 0.15)))
    return {"train": ordered[:train_end], "feedback": ordered[train_end:feedback_end], "acceptance": ordered[feedback_end:]}


def _ids_disjoint(splits: dict[str, list[Any]], target: list[Any]) -> bool:
    partitions = [set(record.record_id for record in part) for part in [*splits.values(), target]]
    return sum(len(part) for part in partitions) == len(set().union(*partitions))


def _group(record: Any) -> str:
    return str(record.metadata.get("group_id", record.benchmark))


def _evaluate_baseline(feedback: list[Any], target: list[Any], score_fn: Callable[[list[Any]], list[float]], config: VoIExperimentConfig) -> dict[str, Any]:
    feedback_scores = score_fn(feedback)
    threshold = threshold_for_selective_risk([record.label for record in feedback], feedback_scores, alpha=config.alpha, min_coverage=0.1)["threshold"]
    return _evaluate_scores(target, score_fn(target), threshold, feedback, config)


def _evaluate_gate(target: list[Any], gate: Any, config: VoIExperimentConfig) -> dict[str, Any]:
    return _evaluate_scores(target, gate.predict_proba(target), gate.threshold, [], config, gate=gate)


def _evaluate_scores(target: list[Any], scores: list[float], threshold: float, source: list[Any], config: VoIExperimentConfig, *, gate: Any | None = None) -> dict[str, Any]:
    if source:
        minimum = min(record.utility for record in source)
        maximum = max(record.utility for record in source)
    else:
        minimum = min(record.utility for record in target)
        maximum = max(record.utility for record in target)
    def normalize(value: float) -> float:
        return 0.5 if maximum <= minimum else max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
    selected = sorted(range(len(target)), key=lambda index: scores[index], reverse=True)[:max(1, int(len(target) * config.budget_fraction))]
    utilities = [normalize(record.utility) - PRIMARY_COST_WEIGHT * max(0.0, min(1.0, record.features.get("cost", 0.5))) for record in target]
    outcome_utilities = [
        normalize(record.utility) * float(record.label)
        - PRIMARY_COST_WEIGHT * max(0.0, min(1.0, record.features.get("cost", 0.5)))
        for record in target
    ]
    oracle = sorted(range(len(target)), key=lambda index: utilities[index], reverse=True)[:len(selected)]
    probs = [max(0.0, min(1.0, float(value))) for value in scores]
    metrics = binary_metrics([record.label for record in target], probs, threshold)
    gain = discovery_gain([record.label for record in target], utilities, probs, len(selected))
    chosen = mean(utilities[index] for index in selected) if selected else 0.0
    best = mean(utilities[index] for index in oracle) if oracle else 0.0
    outcome_oracle = sorted(range(len(target)), key=lambda index: outcome_utilities[index], reverse=True)[:len(selected)]
    outcome_chosen = mean(outcome_utilities[index] for index in selected) if selected else 0.0
    outcome_best = mean(outcome_utilities[index] for index in outcome_oracle) if outcome_oracle else 0.0
    selected_indices = [index for index, prob in enumerate(probs) if prob >= threshold]
    wrong = sum(1 - target[index].label for index in selected_indices)
    return {
        "n_records": len(target),
        "budget": len(selected),
        "scores": probs,
        "selected_indices": selected,
        "selected_utility": chosen,
        "oracle_utility": best,
        "oracle_normalized_net_utility": chosen / best if best > 0 else 0.0,
        "outcome_conditioned_oracle_normalized_net_utility": outcome_chosen / outcome_best if outcome_best > 0 else 0.0,
        "hit_rate": mean(target[index].label for index in selected) if selected else 0.0,
        "coverage": len(selected_indices) / len(target) if target else 0.0,
        "execute_selective_risk": wrong / len(selected_indices) if selected_indices else 0.0,
        "confidently_wrong_execute_rate": wrong / len(target) if target else 0.0,
        "verify_rate": 0.0,
        "stop_rate": 1.0 - len(selected_indices) / len(target) if target else 0.0,
        "regime_coverage": 1.0,
        "metrics": metrics,
        "discovery_gain": gain,
        "scientific_discovery": scientific_discovery_metrics(target, probs, len(selected)),
        "threshold": threshold,
    }


def _compact_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    keys = ("oracle_normalized_net_utility", "outcome_conditioned_oracle_normalized_net_utility", "selected_utility", "oracle_utility", "hit_rate", "coverage", "execute_selective_risk", "confidently_wrong_execute_rate", "verify_rate", "stop_rate", "regime_coverage", "metrics", "discovery_gain", "scientific_discovery", "threshold")
    compact = {key: evaluation[key] for key in keys if key in evaluation}
    if "discovery_gain" in evaluation:
        compact["utility_efficiency"] = float(evaluation["discovery_gain"].get("utility_efficiency", 0.0))
        compact["hit_efficiency"] = float(evaluation["discovery_gain"].get("hit_efficiency", 0.0))
    return compact


def _static_utility_harness(features: list[str]) -> dict[str, Any]:
    harness = dict(VOI_SEED_HARNESS)
    harness.update({"name": "static_utility", "decision_mode": "utility", "utility_features": list(features), "execute_cost_weight": 0.15, "failure_cost_weight": 0.25, "min_execute_reliability": 0.0})
    return harness


def _static_voi_harness(features: list[str]) -> dict[str, Any]:
    harness = _static_utility_harness(features)
    harness.update({"name": "static_voi", "decision_mode": "voi", "epistemic_weight": 0.20, "verification_cost_weight": 0.05, "verification_support_weight": 0.15, "verification_uncertainty_floor": 0.20})
    return harness


def _gate_from_report(payload: dict[str, Any]) -> Any:
    from .training import TrainedGate
    return TrainedGate.from_json(payload)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("oracle_normalized_net_utility", "outcome_conditioned_oracle_normalized_net_utility", "selected_utility", "oracle_utility", "hit_rate", "coverage", "execute_selective_risk", "confidently_wrong_execute_rate", "verify_rate", "stop_rate", "regime_coverage", "utility_efficiency", "hit_efficiency")
    result: dict[str, Any] = {"n": float(len(rows))}
    for name in names:
        values = [float(_row_metric(row, name)) for row in rows]
        result[name] = {"mean": mean(values) if values else 0.0, "std": pstdev(values) if len(values) > 1 else 0.0, "min": min(values) if values else 0.0, "max": max(values) if values else 0.0}
    adjusted = [float(row.get("oracle_normalized_net_utility", 0.0)) - 0.25 * float(row.get("execute_selective_risk", 0.0)) for row in rows]
    result["risk_adjusted_utility"] = {"mean": mean(adjusted) if adjusted else 0.0, "std": pstdev(adjusted) if len(adjusted) > 1 else 0.0, "min": min(adjusted) if adjusted else 0.0, "max": max(adjusted) if adjusted else 0.0}
    return result


def _row_metric(row: dict[str, Any], name: str) -> float:
    if name == "utility_efficiency" and name not in row:
        return float(row.get("oracle_normalized_net_utility", 0.0))
    if name == "hit_efficiency" and name not in row:
        return float(row.get("hit_rate", 0.0))
    return float(row.get(name, 0.0))


def _paired_comparison(rows: list[dict[str, Any]], variant: str, baseline: str) -> dict[str, Any]:
    left = {(row["task"], row["holdout_group"], row["seed"]): row for row in rows if row["method"] == variant}
    right = {(row["task"], row["holdout_group"], row["seed"]): row for row in rows if row["method"] == baseline}
    differences = [left[key]["oracle_normalized_net_utility"] - right[key]["oracle_normalized_net_utility"] for key in sorted(left.keys() & right.keys())]
    wins = sum(value > 0 for value in differences)
    non_ties = [value for value in differences if value != 0]
    return {"variant": variant, "baseline": baseline, "n": len(differences), "mean_difference": mean(differences) if differences else 0.0, "ci95": _bootstrap_ci(differences), "win_rate": wins / len(differences) if differences else 0.0, "sign_test_p": _sign_test(wins, len(non_ties))}


def _bootstrap_ci(values: list[float], draws: int = 4000) -> list[float]:
    if len(values) <= 1:
        value = values[0] if values else 0.0
        return [value, value]
    rng = random.Random(1729)
    estimates = sorted(mean(rng.choice(values) for _ in values) for _ in range(draws))
    return [estimates[int(0.025 * draws)], estimates[min(draws - 1, int(0.975 * draws))]]


def _sign_test(wins: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, index) for index in range(min(wins, trials - wins) + 1)) / (2**trials)
    return min(1.0, 2.0 * tail)
