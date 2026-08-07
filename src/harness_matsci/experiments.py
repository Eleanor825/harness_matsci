from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

from .benchmarks import make_records
from .direct_judge import LLMDirectJudge
from .features import DEFAULT_FEATURES, feature_names_from_records
from .historical import grouped_four_way_split, load_historical_task_records, random_four_way_split
from .io import write_json
from .metrics import (
    action_worthiness_score,
    binary_metrics,
    discovery_gain,
    fixed_coverage_metrics,
    scientific_discovery_metrics,
)
from .rhi import RHI_SEED_FEATURES, train_rhi_from_splits
from .schema import ActionRecord
from .training import (
    TrainedGate,
    evidence_heuristic_scores,
    evaluate_gate,
    evaluate_probability_signal,
    train_gate_with_features,
    verbal_confidence_scores,
)

TASKS = ("preferential_bo", "discover_unique", "extreme_properties")
TASK_LABELS = {
    "preferential_bo": "Pairwise optimization",
    "discover_unique": "Unique-material discovery",
    "extreme_properties": "Extreme-property discovery",
}
CORE_METHODS = (
    "rhi",
    "non_rhi_seed",
    "static_full",
    "verbal_confidence",
    "evidence_heuristic",
    "llm_direct_judge",
)
TRANSFER_REFERENCE = "target_supervised_reference"


@dataclass(frozen=True)
class ExperimentSuiteConfig:
    tasks: tuple[str, ...] = TASKS
    experiments: tuple[int, ...] = (1, 2, 3)
    seeds: tuple[int, ...] = (1, 7, 13, 21, 42)
    n_per_task: int = 300
    data_dir: str | None = None
    train_fraction: float = 0.6
    val_fraction: float = 0.2
    feedback_fraction: float = 0.15
    acceptance_fraction: float = 0.1
    alpha: float = 0.1
    min_coverage: float = 0.1
    budget_fraction: float = 0.1
    rhi_iterations: int = 3
    epochs: int = 240
    learning_rate: float = 0.08
    l2: float = 0.001
    rhi_epsilon: float = 0.01
    direct_judge_model: str | None = None
    direct_judge_base_url: str | None = None
    direct_judge_cache: str | None = None
    direct_judge_timeout: float = 90.0
    direct_judge_retries: int = 3


@dataclass(frozen=True)
class RunRecord:
    experiment: str
    setting: str
    seed: int
    task: str
    method: str
    split: dict[str, int]
    source_tasks: list[str] = field(default_factory=list)
    holdout_task: str = ""
    report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_experiment_suite(config: ExperimentSuiteConfig, *, direct_judge: Any | None = None) -> dict[str, Any]:
    _validate_config(config)
    if direct_judge is None and config.direct_judge_model:
        direct_judge = LLMDirectJudge.from_env(
            model=config.direct_judge_model,
            base_url=config.direct_judge_base_url,
            cache_path=config.direct_judge_cache,
            timeout=config.direct_judge_timeout,
            max_retries=config.direct_judge_retries,
        )
    task_cache = _build_task_cache(config)
    single_runs: list[RunRecord] = []
    transfer_runs: list[RunRecord] = []
    joint_runs: list[RunRecord] = []
    for seed in config.seeds:
        if 1 in config.experiments:
            single_runs.extend(_run_single_task_experiment(task_cache, seed, config, direct_judge))
        if 2 in config.experiments:
            transfer_runs.extend(_run_leave_one_out_experiment(task_cache, seed, config, direct_judge))
        if 3 in config.experiments:
            joint_runs.extend(_run_joint_experiment(task_cache, seed, config, direct_judge))

    experiments: dict[str, Any] = {}
    if single_runs:
        experiments["experiment_1_single_task"] = _summarize_runs(single_runs, config)
    if transfer_runs:
        experiments["experiment_2_leave_one_task_out"] = _summarize_runs(transfer_runs, config)
    if joint_runs:
        experiments["experiment_3_joint_training_stability"] = _summarize_runs(joint_runs, config)

    report = {
        "config": asdict(config),
        "task_labels": TASK_LABELS,
        "data_regime": "historical_offline_proxy" if config.data_dir else "synthetic_benchmark",
        "experimental_design": _experimental_design(),
        "split_policy": {
            "historical": "complete scientific regimes are held out for test; remaining records are disjointly assigned to train, feedback, and acceptance",
            "synthetic": "seeded record-level four-way split",
            "feedback_role": "trajectory failure diagnosis and threshold calibration",
            "acceptance_role": "RHI candidate H_i versus H_(i+1) model selection only",
            "test_role": "final reporting only; never used to propose or accept a harness",
            "transfer_role": "held-out target task contributes no records to zero-shot training, feedback, calibration, or acceptance",
        },
        "data_audit": _build_data_audit(task_cache, config),
        "limitations": [
            "Historical outcomes are benchmark-derived offline action-worthiness proxies, not expert annotations or online MatBot trajectory outcomes.",
            "The deterministic proposer is a reproducible trajectory-conditioned RHI implementation, not evidence that an LLM proposer improves the harness.",
            "llm_direct_judge is an optional one-shot baseline: it sees only the visible action context and does not train, mutate a harness, or receive trajectory feedback.",
            "Scientific utilities are reported per task and as macro summaries; pooled utility is not interpreted as a common physical unit.",
            "Source train, feedback, and acceptance records are disjoint but can share source regimes; final test regimes remain group-disjoint.",
        ],
        "runs": {
            "single_task": [run.to_dict() for run in single_runs],
            "leave_one_out": [run.to_dict() for run in transfer_runs],
            "joint": [run.to_dict() for run in joint_runs],
        },
        "experiments": experiments,
        "summary": {
            "n_single_runs": len(single_runs),
            "n_transfer_runs": len(transfer_runs),
            "n_joint_runs": len(joint_runs),
        },
        "baselines": {
            "llm_direct_judge": {
                "enabled": direct_judge is not None,
                "model": config.direct_judge_model if direct_judge is None else getattr(direct_judge, "model", "injected_scorer"),
                "calibration": "validation/feedback split only",
                "test_access": "visible_context, candidate_action, and pre-execution evidence only",
            }
        },
    }
    return report


def save_experiment_suite(
    config: ExperimentSuiteConfig,
    json_path: str | Path,
    markdown_path: str | Path | None = None,
    *,
    direct_judge: Any | None = None,
) -> dict[str, Any]:
    report = run_experiment_suite(config, direct_judge=direct_judge)
    write_json(report, json_path)
    if markdown_path is not None:
        Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
        Path(markdown_path).write_text(render_experiment_suite_markdown(report), encoding="utf-8")
    return report


def render_experiment_suite_markdown(report: dict[str, Any]) -> str:
    config = report["config"]
    lines = [
        "# RHI-MatSci Experiment Suite",
        "",
        "## Protocol",
        f"- Data regime: `{report['data_regime']}`.",
        f"- Tasks: {', '.join(config['tasks'])}.",
        f"- Seeds: {', '.join(str(seed) for seed in config['seeds'])}.",
        f"- Target risk: {config['alpha']:.2f}; minimum validation coverage: {config['min_coverage']:.2f}; fixed scientific budget: {config['budget_fraction']:.2f}.",
        "- Primary score is threshold-independent and combines AURC, calibration, and oracle-normalized fixed-budget discovery efficiency; threshold risk and coverage are reported together.",
        "- `llm_direct_judge` is included only when configured; it is a one-shot LLM judge with no training, recursive mutation, or trajectory feedback, and its threshold is calibrated on validation/feedback records.",
        "",
        "## What Each Experiment Tests",
    ]
    for key in ("experiment_1", "experiment_2", "experiment_3"):
        design = report["experimental_design"][key]
        lines.extend(
            [
                f"### {design['title']}",
                f"- **Claim tested**: {design['claim']} ",
                f"- **Reviewer challenge**: {design['reviewer_challenge']} ",
                f"- **Design response**: {design['response']} ",
            ]
        )
    lines.extend(
        [
            "## Leakage and Split Audit",
            "- Benchmark-derived post-outcome signals are excluded from model features and retained only as provenance metadata.",
            "- Record IDs are disjoint across train/feedback/acceptance/test; historical test regimes are group-disjoint from all source partitions.",
            "- Experiment 2 calibrates only on source tasks. `target_supervised_reference` is explicitly non-zero-shot and is reported only as a target-data reference, not as a strict oracle upper bound.",
            "",
        ]
    )
    for key, title in [
        ("experiment_1_single_task", "Experiment 1: Single-Task Action Worthiness"),
        ("experiment_2_leave_one_task_out", "Experiment 2: Leave-One-Task-Out Transfer"),
        ("experiment_3_joint_training_stability", "Experiment 3: Joint Training and Stability"),
    ]:
        if key not in report["experiments"]:
            continue
        lines.extend([f"## {title}"])
        lines.extend(_render_experiment_section(report["experiments"][key]))
    lines.extend(["## Limitations"] + [f"- {item}" for item in report["limitations"]])
    return "\n".join(lines) + "\n"


def _render_experiment_section(summary: dict[str, Any]) -> list[str]:
    lines = [
        f"- Runs: {summary['n_runs']}; seeds: {summary['seed_count']}; tasks: {summary['task_count']}.",
        "",
        "| Method | Primary score ↓ | AURC ↓ | Risk @ 10% ↓ | Budget hit rate ↑ | Oracle-normalized utility ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, aggregate in sorted(summary["by_method"].items()):
        lines.append(
            f"| `{method}` | {_mean_std(aggregate.get('score'))} | {_mean_std(aggregate.get('aurc'))} | "
            f"{_mean_std(aggregate.get('risk_at_0.10'))} | {_mean_std(aggregate.get('hit_rate'))} | "
            f"{_mean_std(aggregate.get('utility_efficiency'))} |"
        )
    lines.extend(
        [
            "",
            "Threshold-selected operating point (secondary; interpret risk with coverage):",
            "",
            "| Method | Selective risk ↓ | Coverage ↑ | ECE ↓ | Brier ↓ |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for method, aggregate in sorted(summary["by_method"].items()):
        lines.append(
            f"| `{method}` | {_mean_std(aggregate.get('selective_risk'))} | {_mean_std(aggregate.get('coverage'))} | "
            f"{_mean_std(aggregate.get('ece'))} | {_mean_std(aggregate.get('brier'))} |"
        )
    if summary.get("paired_comparisons"):
        lines.extend(["", "Paired comparisons against RHI (negative score difference favors RHI):"])
        for baseline, comparison in sorted(summary["paired_comparisons"].items()):
            interval = comparison["score_difference_95ci"]
            lines.append(
                f"- `{baseline}`: Δscore={comparison['score_difference_mean']:.4f} "
                f"[{interval[0]:.4f}, {interval[1]:.4f}], RHI wins={comparison['win_rate']:.3f}, "
                f"exact sign p={comparison['sign_test_p']:.4f}, n={comparison['n']}."
            )
    lines.append("")
    for task, task_summary in sorted(summary["by_task"].items()):
        lines.append(f"### {TASK_LABELS.get(task, task)}")
        for method, aggregate in sorted(task_summary["by_method"].items()):
            discovery_key = {
                "preferential_bo": "pairwise_latent_regret",
                "discover_unique": "unique_material_recall",
                "extreme_properties": "extreme_hit_recall",
            }.get(task, "positive_recall")
            lines.append(
                f"- `{method}`: score={_mean_std(aggregate.get('score'))}, AURC={_mean_std(aggregate.get('aurc'))}, "
                f"risk@10%={_mean_std(aggregate.get('risk_at_0.10'))}, hit={_mean_std(aggregate.get('hit_rate'))}, "
                f"{discovery_key}={_mean_std(aggregate.get(discovery_key))}."
            )
    return lines


def _validate_config(config: ExperimentSuiteConfig) -> None:
    if not config.tasks or any(task not in TASKS for task in config.tasks):
        raise ValueError(f"tasks must be a non-empty subset of {TASKS}")
    if not config.seeds:
        raise ValueError("seeds cannot be empty")
    if not config.experiments or any(experiment not in {1, 2, 3} for experiment in config.experiments):
        raise ValueError("experiments must contain one or more of 1, 2, and 3")
    if 2 in config.experiments and len(config.tasks) < 2:
        raise ValueError("leave-one-task-out requires at least two tasks")
    if not 0.0 < config.min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    if not 0.0 < config.budget_fraction <= 1.0:
        raise ValueError("budget_fraction must be in (0, 1]")


def _experimental_design() -> dict[str, dict[str, str]]:
    return {
        "experiment_1": {
            "title": "Experiment 1 — task-specific validity",
            "claim": "recursive harness revisions improve ranking and fixed-budget scientific action selection within each task family",
            "reviewer_challenge": "a low selective risk may be produced by abstaining, a random split may leak regimes, or gains may come from a learned classifier rather than recursion",
            "response": "use regime-held-out test groups, independent feedback and acceptance sets, strong non-RHI learned baselines, fixed-coverage risk, AURC, fixed-budget utility, and five paired seeds",
        },
        "experiment_2": {
            "title": "Experiment 2 — cross-task transfer",
            "claim": "an action-worthiness contract learned on two task families transfers zero-shot to a third",
            "reviewer_challenge": "target calibration leakage or task-size imbalance can manufacture transfer, and failure may simply reflect an impossible target",
            "response": "exclude all target records from source training/calibration/acceptance, balance source tasks in loss and calibration, evaluate on held-out target regimes, and report a target-supervised upper bound",
        },
        "experiment_3": {
            "title": "Experiment 3 — joint robustness",
            "claim": "one jointly trained harness remains stable across seeds and does not sacrifice a smaller task to optimize the largest dataset",
            "reviewer_challenge": "pooled metrics are dominated by the 10,987-record unique-material task",
            "response": "use inverse-frequency task weighting, macro acceptance, apply exactly the same selected joint gate to every task slice, and report per-task and macro-over-task results",
        },
    }


def _build_task_cache(config: ExperimentSuiteConfig) -> dict[str, dict[int, dict[str, list[ActionRecord]]]]:
    cache: dict[str, dict[int, dict[str, list[ActionRecord]]]] = {}
    for task in config.tasks:
        cache[task] = {}
        for seed in config.seeds:
            records = (
                load_historical_task_records(config.data_dir, task)
                if config.data_dir
                else make_records(task, n=config.n_per_task, seed=seed)
            )
            splitter = grouped_four_way_split if config.data_dir else random_four_way_split
            splits = splitter(
                records,
                seed=seed,
                train_fraction=config.train_fraction,
                feedback_fraction=config.feedback_fraction,
                acceptance_fraction=config.acceptance_fraction,
            )
            splits["all"] = records
            cache[task][seed] = splits
    return cache


def _run_single_task_experiment(
    task_cache: dict[str, dict[int, dict[str, list[ActionRecord]]]],
    seed: int,
    config: ExperimentSuiteConfig,
    direct_judge: Any | None = None,
) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for task in config.tasks:
        splits = task_cache[task][seed]
        split_sizes = _split_sizes(splits)
        rhi_report = train_rhi_from_splits(
            splits["train"],
            splits["feedback"],
            splits["test"],
            acceptance_records=splits["acceptance"],
            iterations=config.rhi_iterations,
            seed=seed,
            alpha=config.alpha,
            min_coverage=config.min_coverage,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            l2=config.l2,
            budget_fraction=config.budget_fraction,
            epsilon=config.rhi_epsilon,
        )
        runs.append(_run("experiment_1_single_task", "single_task", seed, task, "rhi", split_sizes, _compact_rhi_report(rhi_report)))
        runs.append(
            _run(
                "experiment_1_single_task",
                "single_task",
                seed,
                task,
                "non_rhi_seed",
                split_sizes,
                _train_feature_gate(splits["train"], splits["feedback"], splits["test"], list(RHI_SEED_FEATURES), config),
            )
        )
        runs.append(
            _run(
                "experiment_1_single_task",
                "single_task",
                seed,
                task,
                "static_full",
                split_sizes,
                _train_feature_gate(splits["train"], splits["feedback"], splits["test"], _full_features(splits["train"]), config),
            )
        )
        runs.extend(
            _weak_baseline_runs(
                "experiment_1_single_task",
                "single_task",
                seed,
                task,
                split_sizes,
                splits["feedback"],
                splits["test"],
                config,
                direct_judge=direct_judge,
            )
        )
    return runs


def _run_leave_one_out_experiment(
    task_cache: dict[str, dict[int, dict[str, list[ActionRecord]]]],
    seed: int,
    config: ExperimentSuiteConfig,
    direct_judge: Any | None = None,
) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for holdout_task in config.tasks:
        source_tasks = [task for task in config.tasks if task != holdout_task]
        source_train = _concat([task_cache[task][seed]["train"] for task in source_tasks])
        source_feedback = _concat([task_cache[task][seed]["feedback"] for task in source_tasks])
        source_acceptance = _concat([task_cache[task][seed]["acceptance"] for task in source_tasks])
        target_splits = task_cache[holdout_task][seed]
        target_test = target_splits["test"]
        split_sizes = {
            "source_train": len(source_train),
            "source_feedback": len(source_feedback),
            "source_acceptance": len(source_acceptance),
            "target_test": len(target_test),
            "target_all": len(target_splits["all"]),
        }
        rhi_report = train_rhi_from_splits(
            source_train,
            source_feedback,
            target_test,
            acceptance_records=source_acceptance,
            iterations=config.rhi_iterations,
            seed=seed,
            alpha=config.alpha,
            min_coverage=config.min_coverage,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            l2=config.l2,
            budget_fraction=config.budget_fraction,
            epsilon=config.rhi_epsilon,
            balance_benchmarks=True,
            macro_acceptance=True,
        )
        runs.append(
            _run(
                "experiment_2_leave_one_task_out",
                "zero_shot_leave_one_task_out",
                seed,
                holdout_task,
                "rhi",
                split_sizes,
                _compact_rhi_report(rhi_report),
                source_tasks,
                holdout_task,
            )
        )
        for method, features in (
            ("non_rhi_seed", list(RHI_SEED_FEATURES)),
            ("static_full", _full_features(source_train)),
        ):
            runs.append(
                _run(
                    "experiment_2_leave_one_task_out",
                    "zero_shot_leave_one_task_out",
                    seed,
                    holdout_task,
                    method,
                    split_sizes,
                    _train_feature_gate(source_train, source_feedback, target_test, features, config, balance_benchmarks=True),
                    source_tasks,
                    holdout_task,
                )
            )
        runs.extend(
            _weak_baseline_runs(
                "experiment_2_leave_one_task_out",
                "zero_shot_leave_one_task_out",
                seed,
                holdout_task,
                split_sizes,
                source_feedback,
                target_test,
                config,
                direct_judge=direct_judge,
                source_tasks=source_tasks,
                holdout_task=holdout_task,
                balance_benchmarks=True,
            )
        )
        upper_bound = _train_feature_gate(
            target_splits["train"],
            target_splits["feedback"],
            target_test,
            _full_features(target_splits["train"]),
            config,
        )
        runs.append(
            _run(
                "experiment_2_leave_one_task_out",
                "target_supervised_reference",
                seed,
                holdout_task,
                TRANSFER_REFERENCE,
                split_sizes,
                upper_bound,
                source_tasks,
                holdout_task,
            )
        )
    return runs


def _run_joint_experiment(
    task_cache: dict[str, dict[int, dict[str, list[ActionRecord]]]],
    seed: int,
    config: ExperimentSuiteConfig,
    direct_judge: Any | None = None,
) -> list[RunRecord]:
    train_records = _concat([task_cache[task][seed]["train"] for task in config.tasks])
    feedback_records = _concat([task_cache[task][seed]["feedback"] for task in config.tasks])
    acceptance_records = _concat([task_cache[task][seed]["acceptance"] for task in config.tasks])
    tests = {task: task_cache[task][seed]["test"] for task in config.tasks}
    pooled_test = _concat(list(tests.values()))
    split_sizes = {
        "train": len(train_records),
        "feedback": len(feedback_records),
        "acceptance": len(acceptance_records),
        "test": len(pooled_test),
    }
    rhi_report = train_rhi_from_splits(
        train_records,
        feedback_records,
        pooled_test,
        acceptance_records=acceptance_records,
        iterations=config.rhi_iterations,
        seed=seed,
        alpha=config.alpha,
        min_coverage=config.min_coverage,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        l2=config.l2,
        budget_fraction=config.budget_fraction,
        epsilon=config.rhi_epsilon,
        balance_benchmarks=True,
        macro_acceptance=True,
    )
    initial_gate = TrainedGate.from_json(rhi_report["initial_gate"])
    final_gate = TrainedGate.from_json(rhi_report["final_gate"])
    seed_gate = _fit_gate(train_records, feedback_records, list(RHI_SEED_FEATURES), config, balance_benchmarks=True)
    full_gate = _fit_gate(train_records, feedback_records, _full_features(train_records), config, balance_benchmarks=True)
    weak_reports = {
        "verbal_confidence": evaluate_probability_signal(
            feedback_records,
            pooled_test,
            verbal_confidence_scores,
            alpha=config.alpha,
            min_coverage=config.min_coverage,
            budget_fraction=config.budget_fraction,
            balance_benchmarks=True,
        ),
        "evidence_heuristic": evaluate_probability_signal(
            feedback_records,
            pooled_test,
            evidence_heuristic_scores,
            alpha=config.alpha,
            min_coverage=config.min_coverage,
            budget_fraction=config.budget_fraction,
            balance_benchmarks=True,
        ),
    }
    if direct_judge is not None:
        weak_reports["llm_direct_judge"] = evaluate_probability_signal(
            feedback_records,
            pooled_test,
            _direct_judge_score_fn(direct_judge),
            alpha=config.alpha,
            min_coverage=config.min_coverage,
            budget_fraction=config.budget_fraction,
            balance_benchmarks=True,
        )
    runs: list[RunRecord] = []
    for task, test_records in tests.items():
        task_split = {**split_sizes, "task_test": len(test_records)}
        task_rhi = _compact_rhi_slice_report(rhi_report, initial_gate, final_gate, test_records, config.budget_fraction)
        runs.append(_run("experiment_3_joint_training_stability", "joint_task_slice", seed, task, "rhi", task_split, task_rhi))
        runs.append(
            _run(
                "experiment_3_joint_training_stability",
                "joint_task_slice",
                seed,
                task,
                "non_rhi_seed",
                task_split,
                _compact_gate_from_gate(seed_gate, feedback_records, test_records, config.budget_fraction),
            )
        )
        runs.append(
            _run(
                "experiment_3_joint_training_stability",
                "joint_task_slice",
                seed,
                task,
                "static_full",
                task_split,
                _compact_gate_from_gate(full_gate, feedback_records, test_records, config.budget_fraction),
            )
        )
        for method, score_fn in (
            ("verbal_confidence", verbal_confidence_scores),
            ("evidence_heuristic", evidence_heuristic_scores),
        ):
            threshold = float(weak_reports[method]["threshold"])
            evaluation = _evaluate_scores(test_records, score_fn(test_records), threshold, config.budget_fraction)
            compact = {
                "threshold": threshold,
                "calibration": weak_reports[method]["calibration"],
                "validation": _compact_eval(weak_reports[method]["validation"]),
                "test": _compact_eval(evaluation),
                "test_score": _evaluation_score(evaluation),
            }
            runs.append(_run("experiment_3_joint_training_stability", "joint_task_slice", seed, task, method, task_split, compact))
        if direct_judge is not None:
            method = "llm_direct_judge"
            threshold = float(weak_reports[method]["threshold"])
            score_fn = _direct_judge_score_fn(direct_judge)
            evaluation = _evaluate_scores(test_records, score_fn(test_records), threshold, config.budget_fraction)
            compact = {
                "threshold": threshold,
                "calibration": weak_reports[method]["calibration"],
                "validation": _compact_eval(weak_reports[method]["validation"]),
                "test": _compact_eval(evaluation),
                "test_score": _evaluation_score(evaluation),
            }
            runs.append(_run("experiment_3_joint_training_stability", "joint_task_slice", seed, task, method, task_split, compact))
    return runs


def _weak_baseline_runs(
    experiment: str,
    setting: str,
    seed: int,
    task: str,
    split_sizes: dict[str, int],
    validation_records: list[ActionRecord],
    test_records: list[ActionRecord],
    config: ExperimentSuiteConfig,
    *,
    source_tasks: list[str] | None = None,
    holdout_task: str = "",
    balance_benchmarks: bool = False,
    direct_judge: Any | None = None,
) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for method, score_fn in (
        ("verbal_confidence", verbal_confidence_scores),
        ("evidence_heuristic", evidence_heuristic_scores),
    ):
        report = evaluate_probability_signal(
            validation_records,
            test_records,
            score_fn,
            alpha=config.alpha,
            min_coverage=config.min_coverage,
            budget_fraction=config.budget_fraction,
            balance_benchmarks=balance_benchmarks,
        )
        runs.append(
            _run(
                experiment,
                setting,
                seed,
                task,
                method,
                split_sizes,
                _compact_probability_report(report),
                source_tasks or [],
                holdout_task,
            )
        )
    if direct_judge is not None:
        method = "llm_direct_judge"
        report = evaluate_probability_signal(
            validation_records,
            test_records,
            _direct_judge_score_fn(direct_judge),
            alpha=config.alpha,
            min_coverage=config.min_coverage,
            budget_fraction=config.budget_fraction,
            balance_benchmarks=balance_benchmarks,
        )
        runs.append(
            _run(
                experiment,
                setting,
                seed,
                task,
                method,
                split_sizes,
                _compact_probability_report(report),
                source_tasks or [],
                holdout_task,
            )
        )
    return runs


def _direct_judge_score_fn(direct_judge: Any) -> Callable[[list[ActionRecord]], list[float]]:
    def score(records: list[ActionRecord]) -> list[float]:
        if hasattr(direct_judge, "score_records"):
            values = direct_judge.score_records(records)
        elif callable(direct_judge):
            values = direct_judge(records)
        else:
            raise TypeError("direct_judge must expose score_records(records) or be callable")
        if isinstance(values, dict):
            return [float(values[record.record_id]) for record in records]
        values = list(values)
        if len(values) != len(records):
            raise ValueError("direct judge returned a score count different from records")
        return [float(value) for value in values]

    return score


def _fit_gate(
    train_records: list[ActionRecord],
    validation_records: list[ActionRecord],
    feature_names: list[str],
    config: ExperimentSuiteConfig,
    *,
    balance_benchmarks: bool = False,
) -> TrainedGate:
    return train_gate_with_features(
        train_records,
        validation_records,
        feature_names=feature_names,
        alpha=config.alpha,
        min_coverage=config.min_coverage,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        l2=config.l2,
        balance_benchmarks=balance_benchmarks,
    )


def _train_feature_gate(
    train_records: list[ActionRecord],
    validation_records: list[ActionRecord],
    test_records: list[ActionRecord],
    feature_names: list[str],
    config: ExperimentSuiteConfig,
    *,
    balance_benchmarks: bool = False,
) -> dict[str, Any]:
    gate = _fit_gate(train_records, validation_records, feature_names, config, balance_benchmarks=balance_benchmarks)
    return _compact_gate_from_gate(gate, validation_records, test_records, config.budget_fraction)


def _compact_gate_from_gate(
    gate: TrainedGate,
    validation_records: list[ActionRecord],
    test_records: list[ActionRecord],
    budget_fraction: float,
) -> dict[str, Any]:
    validation = evaluate_gate(validation_records, gate, budget_fraction=budget_fraction)
    test = evaluate_gate(test_records, gate, budget_fraction=budget_fraction)
    return {
        "threshold": gate.threshold,
        "calibration": gate.calibration,
        "feature_names": list(gate.model.feature_names),
        "validation": _compact_eval(validation),
        "test": _compact_eval(test),
        "test_score": _evaluation_score(test),
    }


def _compact_probability_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold": report["threshold"],
        "calibration": report["calibration"],
        "validation": _compact_eval(report["validation"]),
        "test": _compact_eval(report["test"]),
        "test_score": _evaluation_score(report["test"]),
    }


def _compact_rhi_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": report["method"]["name"],
        "proposer": report["method"]["proposer"],
        "initial_harness": report["initial_harness"]["name"],
        "final_harness": report["final_harness"]["name"],
        "accepted_versions": report["accepted_versions"],
        "n_comparisons": len(report["comparisons"]),
        "n_accepted": max(0, len(report["accepted_versions"]) - 1),
        "comparisons": report["comparisons"],
        "feature_names": list(report["final_gate"]["model"]["feature_names"]),
        "calibration": report["final_gate"]["calibration"],
        "validation": _compact_eval(report["validation"]),
        "initial_test": _compact_eval(report["initial_test"]),
        "final_test": _compact_eval(report["test"]),
        "initial_score": _evaluation_score(report["initial_test"]),
        "final_score": _evaluation_score(report["test"]),
    }


def _compact_rhi_slice_report(
    report: dict[str, Any],
    initial_gate: TrainedGate,
    final_gate: TrainedGate,
    test_records: list[ActionRecord],
    budget_fraction: float,
) -> dict[str, Any]:
    initial_test = evaluate_gate(test_records, initial_gate, budget_fraction=budget_fraction)
    final_test = evaluate_gate(test_records, final_gate, budget_fraction=budget_fraction)
    compact = _compact_rhi_report(report)
    compact["initial_test"] = _compact_eval(initial_test)
    compact["final_test"] = _compact_eval(final_test)
    compact["initial_score"] = _evaluation_score(initial_test)
    compact["final_score"] = _evaluation_score(final_test)
    return compact


def _compact_eval(report: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "metrics": report["metrics"],
        "discovery_gain": report["discovery_gain"],
        "scientific_discovery": report.get("scientific_discovery", {}),
        "n_records": report.get("n_records", int(report["metrics"].get("n", 0))),
    }
    if "risk_coverage" in report:
        compact["risk_coverage"] = report["risk_coverage"]
    if "macro_metrics" in report:
        compact["macro_metrics"] = report["macro_metrics"]
        compact["macro_discovery_gain"] = report["macro_discovery_gain"]
    return compact


def _evaluate_scores(
    records: list[ActionRecord],
    scores: list[float],
    threshold: float,
    budget_fraction: float,
) -> dict[str, Any]:
    labels = [record.label for record in records]
    budget = max(1, int(len(records) * budget_fraction)) if records else 0
    return {
        "metrics": binary_metrics(labels, scores, threshold),
        "risk_coverage": fixed_coverage_metrics(labels, scores),
        "discovery_gain": discovery_gain(labels, [record.utility for record in records], scores, budget),
        "scientific_discovery": scientific_discovery_metrics(records, scores, budget),
        "threshold": threshold,
        "n_records": len(records),
    }


def _evaluation_score(evaluation: dict[str, Any]) -> float:
    metrics = evaluation.get("macro_metrics", evaluation["metrics"])
    gain = evaluation.get("macro_discovery_gain", evaluation["discovery_gain"])
    return action_worthiness_score(metrics, gain)


def _summarize_runs(runs: list[RunRecord], config: ExperimentSuiteConfig) -> dict[str, Any]:
    rows = [run.to_dict() for run in runs]
    by_method_rows = _group_rows(rows, "method")
    by_task_rows = _group_rows(rows, "task")
    summary = {
        "n_runs": len(rows),
        "seed_count": len({row["seed"] for row in rows}),
        "task_count": len({row["task"] for row in rows}),
        "by_method": {method: _aggregate_method(group) for method, group in by_method_rows.items()},
        "by_task": {
            task: {
                "n_runs": len(group),
                "by_method": {
                    method: _aggregate_method(method_rows)
                    for method, method_rows in _group_rows(group, "method").items()
                },
            }
            for task, group in by_task_rows.items()
        },
        "primary_metric": "threshold-independent action-worthiness score (lower is better)",
        "paired_comparisons": {},
        "seed_protocol": list(config.seeds),
    }
    rhi_rows = by_method_rows.get("rhi", [])
    for method, method_rows in by_method_rows.items():
        if method != "rhi":
            summary["paired_comparisons"][method] = _paired_comparison(rhi_rows, method_rows)
    return summary


def _aggregate_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    evaluations = [_test_evaluation(row) for row in rows]
    scores = [_row_score(row) for row in rows]
    aggregate: dict[str, Any] = {
        "n": len(rows),
        "score": _numeric_summary(scores),
        "seeds": sorted({row["seed"] for row in rows}),
    }
    metric_names = (
        "selective_risk",
        "coverage",
        "ece",
        "brier",
        "log_loss",
        "confidently_wrong_rate",
        "aurc",
        "coverage_at_risk_0.05",
        "coverage_at_risk_0.10",
    )
    for metric_name in metric_names:
        aggregate[metric_name] = _numeric_summary([float(item["metrics"][metric_name]) for item in evaluations])
    for risk_key in ("risk_at_0.10", "risk_at_0.25", "risk_at_0.50"):
        aggregate[risk_key] = _numeric_summary(
            [float(item["risk_coverage"][risk_key]["selective_risk"]) for item in evaluations]
        )
    gain_names = (
        "hit_rate",
        "mean_utility",
        "best_utility",
        "random_hit_rate",
        "random_mean_utility",
        "hit_rate_lift",
        "mean_utility_lift",
        "hit_efficiency",
        "utility_efficiency",
        "normalized_hit_lift",
        "normalized_utility_lift",
    )
    for gain_name in gain_names:
        aggregate[gain_name] = _numeric_summary([float(item["discovery_gain"][gain_name]) for item in evaluations])
    scientific_names = sorted(
        {
            name
            for item in evaluations
            for name, value in item.get("scientific_discovery", {}).items()
            if isinstance(value, (int, float)) and name not in {"selected_regimes"}
        }
    )
    for name in scientific_names:
        values = [
            float(item["scientific_discovery"][name])
            for item in evaluations
            if name in item.get("scientific_discovery", {})
        ]
        aggregate[name] = _numeric_summary(values)
    if rows[0]["method"] == "rhi":
        aggregate["n_accepted"] = _numeric_summary([float(row["report"]["n_accepted"]) for row in rows])
        aggregate["initial_score"] = _numeric_summary([float(row["report"]["initial_score"]) for row in rows])
    return aggregate


def _paired_comparison(rhi_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_index = {(row["seed"], row["task"]): row for row in baseline_rows}
    pairs = [(row, baseline_index[(row["seed"], row["task"])]) for row in rhi_rows if (row["seed"], row["task"]) in baseline_index]
    if not pairs:
        return {
            "n": 0,
            "score_difference_mean": 0.0,
            "score_difference_95ci": [0.0, 0.0],
            "win_rate": 0.0,
            "sign_test_p": 1.0,
        }
    differences = [_row_score(rhi) - _row_score(base) for rhi, base in pairs]
    wins = sum(difference < 0 for difference in differences)
    non_ties = [difference for difference in differences if difference != 0]
    return {
        "n": len(pairs),
        "score_difference_mean": mean(differences),
        "score_difference_95ci": _bootstrap_mean_ci(differences),
        "win_rate": wins / len(pairs),
        "sign_test_p": _exact_sign_test(sum(difference < 0 for difference in non_ties), len(non_ties)),
        "utility_difference_mean": mean(
            _test_evaluation(rhi)["discovery_gain"]["mean_utility"]
            - _test_evaluation(base)["discovery_gain"]["mean_utility"]
            for rhi, base in pairs
        ),
    }


def _bootstrap_mean_ci(values: list[float], *, draws: int = 4000) -> list[float]:
    if len(values) <= 1:
        value = values[0] if values else 0.0
        return [value, value]
    rng = random.Random(1729)
    estimates = sorted(mean(rng.choice(values) for _ in values) for _ in range(draws))
    return [estimates[int(0.025 * draws)], estimates[min(draws - 1, int(0.975 * draws))]]


def _exact_sign_test(wins: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, index) for index in range(0, min(wins, trials - wins) + 1)) / (2**trials)
    return min(1.0, 2.0 * tail)


def _test_evaluation(row: dict[str, Any]) -> dict[str, Any]:
    return row["report"]["final_test"] if row["method"] == "rhi" else row["report"]["test"]


def _row_score(row: dict[str, Any]) -> float:
    return float(row["report"]["final_score"] if row["method"] == "rhi" else row["report"]["test_score"])


def _run(
    experiment: str,
    setting: str,
    seed: int,
    task: str,
    method: str,
    split: dict[str, int],
    report: dict[str, Any],
    source_tasks: list[str] | None = None,
    holdout_task: str = "",
) -> RunRecord:
    return RunRecord(experiment, setting, seed, task, method, split, source_tasks or [], holdout_task, report)


def _build_data_audit(
    task_cache: dict[str, dict[int, dict[str, list[ActionRecord]]]],
    config: ExperimentSuiteConfig,
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    all_disjoint = True
    all_group_disjoint = True
    for task in config.tasks:
        seed_rows: dict[str, Any] = {}
        for seed in config.seeds:
            splits = task_cache[task][seed]
            partitions = {name: splits[name] for name in ("train", "feedback", "acceptance", "test")}
            ids = {name: {record.record_id for record in rows} for name, rows in partitions.items()}
            disjoint = all(
                ids[left].isdisjoint(ids[right])
                for left in ids
                for right in ids
                if left < right
            )
            source_groups = {
                str(record.metadata.get("group_id", record.benchmark))
                for name in ("train", "feedback", "acceptance")
                for record in partitions[name]
            }
            test_groups = {str(record.metadata.get("group_id", record.benchmark)) for record in partitions["test"]}
            group_disjoint = source_groups.isdisjoint(test_groups) if config.data_dir else True
            all_disjoint = all_disjoint and disjoint
            all_group_disjoint = all_group_disjoint and group_disjoint
            seed_rows[str(seed)] = {
                "sizes": {name: len(rows) for name, rows in partitions.items()},
                "positive_rates": {
                    name: sum(record.label for record in rows) / len(rows) for name, rows in partitions.items()
                },
                "record_partitions_disjoint": disjoint,
                "source_test_groups_disjoint": group_disjoint,
                "source_groups": sorted(source_groups),
                "test_groups": sorted(test_groups),
            }
        records = task_cache[task][config.seeds[0]]["all"]
        excluded = sorted({name for record in records for name in record.metadata.get("excluded_oracle_features", [])})
        tasks[task] = {
            "n_records": len(records),
            "n_positive": sum(record.label for record in records),
            "n_groups": len({str(record.metadata.get("group_id", record.benchmark)) for record in records}),
            "label_sources": sorted({str(record.metadata.get("label_source", "unknown")) for record in records}),
            "excluded_oracle_features": excluded,
            "visible_text_outcome_leakage_free": _visible_text_audit(records),
            "model_feature_union": sorted({feature for record in records for feature in record.features}),
            "seeds": seed_rows,
        }
    return {
        "all_record_partitions_disjoint": all_disjoint,
        "all_historical_source_test_groups_disjoint": all_group_disjoint,
        "tasks": tasks,
    }


def _visible_text_audit(records: list[ActionRecord]) -> dict[str, Any]:
    forbidden_patterns = (
        "log10(k_vrh)=",
        "hit_fraction=",
        "reward=",
        "all_hit=",
        "hidden objective value is",
        "objective value is",
    )
    offenders: list[str] = []
    for record in records:
        text = " ".join([record.visible_context, record.candidate_action, *record.evidence]).lower()
        if any(pattern in text for pattern in forbidden_patterns):
            offenders.append(record.record_id)
    return {"passed": not offenders, "n_offending_records": len(offenders), "offending_record_ids": offenders[:20]}


def _full_features(records: list[ActionRecord]) -> list[str]:
    return feature_names_from_records(records, preferred=list(DEFAULT_FEATURES))


def _split_sizes(splits: dict[str, list[ActionRecord]]) -> dict[str, int]:
    return {name: len(splits[name]) for name in ("train", "feedback", "acceptance", "test")}


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return groups


def _concat(parts: list[list[ActionRecord]]) -> list[ActionRecord]:
    return [record for part in parts for record in part]


def _numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _mean_std(summary: dict[str, float] | None) -> str:
    if not summary:
        return "n/a"
    return f"{summary['mean']:.3f} ± {summary['std']:.3f}"
