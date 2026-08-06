from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Any

from .benchmarks import BENCHMARK_BUILDERS, make_records
from .io import write_json
from .training import evidence_heuristic_baseline, evaluate_gate, split_records, train_gate, verbal_confidence_baseline


@dataclass(frozen=True)
class CampaignConfig:
    benchmarks: list[str]
    seeds: list[int]
    n: int = 300
    alpha: float = 0.1
    budget_fraction: float = 0.1
    lineage_weight: float = 0.0


def _numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "min": values[0], "max": values[0]}
    return {"mean": mean(values), "std": pstdev(values), "min": min(values), "max": max(values)}


def run_campaign(config: CampaignConfig) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for benchmark in config.benchmarks:
        if benchmark not in BENCHMARK_BUILDERS:
            choices = ", ".join(sorted(BENCHMARK_BUILDERS))
            raise ValueError(f"unknown benchmark {benchmark!r}; choose one of: {choices}")
        for seed in config.seeds:
            records = make_records(benchmark, n=config.n, seed=seed)
            train_records, val_records, test_records = split_records(records, seed=seed)
            gate = train_gate(train_records, val_records, alpha=config.alpha, lineage_weight=config.lineage_weight)
            learned = evaluate_gate(test_records, gate, budget_fraction=config.budget_fraction)
            verbal = verbal_confidence_baseline(test_records, alpha=config.alpha, budget_fraction=config.budget_fraction)
            heuristic = evidence_heuristic_baseline(test_records, alpha=config.alpha, budget_fraction=config.budget_fraction)
            run = {
                "benchmark": benchmark,
                "seed": seed,
                "sizes": {
                    "train": len(train_records),
                    "val": len(val_records),
                    "test": len(test_records),
                },
                "learned": learned,
                "verbal_baseline": verbal,
                "heuristic_baseline": heuristic,
            }
            runs.append(run)

    aggregate = _aggregate_runs(runs, config)
    by_benchmark = {
        benchmark: _aggregate_runs([run for run in runs if run["benchmark"] == benchmark], config)
        for benchmark in config.benchmarks
    }

    return {"config": asdict(config), "runs": runs, "aggregate": aggregate, "by_benchmark": by_benchmark}


def _aggregate_runs(runs: list[dict[str, Any]], config: CampaignConfig) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "n_runs": len(runs),
        "benchmarks": sorted({run["benchmark"] for run in runs}) if runs else list(config.benchmarks),
        "seeds": sorted({run["seed"] for run in runs}) if runs else list(config.seeds),
        "alpha": config.alpha,
        "budget_fraction": config.budget_fraction,
    }
    for section in ["learned", "verbal_baseline", "heuristic_baseline"]:
        for metric_key in ["threshold"]:
            values = [float(run[section][metric_key]) for run in runs if metric_key in run[section]]
            aggregate[f"{section}.{metric_key}"] = _numeric_summary(values)
        metric_paths = [
            ("metrics.coverage", ["metrics", "coverage"]),
            ("metrics.selective_accuracy", ["metrics", "selective_accuracy"]),
            ("metrics.selective_risk", ["metrics", "selective_risk"]),
            ("metrics.ece", ["metrics", "ece"]),
            ("metrics.brier", ["metrics", "brier"]),
            ("metrics.log_loss", ["metrics", "log_loss"]),
            ("discovery_gain.hit_rate", ["discovery_gain", "hit_rate"]),
            ("discovery_gain.mean_utility", ["discovery_gain", "mean_utility"]),
            ("discovery_gain.best_utility", ["discovery_gain", "best_utility"]),
        ]
        for summary_key, path in metric_paths:
            values = [float(_nested_get(run[section], path)) for run in runs]
            aggregate[f"{section}.{summary_key}"] = _numeric_summary(values)

    return aggregate

def _nested_get(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for key in path:
        current = current[key]
    return current


def save_campaign_report(config: CampaignConfig, path: str) -> dict[str, Any]:
    report = run_campaign(config)
    write_json(report, path)
    return report
