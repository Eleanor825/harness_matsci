from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .historical import HISTORICAL_TASK_FILES
from .voi_experiments import VoIExperimentConfig, run_voi_experiment_suite


DEFAULT_MECHANISM_METHODS = (
    "h0_reliability",
    "static_full_reliability",
    "static_utility_no_cost",
    "static_voi_no_uncertainty",
    "static_voi_no_routing",
    "static_voi_no_cost",
    "static_voi",
    "original_rhi",
    "scivoi_rhi",
)
DEFAULT_MECHANISM_POLICIES = ("mean_guarded", "always_accept", "never_accept")

MECHANISM_COMPARISONS = (
    ("static_utility_no_cost", "h0_reliability", "continuous utility head vs reliability-only H0"),
    ("static_voi_no_uncertainty", "static_utility_no_cost", "uncertainty removal vs utility-only head"),
    ("static_voi_no_routing", "static_voi_no_uncertainty", "routing removal vs uncertainty-only VoI"),
    ("static_voi", "static_voi_no_cost", "cost-aware VoI vs cost-blind VoI"),
    ("scivoi_policy_always_accept", "scivoi_policy_never_accept", "recursive updates vs frozen harness"),
    ("scivoi_rhi", "scivoi_policy_never_accept", "guarded recursive updates vs frozen harness"),
    ("scivoi_policy_always_accept", "original_rhi", "Sci-VoI RHI vs reliability-only RHI"),
    ("scivoi_rhi", "original_rhi", "guarded Sci-VoI RHI vs reliability-only RHI"),
    ("scivoi_policy_always_accept", "static_full_reliability", "full action-value policy vs static reliability"),
)


@dataclass(frozen=True)
class MechanismAblationConfig:
    data_dir: str
    tasks: tuple[str, ...] = tuple(HISTORICAL_TASK_FILES)
    seeds: tuple[int, ...] = (1, 7, 13, 21, 42)
    iterations: int = 3
    budget_fraction: float = 0.1
    alpha: float = 0.1
    epochs: int = 60
    learning_rate: float = 0.08
    l2: float = 0.01


def run_mechanism_ablation(config: MechanismAblationConfig) -> dict[str, Any]:
    voi_config = VoIExperimentConfig(
        data_dir=config.data_dir,
        tasks=config.tasks,
        methods=DEFAULT_MECHANISM_METHODS,
        components=(),
        acceptance_policies=DEFAULT_MECHANISM_POLICIES,
        iterations=config.iterations,
        budget_fraction=config.budget_fraction,
        alpha=config.alpha,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        l2=config.l2,
        seeds=config.seeds,
    )
    base_report = run_voi_experiment_suite(voi_config)
    compact_runs = [_strip_run(row) for row in base_report["runs"]]
    method_names = sorted({str(row["method"]) for row in compact_runs})
    comparisons = [
        _paired_comparison(compact_runs, variant, baseline, label)
        for variant, baseline, label in MECHANISM_COMPARISONS
        if variant in method_names and baseline in method_names
    ]
    return {
        "schema": "scivoi-mechanism-ablation-v1",
        "config": asdict(config),
        "protocol": base_report["protocol"],
        "data": base_report["data"],
        "summary": base_report["summary"],
        "mechanism_comparisons": comparisons,
        "runs": compact_runs,
        "limitations": [
            "This is an offline historical-proxy mechanism ablation, not an online MatBot deployment.",
            "Component-only RHI variants isolate which mutation family is sufficient; static variants provide leave-out interpretations for utility, cost, and verification behavior.",
            "Direct LLM-as-judge is intentionally absent because no API key is configured.",
        ],
    }


def save_mechanism_ablation(report: dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_target.write_text(render_mechanism_ablation_markdown(report), encoding="utf-8")


def render_mechanism_ablation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sci-VoI Mechanism Ablation",
        "",
        "> Primary metric: oracle-normalized net utility at a fixed 10% action budget; higher is better. Risk is selective risk among executed actions; lower is better.",
        "",
        f"- Data: `{report['data']['total_records']}` historical proxy records.",
        f"- Outer folds: `{report['protocol']['outer_folds']}` held-out scientific regimes × `{len(report['config']['seeds'])}` seeds.",
        "- Direct LLM judge is excluded because no API key is configured.",
        "",
        "## Aggregate Methods",
        "",
        "| Method | Net utility ↑ | Risk-adjusted ↑ | Risk ↓ | Hit rate ↑ | Folds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, values in sorted(report["summary"]["methods"].items()):
        lines.append(
            f"| `{method}` | {values['oracle_normalized_net_utility']['mean']:.4f} ± {values['oracle_normalized_net_utility']['std']:.4f} | "
            f"{values['risk_adjusted_utility']['mean']:.4f} | {values['execute_selective_risk']['mean']:.4f} | "
            f"{values['hit_rate']['mean']:.4f} | {int(values['n'])} |"
        )
    lines.extend([
        "",
        "## Mechanism Comparisons",
        "",
        "| Mechanism test | Variant | Baseline | Utility Δ ↑ | Risk Δ ↓ | Risk-adjusted Δ ↑ | Win rate | 95% CI |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for comparison in report["mechanism_comparisons"]:
        lines.append(
            f"| {comparison['label']} | `{comparison['variant']}` | `{comparison['baseline']}` | "
            f"{comparison['utility_delta_mean']:.4f} | {comparison['risk_delta_mean']:.4f} | "
            f"{comparison['risk_adjusted_delta_mean']:.4f} | {comparison['utility_win_rate']:.3f} | "
            f"[{comparison['utility_delta_ci95'][0]:.4f}, {comparison['utility_delta_ci95'][1]:.4f}] |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `static_utility` vs `h0_reliability` isolates whether modeling continuous scientific utility helps beyond reliability alone.",
        "- `static_voi` vs `static_utility` isolates epistemic uncertainty and verification routing in a non-recursive harness.",
        "- `static_voi` vs `static_voi_no_cost` tests whether using cost in the decision score matters when evaluation still charges action cost.",
        "- `scivoi_policy_always_accept` vs `static_voi` tests whether recursive harness mutation adds value beyond a fixed VoI contract.",
        "- `scivoi_policy_always_accept` vs `original_rhi` tests the main method against reliability-only RHI.",
        "",
        "## Claim Boundary",
        "",
        "These results support mechanism attribution on offline proxy tasks. They should be paired with the label/utility audit and, later, direct LLM judge and live MatBot trajectory baselines.",
    ])
    return "\n".join(lines) + "\n"


def _strip_run(row: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "task",
        "holdout_group",
        "seed",
        "method",
        "oracle_normalized_net_utility",
        "outcome_conditioned_oracle_normalized_net_utility",
        "selected_utility",
        "oracle_utility",
        "hit_rate",
        "coverage",
        "execute_selective_risk",
        "confidently_wrong_execute_rate",
        "verify_rate",
        "stop_rate",
        "regime_coverage",
        "utility_efficiency",
        "hit_efficiency",
        "threshold",
        "accepted_versions",
    }
    compact = {key: row[key] for key in keys if key in row}
    compact["risk_adjusted_utility"] = float(compact.get("oracle_normalized_net_utility", 0.0)) - 0.25 * float(compact.get("execute_selective_risk", 0.0))
    return compact


def _paired_comparison(rows: list[dict[str, Any]], variant: str, baseline: str, label: str) -> dict[str, Any]:
    variant_rows = {(row["task"], row["holdout_group"], row["seed"]): row for row in rows if row["method"] == variant}
    baseline_rows = {(row["task"], row["holdout_group"], row["seed"]): row for row in rows if row["method"] == baseline}
    keys = sorted(variant_rows.keys() & baseline_rows.keys())
    utility_deltas = [
        float(variant_rows[key]["oracle_normalized_net_utility"]) - float(baseline_rows[key]["oracle_normalized_net_utility"])
        for key in keys
    ]
    risk_deltas = [
        float(variant_rows[key]["execute_selective_risk"]) - float(baseline_rows[key]["execute_selective_risk"])
        for key in keys
    ]
    adjusted_deltas = [
        float(variant_rows[key]["risk_adjusted_utility"]) - float(baseline_rows[key]["risk_adjusted_utility"])
        for key in keys
    ]
    utility_wins = sum(value > 0.0 for value in utility_deltas)
    adjusted_wins = sum(value > 0.0 for value in adjusted_deltas)
    return {
        "label": label,
        "variant": variant,
        "baseline": baseline,
        "n": len(keys),
        "utility_delta_mean": mean(utility_deltas) if utility_deltas else 0.0,
        "utility_delta_std": pstdev(utility_deltas) if len(utility_deltas) > 1 else 0.0,
        "utility_delta_ci95": _bootstrap_ci(utility_deltas),
        "risk_delta_mean": mean(risk_deltas) if risk_deltas else 0.0,
        "risk_adjusted_delta_mean": mean(adjusted_deltas) if adjusted_deltas else 0.0,
        "utility_win_rate": utility_wins / len(keys) if keys else 0.0,
        "risk_adjusted_win_rate": adjusted_wins / len(keys) if keys else 0.0,
        "utility_sign_test_p": _sign_test(utility_wins, len([value for value in utility_deltas if value != 0.0])),
    }


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
