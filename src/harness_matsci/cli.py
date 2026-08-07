from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .benchmarks import BENCHMARK_BUILDERS, make_records
from .campaign import CampaignConfig, save_campaign_report
from .experiments import ExperimentSuiteConfig, save_experiment_suite
from .io import read_json, read_jsonl, write_json, write_jsonl
from .paper_bootstrap import DEFAULT_PAPER_ACTIONS_PATH, run_paper_bootstrap_experiment
from .rhi import train_rhi
from .training import TrainedGate, evaluate_gate, split_records, train_gate


def _comma_list(value: str, cast=str) -> list[Any]:
    if not value:
        return []
    return [cast(item) for item in value.split(",") if item]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness-matsci")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser("make-benchmark", help="Generate a benchmark JSONL dataset")
    benchmark_parser.add_argument("--benchmark", choices=sorted(BENCHMARK_BUILDERS), required=True)
    benchmark_parser.add_argument("--out", required=True)
    benchmark_parser.add_argument("--n", type=int, default=300)
    benchmark_parser.add_argument("--seed", type=int, default=0)
    benchmark_parser.set_defaults(func=_cmd_make_benchmark)

    train_parser = subparsers.add_parser("train", help="Train a gate on a JSONL dataset")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--out", required=True)
    train_parser.add_argument("--alpha", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--train-fraction", type=float, default=0.6)
    train_parser.add_argument("--val-fraction", type=float, default=0.2)
    train_parser.add_argument("--epochs", type=int, default=700)
    train_parser.add_argument("--learning-rate", type=float, default=0.08)
    train_parser.add_argument("--l2", type=float, default=0.001)
    train_parser.add_argument("--lineage-weight", type=float, default=0.0)
    train_parser.add_argument("--report", help="Optional training report JSON path")
    train_parser.set_defaults(func=_cmd_train)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a saved gate on a JSONL dataset")
    evaluate_parser.add_argument("--data", required=True)
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--out", required=True)
    evaluate_parser.add_argument("--budget-fraction", type=float, default=0.1)
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    campaign_parser = subparsers.add_parser("campaign", help="Run benchmark/baseline comparisons")
    campaign_parser.add_argument("--benchmark", action="append", choices=sorted(BENCHMARK_BUILDERS))
    campaign_parser.add_argument("--benchmarks", help="Comma-separated benchmark list")
    campaign_parser.add_argument("--seeds", default="0,1,2")
    campaign_parser.add_argument("--n", type=int, default=300)
    campaign_parser.add_argument("--alpha", type=float, default=0.1)
    campaign_parser.add_argument("--budget-fraction", type=float, default=0.1)
    campaign_parser.add_argument("--lineage-weight", type=float, default=0.0)
    campaign_parser.add_argument("--out", required=True)
    campaign_parser.set_defaults(func=_cmd_campaign)

    demo_parser = subparsers.add_parser("run-demo", help="Generate data, train, and evaluate in one shot")
    demo_parser.add_argument("--benchmark", choices=sorted(BENCHMARK_BUILDERS), default="preferential_bo")
    demo_parser.add_argument("--n", type=int, default=300)
    demo_parser.add_argument("--seed", type=int, default=0)
    demo_parser.add_argument("--alpha", type=float, default=0.1)
    demo_parser.add_argument("--budget-fraction", type=float, default=0.1)
    demo_parser.set_defaults(func=_cmd_demo)

    paper_parser = subparsers.add_parser("paper-bootstrap", help="Run the first historical-paper bootstrap experiment")
    paper_parser.add_argument("--data", default=str(DEFAULT_PAPER_ACTIONS_PATH))
    paper_parser.add_argument("--workdir", default="runs/paper_bootstrap_v1")
    paper_parser.add_argument("--seed", type=int, default=7)
    paper_parser.add_argument("--alpha", type=float, default=0.1)
    paper_parser.add_argument("--train-fraction", type=float, default=0.6)
    paper_parser.add_argument("--val-fraction", type=float, default=0.2)
    paper_parser.add_argument("--epochs", type=int, default=700)
    paper_parser.add_argument("--learning-rate", type=float, default=0.08)
    paper_parser.add_argument("--l2", type=float, default=0.001)
    paper_parser.add_argument("--budget-fraction", type=float, default=0.1)
    paper_parser.set_defaults(func=_cmd_paper_bootstrap)

    rhi_parser = subparsers.add_parser("rhi", help="Run trajectory-feedback Recursive Harness Self-Improvement")
    rhi_parser.add_argument("--data", required=True)
    rhi_parser.add_argument("--out", required=True)
    rhi_parser.add_argument("--iterations", type=int, default=3)
    rhi_parser.add_argument("--seed", type=int, default=7)
    rhi_parser.add_argument("--alpha", type=float, default=0.1)
    rhi_parser.add_argument("--train-fraction", type=float, default=0.6)
    rhi_parser.add_argument("--val-fraction", type=float, default=0.2)
    rhi_parser.add_argument("--epochs", type=int, default=700)
    rhi_parser.add_argument("--learning-rate", type=float, default=0.08)
    rhi_parser.add_argument("--l2", type=float, default=0.001)
    rhi_parser.add_argument("--budget-fraction", type=float, default=0.1)
    rhi_parser.add_argument("--min-coverage", type=float, default=0.0)
    rhi_parser.add_argument("--epsilon", type=float, default=0.01)
    rhi_parser.set_defaults(func=_cmd_rhi)

    suite_parser = subparsers.add_parser("experiment-suite", help="Run the paper-grade RHI experiments 1, 2, and 3")
    suite_parser.add_argument("--tasks", default="preferential_bo,discover_unique,extreme_properties")
    suite_parser.add_argument("--experiments", default="1,2,3", help="Comma-separated experiment IDs")
    suite_parser.add_argument("--seeds", default="1,7,13,21,42")
    suite_parser.add_argument("--n-per-task", type=int, default=300)
    suite_parser.add_argument("--data-dir", help="Directory containing historical task JSONL files")
    suite_parser.add_argument("--train-fraction", type=float, default=0.6)
    suite_parser.add_argument("--val-fraction", type=float, default=0.2)
    suite_parser.add_argument("--feedback-fraction", type=float, default=0.15)
    suite_parser.add_argument("--acceptance-fraction", type=float, default=0.1)
    suite_parser.add_argument("--alpha", type=float, default=0.1)
    suite_parser.add_argument("--budget-fraction", type=float, default=0.1)
    suite_parser.add_argument("--min-coverage", type=float, default=0.1)
    suite_parser.add_argument("--rhi-iterations", type=int, default=3)
    suite_parser.add_argument("--epochs", type=int, default=240)
    suite_parser.add_argument("--learning-rate", type=float, default=0.08)
    suite_parser.add_argument("--l2", type=float, default=0.001)
    suite_parser.add_argument("--epsilon", type=float, default=0.01)
    suite_parser.add_argument(
        "--direct-judge-model",
        help="Enable the one-shot LLM direct-as-judge baseline with this model; requires OPENAI_API_KEY",
    )
    suite_parser.add_argument(
        "--direct-judge-base-url",
        help="OpenAI-compatible API base URL (default: OPENAI_BASE_URL or api.openai.com)",
    )
    suite_parser.add_argument(
        "--direct-judge-cache",
        default="runs/direct_judge_cache/scores.json",
        help="JSON score cache for direct judge calls",
    )
    suite_parser.add_argument("--direct-judge-timeout", type=float, default=90.0)
    suite_parser.add_argument("--direct-judge-retries", type=int, default=3)
    suite_parser.add_argument("--out", required=True)
    suite_parser.add_argument("--markdown-out")
    suite_parser.set_defaults(func=_cmd_experiment_suite)

    return parser


def _cmd_make_benchmark(args: argparse.Namespace) -> int:
    records = make_records(args.benchmark, n=args.n, seed=args.seed)
    write_jsonl(records, args.out)
    print(f"wrote {len(records)} records to {args.out}")
    return 0


def _load_dataset(path: str):
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        payload = read_json(path)
        if isinstance(payload, list):
            from .schema import ActionRecord

            return [ActionRecord.from_json(item) for item in payload]
    raise ValueError(f"unsupported dataset format: {path}")


def _cmd_train(args: argparse.Namespace) -> int:
    records = _load_dataset(args.data)
    train_records, val_records, test_records = split_records(records, seed=args.seed, train_fraction=args.train_fraction, val_fraction=args.val_fraction)
    gate = train_gate(
        train_records,
        val_records,
        alpha=args.alpha,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        lineage_weight=args.lineage_weight,
    )
    write_json(gate.to_json(), args.out)
    report = {
        "sizes": {"train": len(train_records), "val": len(val_records), "test": len(test_records)},
        "alpha": args.alpha,
        "seed": args.seed,
        "gate": gate.to_json(),
        "test": evaluate_gate(test_records, gate),
    }
    if args.report:
        write_json(report, args.report)
    print(f"trained gate on {len(train_records)} train / {len(val_records)} val records; wrote model to {args.out}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    records = _load_dataset(args.data)
    gate = TrainedGate.from_json(read_json(args.model))
    report = evaluate_gate(records, gate, budget_fraction=args.budget_fraction)
    write_json(report, args.out)
    print(f"wrote evaluation report to {args.out}")
    return 0


def _cmd_campaign(args: argparse.Namespace) -> int:
    benchmarks = list(args.benchmark or [])
    if args.benchmarks:
        benchmarks.extend(_comma_list(args.benchmarks, str))
    if not benchmarks:
        benchmarks = sorted(BENCHMARK_BUILDERS)
    seeds = _comma_list(args.seeds, int)
    if not seeds:
        seeds = [0]
    config = CampaignConfig(
        benchmarks=benchmarks,
        seeds=seeds,
        n=args.n,
        alpha=args.alpha,
        budget_fraction=args.budget_fraction,
        lineage_weight=args.lineage_weight,
    )
    report = save_campaign_report(config, args.out)
    print(f"wrote campaign report with {report['aggregate']['n_runs']} runs to {args.out}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    records = make_records(args.benchmark, n=args.n, seed=args.seed)
    train_records, val_records, test_records = split_records(records, seed=args.seed)
    gate = train_gate(train_records, val_records, alpha=args.alpha)
    report = {
        "benchmark": args.benchmark,
        "seed": args.seed,
        "sizes": {"train": len(train_records), "val": len(val_records), "test": len(test_records)},
        "test": evaluate_gate(test_records, gate, budget_fraction=args.budget_fraction),
    }
    print(report)
    return 0


def _cmd_paper_bootstrap(args: argparse.Namespace) -> int:
    report = run_paper_bootstrap_experiment(
        args.data,
        workdir=args.workdir,
        seed=args.seed,
        alpha=args.alpha,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        budget_fraction=args.budget_fraction,
    )
    selected = report["selected_version"]
    metrics = report["versions"][selected]["test"]["metrics"]
    print(
        "wrote paper bootstrap experiment to "
        f"{args.workdir}; selected={selected}; "
        f"test_selective_accuracy={metrics['selective_accuracy']:.3f}; "
        f"test_selective_risk={metrics['selective_risk']:.3f}; "
        f"coverage={metrics['coverage']:.3f}"
    )
    return 0


def _cmd_rhi(args: argparse.Namespace) -> int:
    records = _load_dataset(args.data)
    report = train_rhi(
        records,
        iterations=args.iterations,
        seed=args.seed,
        alpha=args.alpha,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        budget_fraction=args.budget_fraction,
        min_coverage=args.min_coverage,
        epsilon=args.epsilon,
    )
    write_json(report, args.out)
    metrics = report["test"]["metrics"]
    print(
        f"wrote RHI report to {args.out}; "
        f"final={report['final_harness']['name']}; "
        f"risk={metrics['selective_risk']:.3f}; coverage={metrics['coverage']:.3f}"
    )
    return 0


def _cmd_experiment_suite(args: argparse.Namespace) -> int:
    config = ExperimentSuiteConfig(
        tasks=tuple(_comma_list(args.tasks, str)),
        experiments=tuple(_comma_list(args.experiments, int)),
        seeds=tuple(_comma_list(args.seeds, int)),
        n_per_task=args.n_per_task,
        data_dir=args.data_dir,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        feedback_fraction=args.feedback_fraction,
        acceptance_fraction=args.acceptance_fraction,
        alpha=args.alpha,
        min_coverage=args.min_coverage,
        budget_fraction=args.budget_fraction,
        rhi_iterations=args.rhi_iterations,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        rhi_epsilon=args.epsilon,
        direct_judge_model=args.direct_judge_model,
        direct_judge_base_url=args.direct_judge_base_url,
        direct_judge_cache=args.direct_judge_cache,
        direct_judge_timeout=args.direct_judge_timeout,
        direct_judge_retries=args.direct_judge_retries,
    )
    report = save_experiment_suite(config, args.out, args.markdown_out)
    print(
        f"wrote experiments 1/2/3 to {args.out}; "
        f"single={report['summary']['n_single_runs']}; "
        f"transfer={report['summary']['n_transfer_runs']}; "
        f"joint={report['summary']['n_joint_runs']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
