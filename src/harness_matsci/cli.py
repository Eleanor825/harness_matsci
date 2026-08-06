from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .benchmarks import BENCHMARK_BUILDERS, make_records
from .campaign import CampaignConfig, save_campaign_report
from .io import read_json, read_jsonl, write_json, write_jsonl
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
