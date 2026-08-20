from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from harness_matsci.benchmarks import make_records
from harness_matsci.direct_judge import LLMDirectJudge
from harness_matsci.voi import VOI_SEED_HARNESS, evaluate_voi, fit_voi_model, train_voi_rhi

from run_self_report import add_features, direct_metrics, permute_confidence, split_feedback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-per-task", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    records = []
    for task in ("preferential_bo", "discover_unique", "extreme_properties"):
        records.extend(make_records(task, n=args.n_per_task, seed=args.seed))
    cache = Path(args.out).with_suffix(".responses.json")
    judge = LLMDirectJudge.from_env(
        model=args.model,
        cache_path=cache,
        timeout=args.timeout,
        max_retries=1,
    )
    cached = dict(judge._cache)
    pending = [record for record in records if record.record_id not in cached]
    scores = dict(cached)
    errors = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(judge._score_record, record): record for record in pending}
        for future in as_completed(futures):
            record = futures[future]
            try:
                scores[record.record_id] = float(future.result())
            except Exception as error:
                errors.append(f"{record.record_id}: {error}")
    if errors:
        raise RuntimeError(f"{args.model} failed on {len(errors)} records; first={errors[0]}")
    judge._cache.update(scores)
    judge._save_cache()
    outputs = {
        record.record_id: {
            "confidence": float(scores[record.record_id]),
            "expected_utility": float(scores[record.record_id]),
            "parse_failed": 0,
            "raw": "provider p_success",
        }
        for record in records
    }
    augmented = add_features(records, outputs)
    train, feedback, acceptance, test = split_feedback(augmented, args.seed)
    result = {
        "protocol": {
            "model": args.model,
            "signal": "closed_model_self_report_confidence",
            "records": len(records),
            "train": len(train), "feedback": len(feedback),
            "acceptance": len(acceptance), "test": len(test),
            "budget_fraction": 0.1,
            "note": "provider p_success is treated as model-reported confidence; not intrinsic uncertainty",
        },
        "direct_closed_model_judge": direct_metrics(test, outputs),
    }
    harness = copy.deepcopy(VOI_SEED_HARNESS)
    harness["required_features"] += ["llm_self_report_confidence", "llm_self_report_uncertainty", "llm_self_report_utility"]
    harness["utility_features"] += ["llm_self_report_confidence", "llm_self_report_uncertainty", "llm_self_report_utility"]
    harness["name"] = f"H0_{args.model}_self_report_voi"
    harness["decision_mode"] = "voi"
    harness["epistemic_weight"] = 0.35
    harness["failure_cost_weight"] = 0.25
    harness["execute_cost_weight"] = 0.15
    harness["verification_cost_weight"] = 0.10
    harness["verification_uncertainty_floor"] = 0.25
    model = fit_voi_model(train, feedback, harness, epochs=80)
    result["static_self_report_voi"] = evaluate_voi(test, model, budget_fraction=0.1)
    result["self_report_voi_rhi_guarded"] = train_voi_rhi(
        train, feedback, acceptance, test, iterations=3, seed=args.seed,
        epochs=80, component="full", acceptance_policy="robust_guarded", initial_harness=harness,
    )
    permuted = permute_confidence(augmented, args.seed + 99)
    by_id = {record.record_id: record for record in permuted}
    p_train = [by_id[r.record_id] for r in train]
    p_feedback = [by_id[r.record_id] for r in feedback]
    p_test = [by_id[r.record_id] for r in test]
    p_model = fit_voi_model(p_train, p_feedback, harness, epochs=80)
    result["permuted_self_report_voi"] = evaluate_voi(p_test, p_model, budget_fraction=0.1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "model": args.model,
        "direct": result["direct_closed_model_judge"],
        "static": result["static_self_report_voi"],
        "rhi_test": result["self_report_voi_rhi_guarded"]["test"],
        "permuted": result["permuted_self_report_voi"],
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
