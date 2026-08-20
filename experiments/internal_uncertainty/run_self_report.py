from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from harness_matsci.benchmarks import make_records
from harness_matsci.metrics import discovery_gain, expected_calibration_error
from harness_matsci.voi import VOI_SEED_HARNESS, evaluate_voi, fit_voi_model, train_voi_rhi


def prompt_for(record) -> str:
    evidence = "; ".join(record.evidence) if record.evidence else "none"
    return (
        "You are a scientific action judge. Decide whether executing the proposed "
        "action is worthwhile now. Use only the visible information. Return JSON only, "
        "with no markdown, using exactly this schema: "
        '{"decision":"execute|defer_or_stop","confidence":0.0,"expected_utility":0.0}. '
        "confidence means your probability that executing the proposed action is worthwhile, "
        "not confidence in the writing. confidence and expected_utility must be numbers in [0,1].\n\n"
        f"Context: {record.visible_context}\n"
        f"Evidence: {evidence}\n"
        f"Candidate action: {record.candidate_action}\n"
        "JSON:"
    )


def parse_json(text: str) -> dict[str, float | str]:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return {"decision": "defer_or_stop", "confidence": 0.5, "expected_utility": 0.5, "parse_failed": 1}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"decision": "defer_or_stop", "confidence": 0.5, "expected_utility": 0.5, "parse_failed": 1}
    confidence = payload.get("confidence", 0.5)
    utility = payload.get("expected_utility", confidence)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.5
    try:
        utility = max(0.0, min(1.0, float(utility)))
    except (TypeError, ValueError):
        utility = confidence
    decision = str(payload.get("decision", "defer_or_stop"))
    if decision not in {"execute", "defer_or_stop"}:
        decision = "defer_or_stop"
    return {"decision": decision, "confidence": confidence, "expected_utility": utility, "parse_failed": 0}


def run_model(records, model_name: str, batch_size: int, max_new_tokens: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs = {}
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = []
        for record in batch:
            messages = [{"role": "user", "content": prompt_for(record)}]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=768)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        prompt_len = inputs["input_ids"].shape[1]
        for record, row in zip(batch, generated):
            raw = tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
            parsed = parse_json(raw)
            outputs[record.record_id] = {**parsed, "raw": raw}
        print(f"scored {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    return outputs


def split_feedback(records, seed: int):
    val = [record for record in records if record.split == "val"]
    rng = random.Random(seed)
    rng.shuffle(val)
    midpoint = max(1, len(val) // 2)
    return [record for record in records if record.split == "train"], val[:midpoint], val[midpoint:], [record for record in records if record.split == "test"]


def add_features(records, outputs):
    augmented = []
    for record in records:
        item = outputs[record.record_id]
        features = dict(record.features)
        features["llm_self_report_confidence"] = float(item["confidence"])
        features["llm_self_report_uncertainty"] = 1.0 - float(item["confidence"])
        features["llm_self_report_utility"] = float(item["expected_utility"])
        augmented.append(replace(record, features=features))
    return augmented


def direct_metrics(records, outputs):
    confidence = [float(outputs[r.record_id]["confidence"]) for r in records]
    labels = [int(r.label) for r in records]
    utilities = [float(r.utility) for r in records]
    budget = max(1, int(len(records) * 0.1))
    ranking = discovery_gain(labels, utilities, confidence, budget)
    correctness = [int((score >= 0.5) == bool(label)) for score, label in zip(confidence, labels)]
    return {
        **ranking,
        "accuracy": mean((score >= 0.5) == bool(label) for score, label in zip(confidence, labels)),
        "ece_on_correctness": expected_calibration_error(correctness, confidence),
        "parse_failure_rate": mean(float(outputs[r.record_id]["parse_failed"]) for r in records),
    }


def permute_confidence(records, seed: int):
    rng = random.Random(seed)
    groups = {}
    for record in records:
        groups.setdefault(record.benchmark, []).append(record)
    result = []
    for group in groups.values():
        values = [record.features["llm_self_report_confidence"] for record in group]
        rng.shuffle(values)
        for record, confidence in zip(group, values):
            features = dict(record.features)
            features["llm_self_report_confidence"] = confidence
            features["llm_self_report_uncertainty"] = 1.0 - confidence
            result.append(replace(record, features=features))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n-per-task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out", default="runs/open_model_self_report_v1/summary.json")
    args = parser.parse_args()

    records = []
    for task in ("preferential_bo", "discover_unique", "extreme_properties"):
        records.extend(make_records(task, n=args.n_per_task, seed=args.seed))
    cache_path = Path(args.out).with_suffix(".responses.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        outputs = raw["outputs"]
    else:
        outputs = run_model(records, args.model, args.batch_size, args.max_new_tokens)
        cache_path.write_text(json.dumps({"model": args.model, "outputs": outputs}, indent=2), encoding="utf-8")

    augmented = add_features(records, outputs)
    train, feedback, acceptance, test = split_feedback(augmented, args.seed)
    result = {
        "protocol": {
            "model": args.model,
            "signal": "open_model_self_report_confidence",
            "tasks": ["preferential_bo", "discover_unique", "extreme_properties"],
            "records": len(records), "train": len(train), "feedback": len(feedback),
            "acceptance": len(acceptance), "test": len(test), "budget_fraction": 0.1,
            "note": "model-reported confidence; not intrinsic logit uncertainty",
        },
        "direct_open_model_judge": direct_metrics(test, outputs),
    }

    harness = copy.deepcopy(VOI_SEED_HARNESS)
    harness["required_features"] += ["llm_self_report_confidence", "llm_self_report_uncertainty", "llm_self_report_utility"]
    harness["utility_features"] += ["llm_self_report_confidence", "llm_self_report_uncertainty", "llm_self_report_utility"]
    harness["name"] = "H0_open_self_report_voi"
    harness["decision_mode"] = "voi"
    harness["epistemic_weight"] = 0.35
    harness["failure_cost_weight"] = 0.25
    harness["execute_cost_weight"] = 0.15
    harness["verification_cost_weight"] = 0.10
    harness["verification_uncertainty_floor"] = 0.25
    model = fit_voi_model(train, feedback, harness, epochs=80)
    result["static_self_report_voi"] = evaluate_voi(test, model, budget_fraction=0.1)
    result["self_report_voi_rhi_guarded"] = train_voi_rhi(train, feedback, acceptance, test, iterations=3, seed=args.seed, epochs=80, component="full", acceptance_policy="robust_guarded", initial_harness=harness)
    result["self_report_voi_rhi_always_accept"] = train_voi_rhi(train, feedback, acceptance, test, iterations=3, seed=args.seed, epochs=80, component="full", acceptance_policy="always_accept", initial_harness=harness)

    permuted = permute_confidence(augmented, args.seed + 99)
    by_id = {record.record_id: record for record in permuted}
    p_train = [by_id[r.record_id] for r in train]
    p_feedback = [by_id[r.record_id] for r in feedback]
    p_test = [by_id[r.record_id] for r in test]
    p_model = fit_voi_model(p_train, p_feedback, harness, epochs=80)
    result["permuted_self_report_voi"] = evaluate_voi(p_test, p_model, budget_fraction=0.1)

    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"model": args.model, "records": len(records), "direct": result["direct_open_model_judge"], "static": result["static_self_report_voi"], "rhi": result["self_report_voi_rhi_guarded"]["test"], "permuted": result["permuted_self_report_voi"], "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
