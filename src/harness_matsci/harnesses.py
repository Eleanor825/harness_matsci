from __future__ import annotations

import copy
import json
import re
from typing import Any

from .features import DEFAULT_FEATURES
from .schema import ALLOWED_ROUTES, HarnessSpec


DEFAULT_HARNESS = {
    "name": "H0_scientific_action_gate",
    "roles": [
        {
            "id": "evidence_auditor",
            "kind": "adviser",
            "instruction": "Audit whether the proposed scientific action is grounded in visible evidence and not merely confident language.",
            "contract": ["evidence_support", "evidence_conflict", "source_reliability", "perturbation_stability"],
        },
        {
            "id": "uncertainty_gate",
            "kind": "builder",
            "instruction": "Estimate action reliability and choose proceed only when selective risk is controlled.",
            "contract": ["calibrated_probability", "target_selective_risk", "cost_sensitive_threshold"],
        },
        {
            "id": "fallback_router",
            "kind": "reviewer",
            "instruction": "If the action is not reliable enough, route it to retrieve, simulate, ask, experiment, or abstain based on reducible uncertainty.",
            "contract": ["route_reason", "reducible_uncertainty", "irreversibility", "cost"],
        },
    ],
    "required_features": list(DEFAULT_FEATURES),
    "proceed_routes": ["proceed", "experiment"],
    "fallback_routes": ["retrieve_more", "simulate", "ask_expert", "abstain"],
    "target_selective_risk": 0.1,
    "gates": [
        "Every decision has an action-level reliability probability.",
        "Proceed thresholds are calibrated on validation data, not hand tuned on test data.",
        "Low reliability maps to an explicit fallback route, not generic refusal.",
    ],
}


STATIC_HIGH_EFFORT_HARNESS = {
    **copy.deepcopy(DEFAULT_HARNESS),
    "name": "static_high_effort_scientific_gate",
    "target_selective_risk": 0.05,
    "gates": DEFAULT_HARNESS["gates"]
    + [
        "High-cost or low-reversibility actions require stricter thresholds.",
        "OOD and evidence conflict trigger verification routes before proceed.",
    ],
}


def harness_text(harness: dict[str, Any]) -> str:
    return json.dumps(harness, sort_keys=True, indent=2)


def to_spec(harness: dict[str, Any]) -> HarnessSpec:
    return HarnessSpec(
        name=str(harness.get("name", "scientific_action_gate")),
        required_features=[str(x) for x in harness.get("required_features", DEFAULT_FEATURES)],
        proceed_routes=[route for route in harness.get("proceed_routes", ["proceed"]) if route in ALLOWED_ROUTES],
        fallback_routes=[route for route in harness.get("fallback_routes", ["abstain"]) if route in ALLOWED_ROUTES],
        target_selective_risk=float(harness.get("target_selective_risk", 0.1)),
    )


def validate_harness(candidate: Any, previous: dict[str, Any] | None = None, iteration: int = 0) -> tuple[dict[str, Any], str]:
    """Constrain model-proposed harness revisions to a safe schema.

    This mirrors the recursive-harness code path but uses scientific action
    contracts instead of repository-building contracts.
    """
    base = copy.deepcopy(previous or DEFAULT_HARNESS)
    if not isinstance(candidate, dict):
        return deterministic_fallback(base, iteration), "fallback_non_object"

    cleaned = copy.deepcopy(base)
    cleaned["name"] = str(candidate.get("name", f"H{iteration}_scientific_gate"))[:100]
    cleaned["target_selective_risk"] = min(0.5, max(0.0, float(candidate.get("target_selective_risk", base["target_selective_risk"]))))

    roles = candidate.get("roles", base.get("roles", []))
    if isinstance(roles, list):
        cleaned_roles = []
        counts = {"adviser": 0, "builder": 0, "reviewer": 0}
        for raw in roles:
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("kind", "")).lower()
            if kind not in counts:
                continue
            if kind == "builder" and counts[kind] >= 1:
                continue
            if kind in {"adviser", "reviewer"} and counts[kind] >= 3:
                continue
            role_id = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("id", kind)).lower()).strip("_")
            contract = raw.get("contract", [])
            if not isinstance(contract, list):
                contract = [contract]
            cleaned_roles.append(
                {
                    "id": role_id or kind,
                    "kind": kind,
                    "instruction": str(raw.get("instruction", ""))[:900],
                    "contract": [str(item)[:160] for item in contract[:10]],
                }
            )
            counts[kind] += 1
        if counts["builder"] == 1 and cleaned_roles:
            cleaned["roles"] = cleaned_roles

    for key, default in [
        ("required_features", DEFAULT_FEATURES),
        ("proceed_routes", ["proceed", "experiment"]),
        ("fallback_routes", ["retrieve_more", "simulate", "ask_expert", "abstain"]),
        ("gates", base.get("gates", [])),
    ]:
        raw = candidate.get(key, default)
        if isinstance(raw, list):
            cleaned[key] = [str(item)[:220] for item in raw[:20]]

    raw_hops = candidate.get("hops", base.get("hops", []))
    if isinstance(raw_hops, list):
        cleaned_hops = []
        role_ids = {str(role.get("id")) for role in cleaned.get("roles", [])}
        allowed_endpoints = role_ids | {"orchestrator", "ask_expert"}
        for raw in raw_hops[:12]:
            if not isinstance(raw, dict):
                continue
            source = str(raw.get("from", ""))[:100]
            target = str(raw.get("to", ""))[:100]
            purpose = str(raw.get("purpose", ""))[:300]
            if source in allowed_endpoints and target in allowed_endpoints and purpose:
                cleaned_hops.append({"from": source, "to": target, "purpose": purpose})
        if cleaned_hops:
            cleaned["hops"] = cleaned_hops

    try:
        spec = to_spec(cleaned)
        if not spec.proceed_routes or not spec.fallback_routes:
            return deterministic_fallback(base, iteration), "fallback_empty_routes"
    except Exception:
        return deterministic_fallback(base, iteration), "fallback_invalid_spec"
    return cleaned, "model"


def deterministic_fallback(previous: dict[str, Any], iteration: int) -> dict[str, Any]:
    result = copy.deepcopy(previous)
    result["name"] = f"H{iteration}_fallback_scientific_gate"
    result["required_features"] = list(dict.fromkeys(result.get("required_features", []) + DEFAULT_FEATURES))
    result["fallback_routes"] = list(dict.fromkeys(result.get("fallback_routes", []) + ["retrieve_more", "simulate", "ask_expert", "abstain"]))
    result["gates"] = list(
        dict.fromkeys(
            result.get("gates", [])
            + [
                "Evidence conflict must lower reliability.",
                "High OOD must trigger verification unless reliability is calibrated.",
            ]
        )
    )
    return result


def structural_metrics(harness: dict[str, Any]) -> dict[str, float]:
    roles = harness.get("roles", [])
    contracts = [item for role in roles for item in role.get("contract", [])]
    return {
        "roles": float(len(roles)),
        "contract_fields": float(len(contracts)),
        "required_features": float(len(harness.get("required_features", []))),
        "routes": float(len(harness.get("proceed_routes", [])) + len(harness.get("fallback_routes", []))),
        "gates": float(len(harness.get("gates", []))),
        "hops": float(len(harness.get("hops", []))),
    }
