from __future__ import annotations

from .discover_unique import make_discover_unique_records
from .extreme_properties import make_extreme_property_records
from .preferential_bo import make_preferential_bo_records


BENCHMARK_BUILDERS = {
    "preferential_bo": make_preferential_bo_records,
    "discover_unique": make_discover_unique_records,
    "extreme_properties": make_extreme_property_records,
}


def make_records(benchmark: str, *, n: int = 300, seed: int = 0):
    try:
        builder = BENCHMARK_BUILDERS[benchmark]
    except KeyError as exc:
        choices = ", ".join(sorted(BENCHMARK_BUILDERS))
        raise ValueError(f"unknown benchmark {benchmark!r}; choose one of: {choices}") from exc
    return builder(n=n, seed=seed)


__all__ = [
    "BENCHMARK_BUILDERS",
    "make_records",
    "make_preferential_bo_records",
    "make_discover_unique_records",
    "make_extreme_property_records",
]
