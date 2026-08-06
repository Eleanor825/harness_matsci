from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import ActionRecord


def write_jsonl(records: Iterable[ActionRecord], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[ActionRecord]:
    records: list[ActionRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(ActionRecord.from_json(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid ActionRecord at {path}:{line_number}: {exc}") from exc
    return records


def write_json(payload: object, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))

