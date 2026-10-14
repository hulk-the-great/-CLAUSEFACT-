"""Validated JSONL loading. Labels: 1=faithful/support, 0=unfaithful."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .preprocessing import decompose_clauses


def validate_example(item: dict[str, Any], *, require_label: bool = False) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Each example must be a JSON object")
    for name in ("source", "summary"):
        if not isinstance(item.get(name), str) or not item[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    if require_label and "label" not in item:
        raise ValueError("Labeled data requires 'label': 0 or 1")
    if "label" in item and item["label"] not in (0, 1):
        raise ValueError("label must be 0 or 1")
    n = len(decompose_clauses(item["summary"]))
    if not n:
        raise ValueError("summary has no detectable clauses")
    for key in ("clause_labels", "contradiction_labels"):
        if key in item:
            labels = item[key]
            if not isinstance(labels, list) or len(labels) != n:
                raise ValueError(f"{key} length must equal split clause count ({n})")
            if any(x not in (0, 1) for x in labels):
                raise ValueError(f"{key} values must be 0 or 1")
    return item


def load_jsonl(path: str | Path, *, require_label: bool = False) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            if line.strip():
                try:
                    examples.append(validate_example(json.loads(line), require_label=require_label))
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{path}:{line_num}: {exc}") from exc
    if not examples:
        raise ValueError(f"Empty JSONL dataset: {path}")
    return examples


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
