"""Convert a user-labeled CSV into validated JSONL splits.

Required columns: source, summary, label, split
Optional: id, clause_labels, contradiction_labels (JSON arrays).
Do not assign labels based on raw XSum/CNN-DM data: they are not fact labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausefact.data import validate_example

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
with Path(a.input).open(newline="", encoding="utf-8") as f:
    for index, row in enumerate(csv.DictReader(f)):
        split = row.get("split", "").strip().lower()
        if split not in rows:
            raise ValueError(f"CSV row {index + 2}: split must be train, val, or test")
        item = {"id": row.get("id") or str(index),
                "source": row["source"], "summary": row["summary"],
                "label": int(row["label"])}
        for name in ("clause_labels", "contradiction_labels"):
            if row.get(name, "").strip():
                item[name] = json.loads(row[name])
        validate_example(item, require_label=True)
        rows[split].append(item)
for split, items in rows.items():
    with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(split, len(items))
