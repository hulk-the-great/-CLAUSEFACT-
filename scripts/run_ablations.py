"""Train/evaluate independent ablation models; never uses paper-reported numbers.

Example:
    python scripts/run_ablations.py --train data/example/train.jsonl \\
      --val data/example/val.jsonl --test data/example/test.jsonl \\
      --out outputs/ablations --epochs 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--train", required=True)
p.add_argument("--val", required=True)
p.add_argument("--test", required=True)
p.add_argument("--out", required=True)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--backbone", default="distilbert-base-uncased")
p.add_argument("--device", default="auto")
a = p.parse_args()
root = Path(a.out)
root.mkdir(parents=True, exist_ok=True)
variants = {
    "full": [],
    "without_graph": ["--no-graph"],
    "without_contradiction": ["--no-contradiction"],
    "without_calibration": ["--no-calibration"],
    "without_importance": ["--no-importance"],
    "without_clauses": ["--no-clauses"],
}


def doc_only(src: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8") as reader, dest.open("w", encoding="utf-8") as writer:
        for line in reader:
            if line.strip():
                row = json.loads(line)
                row.pop("clause_labels", None)
                row.pop("contradiction_labels", None)
                writer.write(json.dumps(row) + "\n")
    return str(dest)


results = {}
for name, flags in variants.items():
    variant_dir = root / name
    train_data, val_data = a.train, a.val
    if name == "without_clauses":
        train_data = doc_only(a.train, variant_dir / "train_document_only.jsonl")
        val_data = doc_only(a.val, variant_dir / "val_document_only.jsonl")
    test_data = doc_only(a.test, variant_dir / "test_document_only.jsonl") if name == "without_clauses" else a.test
    if name == "without_calibration":
        flags = flags + ["--lambda-cal", "0"]
    if name == "without_contradiction":
        flags = flags + ["--lambda-contra", "0"]
    train_cmd = [sys.executable, "-m", "clausefact", "train",
                 "--train", train_data, "--val", val_data,
                 "--out", str(variant_dir / "model"), "--epochs", str(a.epochs),
                 "--backbone", a.backbone, "--device", a.device, *flags]
    subprocess.run(train_cmd, check=True)
    eval_cmd = [sys.executable, "-m", "clausefact", "evaluate",
                "--data", test_data, "--checkpoint", str(variant_dir / "model"),
                "--out", str(variant_dir / "evaluation"), "--device", a.device]
    subprocess.run(eval_cmd, check=True)
    metrics = json.loads((variant_dir / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
    results[name] = metrics
(root / "ablation_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(f"Ablation metrics saved: {root / 'ablation_metrics.json'}")
