"""Validate a CLAUSEFACT JSONL dataset and print class balance.

Usage: python scripts/check_dataset.py data/example/train.jsonl
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausefact.data import load_jsonl

p = argparse.ArgumentParser()
p.add_argument("jsonl")
args = p.parse_args()
items = load_jsonl(args.jsonl, require_label=True)
print(f"Examples: {len(items)}")
print(f"Document labels: {dict(Counter(item['label'] for item in items))}")
print(f"With clause labels: {sum('clause_labels' in x for x in items)}")
print(f"With contradiction labels: {sum('contradiction_labels' in x for x in items)}")
