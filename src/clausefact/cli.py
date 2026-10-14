"""Command-line interface. `inspect` works without PyTorch/Transformers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_jsonl, write_json
from .graph import build_adjacency, edge_count
from .preprocessing import build_evidence_spans, decompose_clauses, fact_signals


def _read_input(args: argparse.Namespace, name: str) -> str:
    inline = getattr(args, name + "_text", None)
    filename = getattr(args, name + "_file", None)
    if inline is not None:
        return inline
    return Path(filename).read_text(encoding="utf-8")


def _add_text_args(p: argparse.ArgumentParser) -> None:
    s = p.add_mutually_exclusive_group(required=True)
    s.add_argument("--source-text", help="Source article as inline text")
    s.add_argument("--source-file", help="Path to UTF-8 source article")
    y = p.add_mutually_exclusive_group(required=True)
    y.add_argument("--summary-text", help="Generated summary as inline text")
    y.add_argument("--summary-file", help="Path to UTF-8 generated summary")
    p.add_argument("--out", help="Optional JSON output file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clausefact", description="Clause-level evidence-graph factuality verifier"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Offline decomposition and graph preview")
    _add_text_args(inspect)
    inspect.add_argument("--window-size", type=int, default=2)
    inspect.add_argument("--max-spans", type=int, default=80)

    train = sub.add_parser("train", help="Fine-tune a backbone on labeled JSONL data")
    train.add_argument("--train", required=True, help="Train JSONL")
    train.add_argument("--val", required=True, help="Validation JSONL")
    train.add_argument("--out", required=True, help="Checkpoint directory")
    train.add_argument("--backbone", default="distilbert-base-uncased")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--grad-accum", type=int, default=1)
    train.add_argument("--patience", type=int, default=3)
    train.add_argument("--device", default="auto", help="auto, cpu, cuda")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-tokens", type=int, default=192)
    train.add_argument("--max-spans", type=int, default=80)
    train.add_argument("--window-size", type=int, default=2)
    train.add_argument("--no-clauses", action="store_true")
    train.add_argument("--no-graph", action="store_true")
    train.add_argument("--no-contradiction", action="store_true")
    train.add_argument("--no-calibration", action="store_true")
    train.add_argument("--no-importance", action="store_true")
    train.add_argument("--lambda-doc", type=float, default=1.)
    train.add_argument("--lambda-clause", type=float, default=1.)
    train.add_argument("--lambda-contra", type=float, default=1.)
    train.add_argument("--lambda-cal", type=float, default=.1)

    evaluate = sub.add_parser("evaluate", help="Evaluate checkpoint on labeled JSONL")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--out", required=True, help="Output metrics/report directory")
    evaluate.add_argument("--device", default="auto")

    predict = sub.add_parser("predict", help="Score a summary using a TRAINED checkpoint")
    _add_text_args(predict)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        source, summary = _read_input(args, "source"), _read_input(args, "summary")
        clauses = decompose_clauses(summary)
        spans = build_evidence_spans(source, window_size=args.window_size,
                                     max_spans=args.max_spans)
        adj = build_adjacency(spans)
        result = {
            "clauses": [{"text": c, "signals": vars(fact_signals(c))} for c in clauses],
            "evidence_spans": spans,
            "graph_edge_count": edge_count(adj),
            "graph_adjacency": adj,
            "note": "Offline preprocessing only. No factuality prediction without a trained checkpoint.",
        }
    elif args.command == "train":
        from .engine import train
        from .model import ModelConfig
        cfg = ModelConfig(
            backbone_name=args.backbone, max_tokens=args.max_tokens,
            max_spans=args.max_spans, window_size=args.window_size,
            use_clauses=not args.no_clauses, use_graph=not args.no_graph,
            use_contradiction=not args.no_contradiction,
            use_calibration=not args.no_calibration,
            use_importance=not args.no_importance,
        )
        result = train(
            args.train, args.val, args.out, cfg=cfg, epochs=args.epochs,
            learning_rate=args.learning_rate, grad_accum=args.grad_accum,
            patience=args.patience, device=args.device, seed=args.seed,
            lambda_doc=args.lambda_doc, lambda_clause=args.lambda_clause,
            lambda_contra=args.lambda_contra, lambda_cal=args.lambda_cal,
        )
    elif args.command == "evaluate":
        from .engine import load_checkpoint, predict_items
        items = load_jsonl(args.data, require_label=True)
        model, tokenizer = load_checkpoint(args.checkpoint, device=args.device)
        metrics, predictions = predict_items(model, tokenizer, items)
        root = Path(args.out)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "metrics.json", metrics)
        with (root / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        result = {"metrics": metrics, "predictions_path": str(root / "predictions.jsonl")}
    elif args.command == "predict":
        from .engine import load_checkpoint
        from .model import serialize_prediction
        from .data import validate_example
        source, summary = _read_input(args, "source"), _read_input(args, "summary")
        validate_example({"source": source, "summary": summary})
        model, tokenizer = load_checkpoint(args.checkpoint, device=args.device)
        import torch
        with torch.no_grad():
            result = serialize_prediction(model(source, summary, tokenizer))
    else:
        raise AssertionError("Unreachable command")
    if getattr(args, "out", None) and args.command in {"inspect", "predict"}:
        write_json(args.out, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
