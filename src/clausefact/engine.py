"""Checkpointing, validation, training, and offline scoring."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from .data import load_jsonl, write_json
from .metrics import classification_metrics
from .model import ClauseFactModel, ModelConfig, compute_loss, serialize_prediction


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    return torch.device(name)


def save_checkpoint(outdir: str | Path, model: ClauseFactModel, tokenizer: Any,
                    *, training_meta: dict[str, Any]) -> None:
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(root / "backbone")
    tokenizer.save_pretrained(root / "tokenizer")
    torch.save(model.state_dict(), root / "model.pt")
    write_json(root / "config.json", model.cfg.to_dict())
    write_json(root / "training_meta.json", training_meta)


def load_checkpoint(path: str | Path, *, device: str = "auto") -> tuple[ClauseFactModel, Any]:
    root = Path(path)
    cfg = ModelConfig(**json.loads((root / "config.json").read_text(encoding="utf-8")))
    model = ClauseFactModel(cfg, backbone_path=str(root / "backbone"))
    state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(resolve_device(device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
    return model, tokenizer


@torch.no_grad()
def predict_items(model: ClauseFactModel, tokenizer: Any,
                  items: list[dict[str, Any]]) -> tuple[dict, list[dict]]:
    model.eval()
    rows: list[dict] = []
    y, probs = [], []
    clause_y, clause_scores = [], []
    for i, item in enumerate(items):
        pred = model(item["source"], item["summary"], tokenizer)
        row = serialize_prediction(pred, id=str(item.get("id", i)))
        if "label" in item:
            y.append(int(item["label"]))
            probs.append(row["document_faithfulness_probability"])
            row["true_label"] = int(item["label"])
        if "clause_labels" in item:
            clause_y.extend(item["clause_labels"])
            clause_scores.extend(c["faithfulness_probability"] for c in row["clauses"])
        rows.append(row)
    metrics = classification_metrics(y, probs) if y else {"n": len(items), "note": "unlabeled examples"}
    if clause_y:
        metrics["clause_level"] = classification_metrics(clause_y, clause_scores)
    return metrics, rows


def train(train_path: str, val_path: str, outdir: str, *,
          cfg: ModelConfig | None = None, epochs: int = 5,
          learning_rate: float = 2e-5, grad_accum: int = 1,
          patience: int = 3, device: str = "auto", seed: int = 42,
          lambda_doc: float = 1.0, lambda_clause: float = 1.0,
          lambda_contra: float = 1.0, lambda_cal: float = 0.1) -> dict:
    if epochs < 1 or grad_accum < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("epochs/grad_accum/patience must be positive; learning_rate > 0")
    set_seed(seed)
    training = load_jsonl(train_path, require_label=True)
    validation = load_jsonl(val_path, require_label=True)
    cfg = cfg or ModelConfig()
    if (not cfg.use_clauses) and any("clause_labels" in x or "contradiction_labels" in x
                                     for x in training + validation):
        raise ValueError("--no-clauses requires labels aligned to whole sentences; remove clause labels")
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_name)
    model = ClauseFactModel(cfg).to(resolve_device(device))
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    best_f1, best_loss, bad_epochs = -1., float("inf"), 0
    history: list[dict] = []
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs+1):
        model.train()
        random.shuffle(training)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for i, item in enumerate(training):
            pred = model(item["source"], item["summary"], tokenizer)
            losses = compute_loss(pred, item, lambda_doc=lambda_doc,
                                  lambda_clause=lambda_clause, lambda_contra=lambda_contra,
                                  lambda_cal=lambda_cal)
            # Normalize the last, possibly shorter, accumulation group correctly.
            group_start = (i // grad_accum) * grad_accum
            group_size = min(grad_accum, len(training) - group_start)
            losses["total"].div(group_size).backward()
            total_loss += float(losses["total"].detach())
            if (i + 1) % grad_accum == 0 or i + 1 == len(training):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        val_metrics, _ = predict_items(model, tokenizer, validation)
        f1 = val_metrics["f1"]
        avg_loss = total_loss / len(training)
        record = {"epoch": epoch, "train_loss": avg_loss, "val": val_metrics}
        history.append(record)
        print(f"Epoch {epoch}/{epochs} | train_loss={avg_loss:.4f} | val_f1={f1:.4f} | val_ece={val_metrics['ece']:.4f}", flush=True)
        if f1 > best_f1 + 1e-9 or (abs(f1 - best_f1) < 1e-9 and avg_loss < best_loss):
            best_f1, best_loss, bad_epochs = f1, avg_loss, 0
            save_checkpoint(out, model, tokenizer, training_meta={
                "best_epoch": epoch, "best_validation_f1": best_f1,
                "learning_rate": learning_rate, "seed": seed, "lambda_weights": {
                    "document": lambda_doc, "clause": lambda_clause,
                    "contradiction": lambda_contra, "calibration": lambda_cal,
                },
            })
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break
        write_json(out / "training_history.json", history)
    write_json(out / "training_history.json", history)
    return {"checkpoint": str(out), "best_validation_f1": best_f1, "epochs_ran": len(history)}
