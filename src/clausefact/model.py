"""Trainable CLAUSEFACT model matching the main operations in Section III.

This is a reference implementation: the manuscript omits backbone identifiers,
exact graph edge thresholds, propagation scoring, data labels and several other
hyperparameters. Those choices are explicitly documented in docs/METHODOLOGY.md.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .graph import build_adjacency, edge_count
from .preprocessing import build_evidence_spans, decompose_clauses, fact_signals, split_sentences

try:
    from transformers import AutoModel
except ImportError as exc:
    raise ImportError("Install ML dependencies: pip install -e '.[ml]'") from exc


@dataclass
class ModelConfig:
    backbone_name: str = "distilbert-base-uncased"
    max_tokens: int = 192
    max_spans: int = 80
    window_size: int = 2
    dropout: float = 0.1
    alpha_entity: float = 0.5
    alpha_number: float = 0.5
    alpha_relation: float = 0.25
    graph_overlap_threshold: float = 0.27
    use_clauses: bool = True
    use_graph: bool = True
    use_contradiction: bool = True
    use_calibration: bool = True
    use_importance: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClauseFactModel(nn.Module):
    def __init__(self, cfg: ModelConfig, *, backbone_path: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(backbone_path or cfg.backbone_name)
        dim = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(cfg.dropout)
        self.affinity = nn.Parameter(torch.eye(dim) / math.sqrt(dim))
        self.support_head = nn.Linear(dim * 3, 1)
        self.contra_head = nn.Linear(dim * 3, 1)
        # gamma coefficients are learned; the manuscript does not set them.
        self.gamma_support = nn.Parameter(torch.tensor(3.0))
        self.gamma_contra = nn.Parameter(torch.tensor(4.0))
        self.gamma_importance = nn.Parameter(torch.tensor(0.25))

    def _encode(self, texts: list[str], tokenizer: Any) -> torch.Tensor:
        device = next(self.parameters()).device
        batch = tokenizer(texts, padding=True, truncation=True,
                          max_length=self.cfg.max_tokens, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        encoded = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.dropout(pooled)

    def forward(self, source: str, summary: str, tokenizer: Any) -> dict[str, Any]:
        clauses = (decompose_clauses(summary) if self.cfg.use_clauses
                   else split_sentences(summary))
        spans = build_evidence_spans(source, window_size=self.cfg.window_size,
                                     max_spans=self.cfg.max_spans)
        if not clauses or not spans:
            raise ValueError("Both source and summary must contain text")
        n, m = len(clauses), len(spans)
        encoded = self._encode(clauses + spans, tokenizer)
        h, t = encoded[:n], encoded[n:]

        # Initial clause-to-evidence attention: a_kj = softmax(h_k^T W_a t_j).
        attention_logits = (h @ self.affinity) @ t.T
        attention = torch.softmax(attention_logits, dim=1)

        # One-hop graph aggregation with trainable representations and
        # attention-normalized neighbor weights beta_jr.
        adj = build_adjacency(spans, overlap_threshold=self.cfg.graph_overlap_threshold,
                              use_graph=self.cfg.use_graph)
        mask = torch.tensor(adj, dtype=torch.bool, device=t.device)
        neighbor_logits = (t @ t.T) / math.sqrt(t.size(-1))
        beta = torch.softmax(neighbor_logits.masked_fill(~mask, -1e9), dim=1)
        refined = beta @ t

        pair_h = h[:, None, :].expand(n, m, -1)
        pair_t = refined[None, :, :].expand(n, m, -1)
        s = torch.sigmoid(self.support_head(torch.cat(
            [pair_h, pair_t, torch.abs(pair_h - pair_t)], dim=-1
        )).squeeze(-1))
        r = torch.sigmoid(self.contra_head(torch.cat(
            [pair_h, pair_t, pair_h * pair_t], dim=-1
        )).squeeze(-1))
        max_support, best_support = s.max(dim=1)
        max_contra, best_contra = r.max(dim=1)

        q = torch.tensor([
            fact_signals(c).prior(self.cfg.alpha_entity, self.cfg.alpha_number,
                                  self.cfg.alpha_relation)
            for c in clauses
        ], device=h.device, dtype=h.dtype)
        if not self.cfg.use_importance:
            q = torch.ones_like(q)

        logit = self.gamma_support * max_support
        if self.cfg.use_contradiction:
            logit = logit - self.gamma_contra * max_contra
        logit = logit + self.gamma_importance * q
        clause_prob = torch.sigmoid(logit)

        if m == 1:
            normalized_entropy = torch.zeros(n, dtype=h.dtype, device=h.device)
        else:
            entropy = -(attention * attention.clamp_min(1e-9).log()).sum(dim=1)
            normalized_entropy = (entropy / math.log(m)).clamp(0, 1)
        if self.cfg.use_calibration:
            weight = q * (1 - normalized_entropy).clamp_min(1e-4)
        else:
            weight = q
        doc_prob = (weight * clause_prob).sum() / weight.sum().clamp_min(1e-8)

        return {
            "document_probability": doc_prob,
            "clause_probabilities": clause_prob,
            "support_probabilities": max_support,
            "contradiction_probabilities": max_contra,
            "importance": q,
            "normalized_uncertainty": normalized_entropy,
            "clause_weights": weight / weight.sum().clamp_min(1e-8),
            "attention": attention,
            "best_support_indices": best_support,
            "best_contradiction_indices": best_contra,
            "clauses": clauses,
            "evidence_spans": spans,
            "graph_edges": edge_count(adj),
        }


def compute_loss(pred: dict[str, Any], item: dict[str, Any], *,
                 lambda_doc: float = 1., lambda_clause: float = 1.,
                 lambda_contra: float = 1., lambda_cal: float = 0.1) -> dict[str, torch.Tensor]:
    """L = lambda1 Ldoc + lambda2 Lclause + lambda3 Lcontra + lambda4 Lcal.

    Lcal is implemented as Brier squared error, a proper scoring rule;
    the paper specifies a calibration regularizer but not its exact form.
    """
    p = pred["document_probability"].clamp(1e-6, 1-1e-6)
    y = p.new_tensor(float(item["label"]))
    ldoc = F.binary_cross_entropy(p, y)
    lcal = (p - y).square()
    zero = ldoc.new_zeros(())
    lclause, lcontra = zero, zero
    if "clause_labels" in item:
        labels = p.new_tensor(item["clause_labels"])
        if len(labels) != len(pred["clauses"]):
            raise ValueError("clause_labels must align with preprocessed clauses")
        lclause = F.binary_cross_entropy(
            pred["support_probabilities"].clamp(1e-6, 1-1e-6), labels)
    if "contradiction_labels" in item:
        labels = p.new_tensor(item["contradiction_labels"])
        if len(labels) != len(pred["clauses"]):
            raise ValueError("contradiction_labels must align with preprocessed clauses")
        lcontra = F.binary_cross_entropy(
            pred["contradiction_probabilities"].clamp(1e-6, 1-1e-6), labels)
    total = lambda_doc * ldoc + lambda_clause * lclause + lambda_contra * lcontra + lambda_cal * lcal
    return {"total": total, "document": ldoc, "clause": lclause,
            "contradiction": lcontra, "calibration": lcal}


def serialize_prediction(pred: dict[str, Any], *, id: str | None = None) -> dict[str, Any]:
    """Make human-readable, JSON-safe clause evidence report."""
    def numbers(t: torch.Tensor) -> list[float]:
        return [float(x) for x in t.detach().cpu().tolist()]

    clause_p = numbers(pred["clause_probabilities"])
    supports = numbers(pred["support_probabilities"])
    contras = numbers(pred["contradiction_probabilities"])
    importance = numbers(pred["importance"])
    uncertainty = numbers(pred["normalized_uncertainty"])
    weights = numbers(pred["clause_weights"])
    attention = pred["attention"].detach().cpu().tolist()
    best_s = pred["best_support_indices"].detach().cpu().tolist()
    best_c = pred["best_contradiction_indices"].detach().cpu().tolist()
    clauses = []
    for i, text in enumerate(pred["clauses"]):
        clauses.append({
            "clause": text,
            "faithfulness_probability": clause_p[i],
            "support_score": supports[i],
            "contradiction_score": contras[i],
            "importance_prior": importance[i],
            "normalized_uncertainty": uncertainty[i],
            "normalized_weight": weights[i],
            "best_support_evidence": pred["evidence_spans"][int(best_s[i])],
            "best_contradicting_evidence": pred["evidence_spans"][int(best_c[i])],
            "highest_attention_evidence": pred["evidence_spans"][max(
                range(len(attention[i])), key=lambda j: attention[i][j]
            )],
        })
    result = {
        "document_faithfulness_probability": float(pred["document_probability"].detach().cpu()),
        "predicted_faithful": bool(pred["document_probability"].detach().cpu().item() >= 0.5),
        "num_evidence_spans": len(pred["evidence_spans"]),
        "graph_edge_count": pred["graph_edges"],
        "clauses": clauses,
    }
    if id is not None:
        result["id"] = id
    return result
