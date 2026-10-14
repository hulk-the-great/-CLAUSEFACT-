"""Model graph/gradient smoke test without network access or HF downloads.

Uses real PyTorch if available and substitutes a tiny fake AutoModel/tokenizer.
"""
import sys
import types
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")


class DummyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(128, 12)
        self.config = types.SimpleNamespace(hidden_size=12)

    def forward(self, input_ids, attention_mask):
        return types.SimpleNamespace(last_hidden_state=self.emb(input_ids))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "dummy_config.json").write_text("{}", encoding="utf-8")


class DummyTokenizer:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "dummy_tokenizer.json").write_text("{}", encoding="utf-8")

    def __call__(self, texts, **kwargs):
        texts = [str(s) for s in texts]
        encoded = [[(ord(ch) % 127) + 1 for ch in s[:32]] for s in texts]
        maxlen = max(map(len, encoded))
        return {"input_ids": torch.tensor([row + [0] * (maxlen - len(row)) for row in encoded]),
                "attention_mask": torch.tensor([[1] * len(row) + [0] * (maxlen - len(row)) for row in encoded])}


def test_model_forward_backward(monkeypatch):
    stub = types.ModuleType("transformers")
    stub.AutoModel = types.SimpleNamespace(from_pretrained=lambda name: DummyTransformer())
    monkeypatch.setitem(sys.modules, "transformers", stub)
    sys.modules.pop("clausefact.model", None)
    from clausefact.model import ClauseFactModel, ModelConfig, compute_loss, serialize_prediction
    model = ClauseFactModel(ModelConfig())
    item = {"source": "Alice won 3 prizes. Bob received 2 prizes.",
            "summary": "Alice won 5 prizes.", "label": 0,
            "clause_labels": [0], "contradiction_labels": [1]}
    pred = model(item["source"], item["summary"], DummyTokenizer())
    assert 0 <= pred["document_probability"].item() <= 1
    assert len(pred["clauses"]) == 1
    losses = compute_loss(pred, item)
    assert losses["total"].item() >= 0
    losses["total"].backward()
    assert model.support_head.weight.grad is not None
    report = serialize_prediction(pred)
    assert len(report["clauses"]) == 1
    assert report["num_evidence_spans"] >= 2


def test_checkpoint_and_training_loop(monkeypatch, tmp_path):
    stub = types.ModuleType("transformers")
    stub.AutoModel = types.SimpleNamespace(from_pretrained=lambda name: DummyTransformer())
    stub.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda name: DummyTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", stub)
    sys.modules.pop("clausefact.model", None)
    sys.modules.pop("clausefact.engine", None)
    from clausefact.model import ModelConfig
    from clausefact.engine import train, load_checkpoint, predict_items
    import json
    records = [
        {"id": "ok", "source": "Alice won an award in 2020.", "summary": "Alice won an award in 2020.",
         "label": 1, "clause_labels": [1], "contradiction_labels": [0]},
        {"id": "bad", "source": "Bob won an award in 2021.", "summary": "Bob won an award in 2023.",
         "label": 0, "clause_labels": [0], "contradiction_labels": [1]},
    ]
    for key in ("train", "val"):
        (tmp_path / f"{key}.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in records), encoding="utf-8")
    root = tmp_path / "checkpoint"
    result = train(str(tmp_path / "train.jsonl"), str(tmp_path / "val.jsonl"),
                   str(root), cfg=ModelConfig(backbone_name="dummy"),
                   epochs=1, device="cpu")
    assert result["epochs_ran"] == 1
    assert (root / "model.pt").is_file()
    model, tokenizer = load_checkpoint(root, device="cpu")
    metrics, pred = predict_items(model, tokenizer, records)
    assert metrics["n"] == 2
    assert len(pred) == 2
