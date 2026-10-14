import json
from pathlib import Path
import pytest
from clausefact.cli import main
from clausefact.data import load_jsonl, validate_example


def test_example_data_is_valid():
    base = Path(__file__).resolve().parents[1] / "data" / "example"
    for name, expected in (("train", 12), ("val", 6), ("test", 6)):
        data = load_jsonl(base / f"{name}.jsonl", require_label=True)
        assert len(data) == expected


def test_reject_clause_label_length_mismatch():
    with pytest.raises(ValueError, match="clause_labels"):
        validate_example({"source": "Alice won.", "summary": "Alice won.",
                          "label": 1, "clause_labels": [1, 0]})


def test_offline_inspect(tmp_path, capsys):
    path = tmp_path / "preview.json"
    main(["inspect", "--source-text", "Alice arrived at noon. Bob left at dusk.",
          "--summary-text", "Alice arrived at noon.", "--out", str(path)])
    report = json.loads(path.read_text(encoding="utf-8"))
    assert len(report["clauses"]) == 1
    assert report["graph_edge_count"] >= 1
    assert "No factuality prediction" in report["note"]
    assert "clauses" in capsys.readouterr().out
