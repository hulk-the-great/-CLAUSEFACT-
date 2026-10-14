from clausefact.preprocessing import (
    build_evidence_spans, decompose_clauses, entity_mentions, fact_signals,
    numeric_mentions, split_sentences,
)


def test_split_and_clauses():
    sentences = split_sentences("Maya went to Delhi. She won a prize.")
    assert len(sentences) == 2
    clauses = decompose_clauses("Maya won a prize and Rina lost the game.")
    assert len(clauses) == 2
    assert clauses[0].startswith("Maya")
    assert clauses[1].startswith("Rina")


def test_fact_prior_and_negation():
    signals = fact_signals("Alice has not received 15 awards.")
    assert signals.entity and signals.number and signals.relation and signals.negation
    assert signals.prior() > 1
    assert "15" in numeric_mentions("She bought 15 items.")
    assert "alice" in entity_mentions("Alice won.")


def test_overlapping_evidence_windows():
    spans = build_evidence_spans("A wrote the report. B reviewed it. C approved it.")
    assert len(spans) == 5
    assert "A wrote the report. B reviewed it." in spans
    assert len(build_evidence_spans("Hello.", max_spans=1)) == 1
