"""Rule-assisted clause decomposition and evidence-span preparation.

The paper does not specify an exact splitter or entity tagger. These simple,
deterministic heuristics are replaceable baselines, not a trained neural parser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z0-9])")
_TOKEN = re.compile(r"[\w]+(?:['’-][\w]+)*", re.UNICODE)
# Conjunction boundaries are conservative: a semicolon is always a boundary,
# but 'and' is split only if a new plausible subject begins a following clause.
_SUBJECT_AFTER_CONJ = re.compile(
    r"\s+(?:and|but|whereas|although|while)\s+(?=(?:[A-Z][a-z]+|he|she|they|it|we|the|a|an|this|that|these|those)\s+)",
    flags=re.IGNORECASE,
)
_NUM = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*%?(?!\w)")
_NEG = re.compile(r"\b(?:not|never|no|none|without|n't)\b", re.I)
_RELATIONS = {
    "is", "are", "was", "were", "has", "have", "had", "said", "reported",
    "won", "lost", "bought", "sold", "increased", "decreased", "fell", "rose",
    "founded", "joined", "caused", "located", "appointed", "announced",
    "produced", "earned", "killed", "visited", "signed", "reached", "received",
}
_STOP = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "of",
    "for", "at", "in", "on", "to", "with", "by", "it", "he", "she", "they",
    "we", "that", "this", "from", "as", "its", "had", "has", "have",
}


@dataclass(frozen=True)
class FactSignals:
    entity: bool
    number: bool
    relation: bool
    negation: bool

    def prior(self, alpha_entity: float = 0.5, alpha_number: float = 0.5,
              alpha_relation: float = 0.25) -> float:
        return (1.0 + alpha_entity * self.entity
                + alpha_number * self.number + alpha_relation * self.relation)


def tokenize_words(text: str) -> list[str]:
    return [m.group().lower() for m in _TOKEN.finditer(text)]


def salient_tokens(text: str) -> set[str]:
    return {x for x in tokenize_words(text) if x not in _STOP and len(x) >= 2}


def entity_mentions(text: str) -> set[str]:
    """Approximate proper-name spans; for research use replace with NER/coref."""
    # Preserve contiguous capitalized sequences like "New York".
    return {m.group().lower() for m in re.finditer(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text
    ) if m.group().lower() not in {"the", "a", "an", "this", "that"}}


def numeric_mentions(text: str) -> set[str]:
    return set(_NUM.findall(text))


def fact_signals(clause: str) -> FactSignals:
    return FactSignals(
        entity=bool(entity_mentions(clause)),
        number=bool(numeric_mentions(clause)),
        relation=bool(set(tokenize_words(clause)) & _RELATIONS),
        negation=bool(_NEG.search(clause)),
    )


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_BREAK.split(cleaned) if part.strip()]


def decompose_clauses(summary: str) -> list[str]:
    """Split sentence-local conjunctions and punctuation into factual claims.

    Limitations: pronoun/coreference resolution, implicit shared subjects, and
    syntactically complicated relatives are not reconstructed here.
    """
    clauses: list[str] = []
    for sentence in split_sentences(summary):
        for semicolon_part in re.split(r"\s*;\s*", sentence):
            for part in _SUBJECT_AFTER_CONJ.split(semicolon_part):
                part = part.strip(" ,;\t\n")
                if part:
                    clauses.append(part)
    return clauses


def build_evidence_spans(source: str, *, window_size: int = 2,
                         max_spans: int = 80) -> list[str]:
    """Single sentences + overlapping windows; deduplicated in source order."""
    if window_size < 1 or max_spans < 1:
        raise ValueError("window_size and max_spans must be positive")
    sentences = split_sentences(source)
    seen: set[str] = set()
    spans: list[str] = []
    for i in range(len(sentences)):
        for width in sorted({1, window_size}):
            if i + width > len(sentences):
                continue
            span = " ".join(sentences[i:i+width])
            if span not in seen:
                spans.append(span)
                seen.add(span)
            if len(spans) >= max_spans:
                return spans
    return spans
    return spans
