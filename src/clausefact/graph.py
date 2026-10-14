"""Undirected evidence graph from discourse and lexical continuity."""
from __future__ import annotations

from .preprocessing import entity_mentions, numeric_mentions, salient_tokens


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def build_adjacency(spans: list[str], *, overlap_threshold: float = 0.27,
                    use_graph: bool = True) -> list[list[bool]]:
    """Self-loop and optional edges for neighbors/entities/tokens/numbers.

    A boolean list is returned instead of a dense tensor so preprocessing does
    not require torch. The model converts the list into a tensor on its device.
    """
    n = len(spans)
    words = [salient_tokens(s) for s in spans]
    names = [entity_mentions(s) for s in spans]
    nums = [numeric_mentions(s) for s in spans]
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        adj[i][i] = True
        if not use_graph:
            continue
        for j in range(i + 1, n):
            linked = (j == i + 1 or bool(names[i] & names[j])
                      or bool(nums[i] & nums[j])
                      or jaccard(words[i], words[j]) >= overlap_threshold)
            adj[i][j] = linked
            adj[j][i] = linked
    return adj


def edge_count(adj: list[list[bool]]) -> int:
    return sum(1 for i in range(len(adj)) for j in range(i + 1, len(adj)) if adj[i][j])
