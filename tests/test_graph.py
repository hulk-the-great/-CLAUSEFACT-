from clausefact.graph import build_adjacency, edge_count, jaccard


def test_symmetric_graph():
    graph = build_adjacency([
        "Alice went to Berlin.", "Bob visited Rome.",
        "Alice met a teacher.", "Three ships arrived.",
    ])
    assert all(graph[i][i] for i in range(4))
    assert graph == [list(r) for r in zip(*graph)]
    assert graph[0][2]  # shared entity Alice
    assert edge_count(graph) >= 3  # three adjacency edges


def test_no_graph_ablation():
    graph = build_adjacency(["Alice went home.", "Alice won."], use_graph=False)
    assert graph == [[True, False], [False, True]]
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
