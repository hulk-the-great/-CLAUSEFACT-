import pytest
from clausefact.metrics import auroc, ece, f1_score, classification_metrics


def test_perfect_metrics():
    truth = [0, 1, 0, 1]
    score = [.01, .99, .02, .98]
    metrics = classification_metrics(truth, score)
    assert metrics["f1"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["ece"] == pytest.approx(.015)


def test_auc_with_tied_scores_and_single_class():
    assert auroc([0, 1], [.5, .5]) == .5
    assert auroc([1, 1], [.2, .8]) is None
    assert f1_score([0, 0], [.1, .2]) == 0
    with pytest.raises(ValueError):
        ece([1], [1.5])
