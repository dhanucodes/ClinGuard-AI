
import pytest

from clinguard.evaluation.metrics import evaluate_detector


def test_perfect_predictions():
    result = evaluate_detector(
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    )

    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0


def test_confusion_matrix_values():
    result = evaluate_detector(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
    )

    assert result["true_negative"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["true_positive"] == 1


def test_no_anomalies_detected():
    result = evaluate_detector(
        [0, 0, 1, 1],
        [0, 0, 0, 0],
    )

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0


def test_empty_data_rejected():
    with pytest.raises(ValueError):
        evaluate_detector([], [])


def test_different_lengths_rejected():
    with pytest.raises(ValueError):
        evaluate_detector([0, 1], [0])


def test_invalid_labels_rejected():
    with pytest.raises(ValueError):
        evaluate_detector([0, 2], [0, 1])


def test_missing_labels_rejected():
    with pytest.raises(ValueError):
        evaluate_detector([0, None], [0, 1])
