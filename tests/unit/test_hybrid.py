
import pandas as pd
import pytest

from clinguard.detection.hybrid import detect_hybrid_anomalies


def prepare_data():
    return pd.DataFrame({
        "statistical_anomaly_flag": [False, True, False, True],
        "isolation_forest_anomaly_flag": [False, False, True, True],
        "isolation_forest_input_valid": [True, True, True, True],
    })


def test_or_strategy():
    result = detect_hybrid_anomalies(prepare_data())

    assert result["hybrid_or_flag"].tolist() == [
        False, True, True, True
    ]


def test_and_strategy():
    result = detect_hybrid_anomalies(prepare_data())

    assert result["hybrid_and_flag"].tolist() == [
        False, False, False, True
    ]


def test_invalid_isolation_input():
    data = prepare_data()
    data.loc[1, "isolation_forest_input_valid"] = False

    result = detect_hybrid_anomalies(data)

    assert result.loc[1, "hybrid_and_flag"] == False
    assert result.loc[1, "hybrid_or_flag"] == True


def test_detection_reason():
    result = detect_hybrid_anomalies(prepare_data())

    assert result.loc[0, "hybrid_detection_reason"] == "No detector flagged"
    assert result.loc[1, "hybrid_detection_reason"] == "Statistical"
    assert result.loc[2, "hybrid_detection_reason"] == "Isolation Forest"
    assert result.loc[3, "hybrid_detection_reason"] == "Statistical + Isolation Forest"


def test_original_data_unchanged():
    data = prepare_data()
    original = data.copy(deep=True)

    detect_hybrid_anomalies(data)

    pd.testing.assert_frame_equal(data, original)


def test_empty_dataset_rejected():
    with pytest.raises(ValueError):
        detect_hybrid_anomalies(pd.DataFrame())


def test_missing_columns_rejected():
    with pytest.raises(ValueError):
        detect_hybrid_anomalies(
            pd.DataFrame({"statistical_anomaly_flag": [True]})
        )
