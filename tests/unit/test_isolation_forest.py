
import pandas as pd
import pytest

from clinguard.detection.isolation_forest import (
    detect_isolation_forest_anomalies,
)

VITALS = [
    "heart_rate",
    "spo2",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "temperature",
]


def prepare_data():
    rows = []

    for i in range(30):
        row = {
            f"{vital}_z_score": 0.1 + (i % 5) * 0.1
            for vital in VITALS
        }
        row["anomaly_label"] = 0
        rows.append(row)

    unusual = {
        f"{vital}_z_score": 10.0
        for vital in VITALS
    }
    unusual["anomaly_label"] = 1
    rows.append(unusual)

    return pd.DataFrame(rows)


def test_prediction_column_exists():
    result = detect_isolation_forest_anomalies(prepare_data())
    assert "isolation_forest_prediction" in result.columns


def test_predictions_are_valid():
    result = detect_isolation_forest_anomalies(prepare_data())
    predictions = set(
        result["isolation_forest_prediction"].dropna().unique()
    )
    assert predictions.issubset({-1, 1})


def test_anomaly_flag_is_boolean():
    result = detect_isolation_forest_anomalies(prepare_data())
    assert result["isolation_forest_anomaly_flag"].dtype == bool


def test_unusual_record_is_detected():
    result = detect_isolation_forest_anomalies(prepare_data())
    assert result.iloc[-1]["isolation_forest_anomaly_flag"]


def test_missing_feature_row_is_excluded():
    data = prepare_data()
    data.loc[0, "heart_rate_z_score"] = float("nan")

    result = detect_isolation_forest_anomalies(data)

    assert not result.iloc[0]["isolation_forest_input_valid"]
    assert pd.isna(result.iloc[0]["isolation_forest_prediction"])


def test_original_data_is_unchanged():
    data = prepare_data()
    original = data.copy(deep=True)

    detect_isolation_forest_anomalies(data)

    pd.testing.assert_frame_equal(data, original)


def test_empty_dataset_is_rejected():
    data = pd.DataFrame(columns=[
        f"{vital}_z_score" for vital in VITALS
    ])

    with pytest.raises(ValueError):
        detect_isolation_forest_anomalies(data)


def test_invalid_contamination_is_rejected():
    data = prepare_data()

    with pytest.raises(ValueError):
        detect_isolation_forest_anomalies(
            data, contamination=0
        )


def test_missing_feature_columns_are_rejected():
    data = pd.DataFrame({"heart_rate_z_score": [1, 2]})

    with pytest.raises(ValueError):
        detect_isolation_forest_anomalies(data)


def test_labels_do_not_affect_model_predictions():
    data = prepare_data()
    changed = data.copy()
    changed["anomaly_label"] = 1 - changed["anomaly_label"]

    first = detect_isolation_forest_anomalies(data)
    second = detect_isolation_forest_anomalies(changed)

    pd.testing.assert_series_equal(
        first["isolation_forest_prediction"],
        second["isolation_forest_prediction"],
    )
