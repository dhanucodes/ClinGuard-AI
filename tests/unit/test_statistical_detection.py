
import pandas as pd
import pytest

from clinguard.detection.statistical import (
    detect_statistical_anomalies,
)

from clinguard.data_generation.generator import VITAL_COLUMNS


def create_sample_data():

    records = []

    for index in range(4):

        record = {"patient_id": f"P00{index + 1}"}

        for vital in VITAL_COLUMNS:
            record[f"{vital}_z_score"] = 0.5

        records.append(record)

    return pd.DataFrame(records)


def test_high_z_score_is_detected():

    data = create_sample_data()
    data.loc[0, "heart_rate_z_score"] = 3.5

    result = detect_statistical_anomalies(data)

    assert result.loc[0, "statistical_anomaly_flag"]


def test_negative_z_score_is_detected():

    data = create_sample_data()
    data.loc[0, "spo2_z_score"] = -3.5

    result = detect_statistical_anomalies(data)

    assert result.loc[0, "statistical_anomaly_flag"]


def test_exact_threshold_is_detected():

    data = create_sample_data()
    data.loc[0, "heart_rate_z_score"] = 3.0

    result = detect_statistical_anomalies(data)

    assert result.loc[0, "statistical_anomaly_flag"]


def test_normal_z_score_is_not_detected():

    data = create_sample_data()

    result = detect_statistical_anomalies(data)

    assert not result["statistical_anomaly_flag"].any()


def test_missing_z_score_is_not_detected():

    data = create_sample_data()
    data.loc[0, "heart_rate_z_score"] = float("nan")

    result = detect_statistical_anomalies(data)

    assert not result.loc[0, "statistical_anomaly_flag"]


def test_detected_vital_is_recorded():

    data = create_sample_data()
    data.loc[0, "heart_rate_z_score"] = 4.0
    data.loc[0, "spo2_z_score"] = -3.5

    result = detect_statistical_anomalies(data)

    assert result.loc[0, "statistical_anomaly_vitals"] == (
        "heart_rate, spo2"
    )


def test_original_dataset_is_not_modified():

    data = create_sample_data()
    original_columns = list(data.columns)

    detect_statistical_anomalies(data)

    assert list(data.columns) == original_columns


def test_invalid_threshold_is_rejected():

    data = create_sample_data()

    with pytest.raises(ValueError):
        detect_statistical_anomalies(data, z_threshold=0)


def test_missing_columns_are_rejected():

    data = pd.DataFrame({
        "patient_id": ["P001"],
    })

    with pytest.raises(ValueError):
        detect_statistical_anomalies(data)
