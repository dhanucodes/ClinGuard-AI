
import pandas as pd
import pytest

from clinguard.explainability.explanations import (
    explain_anomaly_records,
)


def prepare_data():

    data = {
        "patient_id": ["P001"],
        "timestamp": ["2026-01-01 10:00:00"],
        "hybrid_or_flag": [True],
        "hybrid_and_flag": [False],
        "statistical_anomaly_flag": [True],
        "isolation_forest_anomaly_flag": [False],
        "verification_status": ["supported_anomaly_candidate"],
        "verification_reason": ["No recorded context issues."],
        "context_issues": ["No recorded issues"],
    }

    vitals = [
        "heart_rate",
        "spo2",
        "systolic_bp",
        "diastolic_bp",
        "respiratory_rate",
        "temperature",
    ]

    for vital in vitals:
        data[vital] = [80.0]
        data[f"{vital}_baseline"] = [75.0]
        data[f"{vital}_deviation"] = [5.0]
        data[f"{vital}_z_score"] = [1.0]

    data["heart_rate"] = [110.0]
    data["heart_rate_deviation"] = [30.0]
    data["heart_rate_z_score"] = [3.5]

    return pd.DataFrame(data)


def test_explanation_columns_created():
    result = explain_anomaly_records(prepare_data())

    assert "explainable_reason" in result.columns
    assert "explanation_summary" in result.columns


def test_deviating_vital_identified():
    result = explain_anomaly_records(prepare_data())

    assert "heart_rate" in result.loc[
        0, "deviating_vital_evidence"
    ]


def test_detector_evidence_included():
    result = explain_anomaly_records(prepare_data())

    assert "Statistical Z-score detector" in result.loc[
        0, "explainable_reason"
    ]


def test_verification_status_included():
    result = explain_anomaly_records(prepare_data())

    assert "supported_anomaly_candidate" in result.loc[
        0, "explainable_reason"
    ]


def test_original_data_unchanged():
    data = prepare_data()
    original = data.copy(deep=True)

    explain_anomaly_records(data)

    pd.testing.assert_frame_equal(data, original)


def test_empty_dataset_rejected():
    with pytest.raises(ValueError):
        explain_anomaly_records(pd.DataFrame())


def test_missing_columns_rejected():
    with pytest.raises(ValueError):
        explain_anomaly_records(
            pd.DataFrame({"patient_id": ["P001"]})
        )
