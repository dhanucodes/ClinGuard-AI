"""Integration tests for the ClinGuard AI LangGraph workflow."""

import pandas as pd

from clinguard.agents.workflow import run_clinguard_workflow


def create_test_dataset():

    data = {
        "patient_id": ["P001"],
        "timestamp": ["2026-01-01 10:00:00"],

        "hybrid_or_flag": [False],
        "hybrid_and_flag": [False],

        "statistical_anomaly_flag": [False],
        "isolation_forest_anomaly_flag": [False],

        "baseline_ready": [True],

        "has_missing_value": [False],
        "has_noise": [False],
        "has_sensor_error": [False],
        "data_quality_issue": [False],

        "invalid_timestamp": [False],
        "timestamp_interval_mismatch": [False],
        "has_implausible_value": [False],

        "verification_status": [""],
        "verification_reason": [""],
        "context_issues": [""],
    }

    vital_values = {
        "heart_rate": 75,
        "spo2": 98,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "respiratory_rate": 16,
        "temperature": 36.8,
    }

    for vital, value in vital_values.items():
        data[vital] = [value]
        data[f"{vital}_baseline"] = [value - 1]
        data[f"{vital}_deviation"] = [1]
        data[f"{vital}_z_score"] = [0.5]

    return pd.DataFrame(data)


def test_workflow_completes_successfully():

    dataset = create_test_dataset()

    result = run_clinguard_workflow(dataset)

    assert "explained_dataset" in result
    assert "summary" in result

    assert len(result["explained_dataset"]) == 1


def test_workflow_generates_summary():

    dataset = create_test_dataset()

    result = run_clinguard_workflow(dataset)

    summary = result["summary"]

    assert summary["total_records"] == 1
    assert summary["no_model_alert"] == 1


def test_workflow_creates_explanation_columns():

    dataset = create_test_dataset()

    result = run_clinguard_workflow(dataset)

    explained = result["explained_dataset"]

    assert "explainable_reason" in explained.columns
    assert "explanation_summary" in explained.columns
    assert "deviating_vital_evidence" in explained.columns