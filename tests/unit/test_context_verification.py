
import pandas as pd
import pytest

from clinguard.verification.context_verification import (
    verify_anomaly_context,
)


def prepare_data():
    return pd.DataFrame({
        "hybrid_or_flag": [True, True, False, False],
        "hybrid_and_flag": [True, False, False, False],
        "baseline_ready": [True, True, True, False],

        "has_missing_value": [False, False, False, False],
        "has_noise": [False, True, False, False],
        "has_sensor_error": [False, False, False, False],
        "data_quality_issue": [False, True, False, False],

        "invalid_timestamp": [False, False, False, False],
        "timestamp_interval_mismatch": [False, False, True, False],
        "has_implausible_value": [False, False, False, False],
    })


def test_clean_candidate_is_supported():
    result = verify_anomaly_context(prepare_data())

    assert (
        result.loc[0, "verification_status"]
        == "supported_anomaly_candidate"
    )


def test_candidate_with_noise_needs_review():
    result = verify_anomaly_context(prepare_data())

    assert (
        result.loc[1, "verification_status"]
        == "candidate_needs_review"
    )


def test_quality_issue_without_candidate():
    result = verify_anomaly_context(prepare_data())

    assert (
        result.loc[2, "verification_status"]
        == "quality_review_without_model_alert"
    )


def test_unready_baseline_needs_review():
    result = verify_anomaly_context(prepare_data())

    assert (
        result.loc[3, "verification_status"]
        == "quality_review_without_model_alert"
    )


def test_and_strategy():
    result = verify_anomaly_context(
        prepare_data(),
        strategy="and",
    )

    assert (
        result.loc[1, "verification_status"]
        == "quality_review_without_model_alert"
    )


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError):
        verify_anomaly_context(
            prepare_data(),
            strategy="unknown",
        )


def test_empty_dataset_rejected():
    with pytest.raises(ValueError):
        verify_anomaly_context(pd.DataFrame())


def test_original_data_unchanged():
    data = prepare_data()
    original = data.copy(deep=True)

    verify_anomaly_context(data)

    pd.testing.assert_frame_equal(data, original)


def test_missing_columns_rejected():
    with pytest.raises(ValueError):
        verify_anomaly_context(
            pd.DataFrame({"hybrid_or_flag": [True]})
        )
