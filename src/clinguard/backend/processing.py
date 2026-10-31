"""AI processing service for ClinGuard AI."""

import pandas as pd

from clinguard.adaptive_baseline.baseline import (
    calculate_adaptive_baselines,
)
from clinguard.adaptive_baseline.deviation import (
    calculate_deviations,
)
from clinguard.detection.statistical import (
    detect_statistical_anomalies,
)
from clinguard.detection.isolation_forest import (
    detect_isolation_forest_anomalies,
)
from clinguard.detection.hybrid import (
    detect_hybrid_anomalies,
)
from clinguard.preprocessing.validation import (
    validate_dataset,
)
from clinguard.agents.workflow import (
    run_clinguard_workflow,
)


def process_patient_history(dataset):
    """Run the AI workflow on one patient's time-series history."""

    if dataset.empty:
        raise ValueError("Patient history is empty.")

    result = dataset.copy()

    required_columns = [
        "patient_id",
        "timestamp",
        "heart_rate",
        "spo2",
        "systolic_bp",
        "diastolic_bp",
        "respiratory_rate",
        "temperature",
        "has_missing_value",
        "has_noise",
        "has_sensor_error",
        "data_quality_issue",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # This service must receive readings for only one patient.
    if result["patient_id"].nunique() != 1:
        raise ValueError(
            "Process one patient's history at a time."
        )

    # Step 1: Validate timestamps and vital-sign values.
    result, validation_report = validate_dataset(
        result,
        expected_interval_minutes=5,
    )

    if result["invalid_timestamp"].any():
        raise ValueError(
            "Patient history contains invalid timestamps."
        )

    # Step 2: Calculate patient-specific adaptive baselines.
    result = calculate_adaptive_baselines(
        result,
        window_size=12,
        min_periods=6,
    )

    # Step 3: Calculate deviations and Z-scores.
    result = calculate_deviations(
        result,
        window_size=12,
        min_periods=6,
    )

    # Step 4: Run statistical anomaly detection.
    result = detect_statistical_anomalies(
        result,
        z_threshold=3.0,
    )

    # Step 5: Run Isolation Forest when enough valid
    # Z-score records are available.
    z_score_columns = [
        "heart_rate_z_score",
        "spo2_z_score",
        "systolic_bp_z_score",
        "diastolic_bp_z_score",
        "respiratory_rate_z_score",
        "temperature_z_score",
    ]

    valid_z_scores = (
        result[z_score_columns]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .all(axis=1)
    )

    if valid_z_scores.sum() >= 2:
        result = detect_isolation_forest_anomalies(
            result,
            contamination=0.05,
            random_state=42,
        )

    else:
        # Do not invent an Isolation Forest result when
        # the patient history is too short.
        result["isolation_forest_prediction"] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64",
        )
        result["isolation_forest_anomaly_flag"] = False
        result["isolation_forest_score"] = float("nan")
        result["isolation_forest_input_valid"] = False

    # Step 6: Combine the detector results.
    result = detect_hybrid_anomalies(result)

    # Step 7: Run context verification and explainability
    # through the existing LangGraph workflow.
    workflow_result = run_clinguard_workflow(
        result,
        strategy="or",
    )

    final_dataset = workflow_result["explained_dataset"]

    # Step 8: Return results and processing information.
    return {
        "dataset": final_dataset,
        "summary": workflow_result["summary"],
        "validation_report": validation_report,
        "status": (
            "completed"
            if final_dataset["baseline_ready"].any()
            else "insufficient_history"
        ),
    }