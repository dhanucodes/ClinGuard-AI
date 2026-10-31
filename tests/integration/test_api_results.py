
"""Integration tests for ClinGuard AI result retrieval endpoints."""

import uuid
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from clinguard.backend.main import app
from clinguard.database.session import SessionLocal, get_db
from clinguard.database.models import (
    Patient,
    VitalReading,
    DetectionResult,
    VerificationResult,
)


@pytest.fixture
def test_client_with_results():
    """Create temporary database records for API testing."""

    db = SessionLocal()
    patient_id = f"TEST-{uuid.uuid4().hex[:8]}"

    try:
        # 1. Create a temporary patient.
        patient = Patient(patient_id=patient_id)
        db.add(patient)
        db.flush()

        # 2. Create a temporary vital reading.
        reading = VitalReading(
            patient_id=patient_id,
            timestamp=datetime(2026, 1, 1, 10, 0),
            heart_rate=75,
            spo2=98,
            systolic_bp=120,
            diastolic_bp=80,
            respiratory_rate=16,
            temperature=36.8,
            has_missing_value=False,
            has_noise=False,
            has_sensor_error=False,
            data_quality_issue=False,
        )

        db.add(reading)
        db.flush()

        # 3. Create a detection result.
        detection = DetectionResult(
            reading_id=reading.reading_id,
            statistical_anomaly_flag=False,
            isolation_forest_anomaly_flag=False,
            hybrid_or_flag=False,
            hybrid_and_flag=False,
            isolation_forest_score=0.15,
        )

        db.add(detection)
        db.flush()

        # 4. Create a verification result.
        verification = VerificationResult(
            detection_id=detection.detection_id,
            verification_status="no_model_alert",
            verification_reason="No detector flagged this reading.",
            context_issues="No recorded issues.",
            explainable_reason="No anomaly evidence was identified.",
        )

        db.add(verification)
        db.commit()

        # 5. Make the API use this test database session.
        def override_get_db():
            yield db

        app.dependency_overrides[get_db] = override_get_db

        with TestClient(app) as client:
            yield client, patient_id

    finally:
        # 6. Remove only the temporary test records.
        app.dependency_overrides.clear()

        db.execute(
            delete(VerificationResult).where(
                VerificationResult.detection_id.in_(
                    db.query(DetectionResult.detection_id)
                    .join(
                        VitalReading,
                        DetectionResult.reading_id
                        == VitalReading.reading_id,
                    )
                    .where(VitalReading.patient_id == patient_id)
                )
            )
        )

        db.execute(
            delete(DetectionResult).where(
                DetectionResult.reading_id.in_(
                    db.query(VitalReading.reading_id).where(
                        VitalReading.patient_id == patient_id
                    )
                )
            )
        )

        db.execute(
            delete(VitalReading).where(
                VitalReading.patient_id == patient_id
            )
        )

        db.execute(
            delete(Patient).where(
                Patient.patient_id == patient_id
            )
        )

        db.commit()
        db.close()


def test_retrieve_patient_detections(test_client_with_results):
    """Check that the detection endpoint returns saved results."""

    client, patient_id = test_client_with_results

    response = client.get(
        f"/patients/{patient_id}/detections"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["patient_id"] == patient_id
    assert data["total_results"] == 1
    assert len(data["detections"]) == 1

    detection = data["detections"][0]

    assert detection["statistical_anomaly_flag"] is False
    assert detection["hybrid_or_flag"] is False


def test_retrieve_patient_verifications(test_client_with_results):
    """Check that the verification endpoint returns saved results."""

    client, patient_id = test_client_with_results

    response = client.get(
        f"/patients/{patient_id}/verifications"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["patient_id"] == patient_id
    assert data["total_results"] == 1
    assert len(data["verifications"]) == 1

    verification = data["verifications"][0]

    assert verification["verification_status"] == "no_model_alert"
    assert (
        verification["explainable_reason"]
        == "No anomaly evidence was identified."
    )


def test_retrieve_results_for_unknown_patient(
    test_client_with_results,
):
    """Check that an unknown patient receives HTTP 404."""

    client, _ = test_client_with_results

    response = client.get(
        "/patients/UNKNOWN-TEST-PATIENT/detections"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Patient not found."
