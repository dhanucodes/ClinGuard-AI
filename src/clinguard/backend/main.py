
"""FastAPI application for ClinGuard AI."""

import pandas as pd

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from clinguard.database.session import get_db
from clinguard.database.models import (
    VitalReading,
    DetectionResult,
    VerificationResult,
)
from clinguard.database.crud import (
    create_patient,
    get_patient,
    create_vital_reading,
    get_patient_readings,
    save_detection_and_verification,
)
from clinguard.backend.schemas import (
    PatientCreate,
    PatientResponse,
    VitalReadingCreate,
    VitalReadingResponse,
)
from clinguard.backend.processing import process_patient_history


app = FastAPI(
    title="ClinGuard AI API",
    description=(
        "Research prototype API for synthetic patient monitoring "
        "and context-aware anomaly analysis."
    ),
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "Welcome to ClinGuard AI API",
        "status": "running",
    }


@app.post(
    "/patients",
    response_model=PatientResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_patient(
    patient_data: PatientCreate,
    db: Session = Depends(get_db),
):
    existing_patient = get_patient(db, patient_data.patient_id)

    if existing_patient:
        raise HTTPException(
            status_code=409,
            detail="Patient already exists.",
        )

    try:
        return create_patient(db, patient_data.patient_id)

    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Patient already exists.",
        )


@app.get(
    "/patients/{patient_id}",
    response_model=PatientResponse,
)
def retrieve_patient(
    patient_id: str,
    db: Session = Depends(get_db),
):
    patient = get_patient(db, patient_id)

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail="Patient not found.",
        )

    return patient


@app.post(
    "/readings",
    response_model=VitalReadingResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_reading(
    reading_data: VitalReadingCreate,
    db: Session = Depends(get_db),
):
    patient = get_patient(db, reading_data.patient_id)

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail="Register the patient before submitting readings.",
        )

    try:
        return create_vital_reading(
            db,
            reading_data.model_dump(),
        )

    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail=(
                "A reading already exists for this patient "
                "and timestamp."
            ),
        )


@app.get(
    "/patients/{patient_id}/readings",
    response_model=list[VitalReadingResponse],
)
def retrieve_patient_readings(
    patient_id: str,
    db: Session = Depends(get_db),
):
    patient = get_patient(db, patient_id)

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail="Patient not found.",
        )

    return get_patient_readings(db, patient_id)


@app.post("/patients/{patient_id}/process")
def process_patient(
    patient_id: str,
    db: Session = Depends(get_db),
):
    patient = get_patient(db, patient_id)

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail="Patient not found.",
        )

    readings = get_patient_readings(db, patient_id)

    if len(readings) < 7:
        return {
            "status": "insufficient_history",
            "message": (
                "Not enough historical readings "
                "to establish an adaptive baseline."
            ),
            "patient_id": patient_id,
            "available_readings": len(readings),
            "required_readings": 7,
        }

    records = [
        {
            "reading_id": reading.reading_id,
            "patient_id": reading.patient_id,
            "timestamp": reading.timestamp,
            "heart_rate": reading.heart_rate,
            "spo2": reading.spo2,
            "systolic_bp": reading.systolic_bp,
            "diastolic_bp": reading.diastolic_bp,
            "respiratory_rate": reading.respiratory_rate,
            "temperature": reading.temperature,
            "has_missing_value": reading.has_missing_value,
            "has_noise": reading.has_noise,
            "has_sensor_error": reading.has_sensor_error,
            "data_quality_issue": reading.data_quality_issue,
        }
        for reading in readings
    ]

    patient_history = pd.DataFrame(records)

    try:
        result = process_patient_history(patient_history)

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    processed_dataset = result["dataset"]

    for _, row in processed_dataset.iterrows():

        score = row["isolation_forest_score"]

        if pd.isna(score):
            score = None
        else:
            score = float(score)

        detection_data = {
            "statistical_anomaly_flag": bool(
                row["statistical_anomaly_flag"]
            ),
            "isolation_forest_anomaly_flag": bool(
                row["isolation_forest_anomaly_flag"]
            ),
            "hybrid_or_flag": bool(row["hybrid_or_flag"]),
            "hybrid_and_flag": bool(row["hybrid_and_flag"]),
            "isolation_forest_score": score,
        }

        verification_data = {
            "verification_status": str(
                row["verification_status"]
            ),
            "verification_reason": str(
                row["verification_reason"]
            ),
            "context_issues": str(row["context_issues"]),
            "explainable_reason": str(
                row["explainable_reason"]
            ),
        }

        save_detection_and_verification(
            db=db,
            reading_id=int(row["reading_id"]),
            detection_data=detection_data,
            verification_data=verification_data,
        )

    latest_record = processed_dataset.iloc[-1]

    return {
        "status": result["status"],
        "patient_id": patient_id,
        "total_records": len(processed_dataset),
        "saved_results": len(processed_dataset),
        "summary": result["summary"],
        "latest_record": {
            "timestamp": str(latest_record["timestamp"]),
            "verification_status": str(
                latest_record["verification_status"]
            ),
            "explanation": str(
                latest_record["explainable_reason"]
            ),
        },
    }


# NEW ENDPOINT 1: Retrieve saved detection results.


@app.get("/patients/{patient_id}/detections")
def retrieve_patient_detections(
    patient_id: str,
    db: Session = Depends(get_db),
):
    patient = get_patient(db, patient_id)

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail="Patient not found.",
        )

    statement = (
        select(
            VitalReading.reading_id,
            VitalReading.timestamp,
            DetectionResult.detection_id,
            DetectionResult.statistical_anomaly_flag,
            DetectionResult.isolation_forest_anomaly_flag,
            DetectionResult.hybrid_or_flag,
            DetectionResult.hybrid_and_flag,
            DetectionResult.isolation_forest_score,
        )
        .join(
            DetectionResult,
            DetectionResult.reading_id == VitalReading.reading_id,
        )
        .where(VitalReading.patient_id == patient_id)
        .order_by(VitalReading.timestamp)
    )

    results = db.execute(statement).mappings().all()

    return {
        "patient_id": patient_id,
        "total_results": len(results),
        "detections": [dict(row) for row in results],
    }


# NEW ENDPOINT 2: Retrieve saved verification results.


@app.get("/patients/{patient_id}/verifications")
def retrieve_patient_verifications(
    patient_id: str,
    db: Session = Depends(get_db),
):
    patient = get_patient(db, patient_id)

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail="Patient not found.",
        )

    statement = (
        select(
            VitalReading.reading_id,
            VitalReading.timestamp,
            DetectionResult.detection_id,
            VerificationResult.verification_id,
            VerificationResult.verification_status,
            VerificationResult.verification_reason,
            VerificationResult.context_issues,
            VerificationResult.explainable_reason,
        )
        .join(
            DetectionResult,
            DetectionResult.reading_id == VitalReading.reading_id,
        )
        .join(
            VerificationResult,
            VerificationResult.detection_id
            == DetectionResult.detection_id,
        )
        .where(VitalReading.patient_id == patient_id)
        .order_by(VitalReading.timestamp)
    )

    results = db.execute(statement).mappings().all()

    return {
        "patient_id": patient_id,
        "total_results": len(results),
        "verifications": [dict(row) for row in results],
    }
