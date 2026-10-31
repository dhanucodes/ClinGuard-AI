"""FastAPI application for ClinGuard AI."""
from clinguard.backend.processing import process_patient_history
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from clinguard.database.session import get_db

from clinguard.database.crud import (
    create_patient,
    get_patient,
    create_vital_reading,
    get_patient_readings,
)

from clinguard.backend.schemas import (
    PatientCreate,
    PatientResponse,
    VitalReadingCreate,
    VitalReadingResponse,
)


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

    existing_patient = get_patient(
        db,
        patient_data.patient_id,
    )

    if existing_patient:
        raise HTTPException(
            status_code=409,
            detail="Patient already exists.",
        )

    try:
        patient = create_patient(
            db,
            patient_data.patient_id,
        )
        return patient

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

    patient = get_patient(
        db,
        reading_data.patient_id,
    )

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail="Register the patient before submitting readings.",
        )

    try:
        reading = create_vital_reading(
            db,
            reading_data.model_dump(),
        )

        return reading

    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="A reading already exists for this patient and timestamp.",
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
def process_patient(patient_id: str, db: Session = Depends(get_db)):

    # 1. Check whether the patient exists.
    patient = get_patient(db, patient_id)

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail="Patient not found."
        )

    # 2. Retrieve the patient's stored readings.
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

    # 3. Convert database records into a DataFrame.
    records = [
        {
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

    # 4. Run the AI processing pipeline.
    try:
        result = process_patient_history(patient_history)

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    # 5. Return the processing summary.
    latest_record = result["dataset"].iloc[-1]

    return {
        "status": result["status"],
        "patient_id": patient_id,
        "total_records": len(result["dataset"]),
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