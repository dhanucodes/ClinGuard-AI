"""Pydantic request and response schemas for ClinGuard AI."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PatientCreate(BaseModel):
    patient_id: str = Field(min_length=1, max_length=50)


class PatientResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    patient_id: str
    created_at: datetime


class VitalReadingCreate(BaseModel):
    patient_id: str = Field(min_length=1, max_length=50)
    timestamp: datetime

    heart_rate: float | None = None
    spo2: float | None = None
    systolic_bp: float | None = None
    diastolic_bp: float | None = None
    respiratory_rate: float | None = None
    temperature: float | None = None

    has_missing_value: bool = False
    has_noise: bool = False
    has_sensor_error: bool = False
    data_quality_issue: bool = False


class VitalReadingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    reading_id: int
    patient_id: str
    timestamp: datetime

    heart_rate: float | None
    spo2: float | None
    systolic_bp: float | None
    diastolic_bp: float | None
    respiratory_rate: float | None
    temperature: float | None

    has_missing_value: bool
    has_noise: bool
    has_sensor_error: bool
    data_quality_issue: bool