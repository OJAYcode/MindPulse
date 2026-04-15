from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class InferenceResultIn(BaseModel):
    timestamp: datetime
    face_emotion: str = Field(min_length=1)
    face_confidence: float = Field(ge=0.0, le=1.0)
    voice_emotion: str = Field(min_length=1)
    voice_confidence: float = Field(ge=0.0, le=1.0)
    stress_level: str = Field(pattern="^(low|medium|high)$")
    source: str = Field(min_length=1)


class InferenceResultOut(InferenceResultIn):
    id: int
    user_id: int | None = None


class HealthResponse(BaseModel):
    status: str
    database: str


class UserCreate(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    username: str = Field(min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_]+$")
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=128)


class UserLogin(BaseModel):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    id: int
    name: str
    username: str
    email: str
    created_at: datetime


class AuthResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    user: UserOut


class UserProfileUpdate(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    username: str = Field(min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_]+$")
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class StatusResponse(BaseModel):
    status: str


class StressDistribution(BaseModel):
    low: int = 0
    medium: int = 0
    high: int = 0


class DashboardSummary(BaseModel):
    total_results: int
    latest_result: InferenceResultOut | None
    stress_distribution: StressDistribution
    average_face_confidence: float
    average_voice_confidence: float
    high_stress_rate: float
    recent_results: list[InferenceResultOut]
