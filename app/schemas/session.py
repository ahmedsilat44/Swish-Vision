from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional


class SessionResponse(BaseModel):
    id: int
    user_id: int
    original_filename: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class SessionListResponse(BaseModel):
    id: int
    original_filename: str
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ShotDetail(BaseModel):
    shot_number: int
    outcome: str  # "made" or "missed"
    release_angle: Optional[float] = None
    elbow_angle_at_release: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class ShotAnalyticsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    session_id: int
    shots: list[ShotDetail]
    total_shots: int
    makes: int
    misses: int


class ReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    session_id: int
    shot_percentage: Optional[float] = None
    shots_made: int = 0
    shots_missed: int = 0
    total_shots: int = 0
    avg_release_angle: Optional[float] = None
    feedback_text: Optional[str] = None


class AngleFrameDetail(BaseModel):
    shot_number: int
    frame_number: int
    elbow_angle: Optional[float] = None
    knee_angle: Optional[float] = None
    shoulder_angle: Optional[float] = None
    outcome: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class AngleDataResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    session_id: int
    frames: list[AngleFrameDetail]


class ShotEventResponse(ShotDetail):
    model_config = ConfigDict(from_attributes=True)


class AngleFrameResponse(AngleFrameDetail):
    model_config = ConfigDict(from_attributes=True)


class DashboardSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    total_sessions: int
    total_shots: int
    total_makes: int
    shot_percentage: Optional[float] = None
    avg_consistency: Optional[float] = None


class TrendPointResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    session_id: int
    created_at: datetime
    total_shots: int
    makes: int
    misses: int
