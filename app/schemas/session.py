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
    shot_percentage: Optional[float] = None
    shots_made: Optional[int] = None
    shots_missed: Optional[int] = None
    total_shots: Optional[int] = None

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



