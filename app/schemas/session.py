from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class SessionResponse(BaseModel):
    id: int
    user_id: int
    original_filename: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SessionListResponse(BaseModel):
    id: int
    original_filename: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ShotDetail(BaseModel):
    shot_number: int
    result: str
    release_angle: Optional[float] = None
    elbow_angle_at_release: Optional[float] = None


class ShotAnalyticsResponse(BaseModel):
    session_id: int
    shots: list[ShotDetail]
    total_shots: int
    makes: int
    misses: int


class AngleFrameDetail(BaseModel):
    frame_number: int
    elbow_angle: Optional[float] = None
    knee_angle: Optional[float] = None
    shoulder_angle: Optional[float] = None


class AngleDataResponse(BaseModel):
    session_id: int
    frames: list[AngleFrameDetail]


class ReportResponse(BaseModel):
    shot_percentage: Optional[float] = None
    shots_made: int = 0
    shots_missed: int = 0
    avg_release_angle: Optional[float] = None
    feedback_text: Optional[str] = None


class ShotEventResponse(BaseModel):
    shot_number: int
    result: str
    release_angle: Optional[float] = None
    elbow_angle_at_release: Optional[float] = None

    class Config:
        from_attributes = True


class AngleFrameResponse(BaseModel):
    frame_number: int
    elbow_angle: Optional[float] = None
    knee_angle: Optional[float] = None
    shoulder_angle: Optional[float] = None

    class Config:
        from_attributes = True


class DashboardSummaryResponse(BaseModel):
    total_sessions: int
    completed_sessions: int
    total_shots: int
    total_makes: int
    shot_percentage: Optional[float] = None


class TrendPointResponse(BaseModel):
    session_id: int
    created_at: datetime
    total_shots: int
    makes: int
    misses: int
