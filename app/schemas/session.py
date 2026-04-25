from pydantic import BaseModel
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
