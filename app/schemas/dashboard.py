from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DashboardSummaryResponse(BaseModel):
    total_sessions: int
    total_shots: int
    total_makes: int
    shot_percentage: Optional[float] = None
    avg_consistency: Optional[float] = None

    class Config:
        from_attributes = True


class TrendPointResponse(BaseModel):
    session_id: int
    created_at: datetime
    total_shots: int
    makes: int
    misses: int

    class Config:
        from_attributes = True