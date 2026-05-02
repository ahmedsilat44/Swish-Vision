from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models.user import User
from app.models.session import SessionModel
from app.models.report import Report
from app.core.security import get_current_user
from app.schemas.session import DashboardSummaryResponse, TrendPointResponse

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummaryResponse)
def dashboard_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    total_sessions = (
        db.query(func.count(SessionModel.id))
        .filter(SessionModel.user_id == current_user.id)
        .scalar()
    )
    agg = (
        db.query(
            func.count(func.distinct(SessionModel.id)).label("completed_sessions"),
            func.coalesce(func.sum(Report.total_shots), 0).label("total_shots"),
            func.coalesce(func.sum(Report.makes), 0).label("total_makes"),
        )
        .outerjoin(Report, Report.session_id == SessionModel.id)
        .filter(
            SessionModel.user_id == current_user.id,
            SessionModel.status == "completed",
        )
        .one()
    )
    completed_sessions = int(agg.completed_sessions)
    total_shots = int(agg.total_shots)
    total_makes = int(agg.total_makes)
    shot_percentage = round(total_makes / total_shots * 100, 1) if total_shots > 0 else None

    return DashboardSummaryResponse(
        total_sessions=total_sessions,
        completed_sessions=completed_sessions,
        total_shots=total_shots,
        total_makes=total_makes,
        shot_percentage=shot_percentage,
    )


@router.get("/trends", response_model=List[TrendPointResponse])
def dashboard_trends(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = (
        db.query(
            SessionModel.id.label("session_id"),
            SessionModel.created_at,
            Report.total_shots,
            Report.makes,
            Report.misses,
        )
        .join(Report, Report.session_id == SessionModel.id)
        .filter(
            SessionModel.user_id == current_user.id,
            SessionModel.status == "completed",
        )
        .order_by(SessionModel.created_at.asc())
        .all()
    )
    return [
        TrendPointResponse(
            session_id=r.session_id,
            created_at=r.created_at,
            total_shots=r.total_shots or 0,
            makes=r.makes or 0,
            misses=r.misses or 0,
        )
        for r in rows
    ]
