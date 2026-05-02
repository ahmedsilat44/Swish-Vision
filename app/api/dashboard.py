from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models.user import User
from app.models.session import SessionModel
from app.models.shot_event import ShotEvent
from app.core.security import get_current_user

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary")
def dashboard_summary(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    total_sessions = db.query(func.count(SessionModel.id)).filter(SessionModel.user_id == current_user.id).scalar()
    completed_sessions = (
        db.query(func.count(SessionModel.id))
        .filter(SessionModel.user_id == current_user.id, SessionModel.status == "completed")
        .scalar()
    )
    # TODO: Aggregate shot stats from ShotEvent table
    return {
        "total_sessions": total_sessions,
        "completed_sessions": completed_sessions,
        "total_shots": 0,
        "total_makes": 0,
        "shot_percentage": 0.0,
    }


@router.get("/trends")
def dashboard_trends(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # TODO: Build trend data across sessions
    return {"trends": []}


