import os
import shutil
import logging
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlalchemy.orm import Session
from app.database import get_db
from app.config import settings
from app.models.user import User
from app.models.session import SessionModel
from app.models.report import Report
from app.models.shot_event import ShotEvent
from app.schemas.session import (
    SessionResponse, SessionListResponse,
    ShotAnalyticsResponse, ShotDetail, AngleDataResponse, ReportResponse,
)
from app.core.security import get_current_user, get_session_or_403
from app.tasks.pipeline_task import process_video

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_MIME_TYPES = {"video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska"}

def verify_session_ownership(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Dependency that validates session ownership"""
    return get_session_or_403(session_id, current_user, db)

@router.post("/upload", response_model=SessionResponse, status_code=201)
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Validate MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid MIME type")

    # Save file
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    session = SessionModel(user_id=current_user.id, original_filename=file.filename, status="uploading")
    db.add(session)
    db.commit()
    db.refresh(session)

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    filepath = os.path.join(settings.UPLOAD_DIR, f"{session.id}{ext}")
    bytes_written = 0
    try:
        with open(filepath, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    f.close()
                    os.remove(filepath)
                    db.delete(session)
                    db.commit()
                    raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit")
                f.write(chunk)
    finally:
        await file.close()

    session.status = "queued"
    session.upload_path = filepath
    db.commit()
    db.refresh(session)

    from app.tasks.pipeline_task import process_video
    try:
        process_video.delay(session.id)
    except Exception as exc:
        logging.warning("Celery dispatch failed (is Redis running?): %s", exc)
        session.status = "pending"
        db.commit()

    return session

@router.get("/", response_model=list[SessionListResponse])
def list_sessions(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    sessions = db.query(SessionModel).filter(SessionModel.user_id == current_user.id).order_by(SessionModel.created_at.desc()).all()
    return sessions

@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session: SessionModel = Depends(verify_session_ownership)):
    return session

@router.get("/{session_id}/shots", response_model=ShotAnalyticsResponse)
def get_shots(
    session: SessionModel = Depends(verify_session_ownership),
    db: Session = Depends(get_db),
):
    events = (
        db.query(ShotEvent)
        .filter(ShotEvent.session_id == session.id)
        .order_by(ShotEvent.shot_number)
        .all()
    )
    outcome_map = {"make": "made", "miss": "missed"}
    shots = [
        ShotDetail(
            shot_number=e.shot_number,
            outcome=outcome_map.get(e.result, e.result),
            release_angle=e.release_angle,
            elbow_angle_at_release=e.elbow_angle_at_release,
        )
        for e in events
    ]
    makes = sum(1 for e in events if e.result == "make")
    misses = sum(1 for e in events if e.result == "miss")
    return ShotAnalyticsResponse(
        session_id=session.id,
        shots=shots,
        total_shots=len(events),
        makes=makes,
        misses=misses,
    )


@router.get("/{session_id}/report", response_model=ReportResponse)
def get_report(
    session: SessionModel = Depends(verify_session_ownership),
    db: Session = Depends(get_db),
):
    report = db.query(Report).filter(Report.session_id == session.id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found for this session")
    pct = (report.makes / report.total_shots * 100.0) if report.total_shots else 0.0
    return ReportResponse(
        session_id=session.id,
        shot_percentage=round(pct, 1),
        shots_made=report.makes,
        shots_missed=report.misses,
        total_shots=report.total_shots,
    )


@router.get("/{session_id}/angles", response_model=AngleDataResponse)
def get_angles(session: SessionModel = Depends(verify_session_ownership)):
    return {"session_id": session.id, "frames": []}

@router.delete("/{session_id}", status_code=204)
def delete_session(
    session: SessionModel = Depends(verify_session_ownership),
    db: Session = Depends(get_db),
):
    # Guard: cannot delete while processing
    if session.status == "processing":
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a session while it is being processed"
        )

    # Remove all associated files
    for path in [session.upload_path, session.output_path, session.report_path]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass  # File may have already been removed; proceed with DB deletion

    db.delete(session)
    db.commit()
