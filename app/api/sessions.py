import os
import uuid
import logging
import mimetypes
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Query
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app.database import get_db
from app.config import settings
from app.models.user import User
from app.models.session import SessionModel
from app.models.report import Report
from app.models.shot_event import ShotEvent
from app.models.angle_frame import AngleFrame
from app.schemas.session import (
    SessionResponse,
    SessionListResponse,
    ReportResponse,
    ShotAnalyticsResponse,
    AngleDataResponse,
)
from app.core.security import get_current_user, get_current_user_token, get_session_or_403
from app.tasks.pipeline_task import process_video

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

# Short-lived media tokens: {token_str: (session_id, user_id, expires_at)}
# These are single-request UUID tokens used so the JWT never appears in video URLs.
_video_tokens: dict[str, tuple[int, int, datetime]] = {}
_VIDEO_TOKEN_TTL = timedelta(minutes=5)

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
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid MIME type")

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    bytes_written = 0
    filepath = None
    session = None

    try:
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        session = SessionModel(user_id=current_user.id, original_filename=file.filename, status="uploading")
        db.add(session)
        db.commit()
        db.refresh(session)

        filepath = os.path.join(settings.UPLOAD_DIR, f"{session.id}{ext}")
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

        session.status = "queued"
        session.upload_path = filepath
        db.commit()
        db.refresh(session)
    except HTTPException:
        raise
    except OSError as exc:
        logging.exception("Upload file write failed: %s", exc)
        db.rollback()
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass
        if session and session.id:
            try:
                db.delete(session)
                db.commit()
            except Exception:
                db.rollback()
        raise HTTPException(status_code=500, detail="Server failed to store uploaded file")
    except SQLAlchemyError as exc:
        logging.exception("Upload DB operation failed: %s", exc)
        db.rollback()
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass
        raise HTTPException(
            status_code=500,
            detail="Database error during upload. If schema changed, run migrations (alembic upgrade head) and restart the backend.",
        )
    finally:
        await file.close()

    try:
        process_video.delay(session.id)
    except Exception as exc:
        logging.warning("Celery dispatch failed (is Redis running?): %s", exc)
        session.status = "pending"
        db.commit()

    return session


@router.get("/", response_model=list[SessionListResponse])
def list_sessions(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    latest_report_id = (
        db.query(func.max(Report.id))
        .filter(Report.session_id == SessionModel.id)
        .correlate(SessionModel)
        .scalar_subquery()
    )
    rows = (
        db.query(SessionModel, Report)
        .outerjoin(Report, Report.id == latest_report_id)
        .filter(SessionModel.user_id == current_user.id)
        .order_by(SessionModel.created_at.desc())
        .all()
    )
    result = []
    for session, report in rows:
        if report is None:
            shot_percentage = None
            shots_made = None
            shots_missed = None
            total_shots = None
        else:
            total = report.total_shots or 0
            makes = report.makes or 0
            misses = report.misses or 0
            shot_percentage = round(makes / total * 100, 1) if total > 0 else None
            shots_made = makes
            shots_missed = misses
            total_shots = total
        result.append(
            SessionListResponse(
                id=session.id,
                original_filename=session.original_filename,
                status=session.status,
                created_at=session.created_at,
                shot_percentage=shot_percentage,
                shots_made=shots_made,
                shots_missed=shots_missed,
                total_shots=total_shots,
            )
        )
    return result


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session: SessionModel = Depends(verify_session_ownership)):
    return session


@router.get("/{session_id}/report", response_model=ReportResponse)
def get_report(
    session: SessionModel = Depends(verify_session_ownership),
    db: Session = Depends(get_db),
):
    report = db.query(Report).filter(Report.session_id == session.id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not available yet. Session may still be processing.")
    total = report.total_shots or 0
    makes = report.makes or 0
    
    # Calculate average SEW angle (stored in elbow_angle column)
    avg_sew_angle = (
        db.query(func.avg(ShotEvent.elbow_angle))
        .filter(ShotEvent.session_id == session.id, ShotEvent.elbow_angle.isnot(None))
        .scalar()
    )
    
    # Calculate average ESH angle (stored in shoulder_angle column)
    avg_esh_angle = (
        db.query(func.avg(ShotEvent.shoulder_angle))
        .filter(ShotEvent.session_id == session.id, ShotEvent.shoulder_angle.isnot(None))
        .scalar()
    )
    
    return ReportResponse(
        session_id=session.id,
        shot_percentage=round(makes / total * 100, 1) if total > 0 else None,
        shots_made=makes,
        shots_missed=report.misses or 0,
        total_shots=total,
        avg_sew_angle=round(avg_sew_angle, 1) if avg_sew_angle is not None else None,
        avg_esh_angle=round(avg_esh_angle, 1) if avg_esh_angle is not None else None,
        feedback_text=report.raw_text,
    )

@router.get("/{session_id}/shots", response_model=ShotAnalyticsResponse)
def get_shots(session: SessionModel = Depends(verify_session_ownership), db: Session = Depends(get_db)):
    shots = db.query(ShotEvent).filter(ShotEvent.session_id == session.id).order_by(ShotEvent.shot_number).all()

    def normalize_result(result):
        if result in ("make", "made", "1", 1, True):
            return "made"
        if result in ("miss", "missed", "0", 0, False):
            return "missed"
        return "missed"

    normalized_shots = [
        {
            "shot_number": s.shot_number,
            "outcome": normalize_result(s.result),
            "sew_angle": s.elbow_angle,
            "esh_angle": s.shoulder_angle,
        }
        for s in shots
    ]

    makes = sum(1 for s in normalized_shots if s["outcome"] == "made")
    misses = len(shots) - makes

    return {
        "session_id": session.id,
        "shots": normalized_shots,
        "total_shots": len(shots),
        "makes": makes,
        "misses": misses,
    }

@router.get("/{session_id}/video_token")
def get_video_token(
    session: SessionModel = Depends(verify_session_ownership),
    current_user: User = Depends(get_current_user),
):
    """Issue a short-lived single-use media token so the JWT never appears in video URLs."""
    # Lazily purge expired tokens to prevent unbounded growth
    now = datetime.now(timezone.utc)
    expired_keys = [k for k, (_, _, exp) in _video_tokens.items() if now > exp]
    for k in expired_keys:
        _video_tokens.pop(k, None)

    token_str = str(uuid.uuid4())
    _video_tokens[token_str] = (session.id, current_user.id, now + _VIDEO_TOKEN_TTL)
    return {"video_token": token_str, "expires_in": int(_VIDEO_TOKEN_TTL.total_seconds())}


@router.get("/{session_id}/output_video")
def get_output_video(
    session_id: int,
    token: Optional[str] = Query(None),
    video_token: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    # Authenticate via short-lived media token (preferred — JWT never in URL)
    if video_token:
        entry = _video_tokens.get(video_token)
        if not entry:
            raise HTTPException(status_code=401, detail="Invalid or expired video token")
        vid_session_id, _user_id, expires_at = entry
        if datetime.now(timezone.utc) > expires_at:
            _video_tokens.pop(video_token, None)
            raise HTTPException(status_code=401, detail="Video token expired")
        if vid_session_id != session_id:
            raise HTTPException(status_code=403, detail="Token does not match session")
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    elif token:
        current_user = get_current_user_token(token, db)
        session = get_session_or_403(session_id, current_user, db)
    else:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    if not session.output_path:
        raise HTTPException(status_code=404, detail="Output video path not set")
    
    video_path = session.output_path
    logging.info(f"Attempting to serve video from: {video_path}")
    
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        raise HTTPException(status_code=404, detail=f"Output video not found at {video_path}")

    file_size = os.path.getsize(video_path)
    if file_size == 0:
        logging.error(f"Video file is empty: {video_path}")
        raise HTTPException(status_code=500, detail="Output video is empty")
    
    logging.info(f"Serving video file: {video_path} (size: {file_size} bytes)")
    
    # Force video/mp4 for MP4 files
    if video_path.lower().endswith('.mp4'):
        media_type = "video/mp4"
    else:
        mime_type, _ = mimetypes.guess_type(video_path)
        media_type = mime_type or "video/mp4"
    
    logging.info(f"Using media type: {media_type}")
    
    return FileResponse(
        path=video_path,
        media_type=media_type,
        filename=os.path.basename(video_path),
    )


@router.get("/{session_id}/angles", response_model=AngleDataResponse)
def get_angles(
    session: SessionModel = Depends(verify_session_ownership),
    db: Session = Depends(get_db),
):
    from app.models.angle_frame import AngleFrame

    # Get all angle frames
    angle_rows = (
        db.query(AngleFrame)
        .filter(AngleFrame.session_id == session.id)
        .order_by(AngleFrame.frame_number)
        .all()
    )
    
    # Get all shot events to map frames to shots
    shot_events = (
        db.query(ShotEvent)
        .filter(ShotEvent.session_id == session.id)
        .order_by(ShotEvent.shot_number)
        .all()
    )
    
    # Create a frame -> shot_number mapping
    frame_to_shot = {}
    for shot in shot_events:
        if shot.start_frame and shot.end_frame:
            for frame_num in range(shot.start_frame, shot.end_frame + 1):
                frame_to_shot[frame_num] = shot.shot_number
    
    # Build response with shot numbers properly mapped
    frames = [
        {
            "shot_number": frame_to_shot.get(row.frame_number, 1),  # Default to 1 if not found
            "frame_number": row.frame_number,
            "elbow_angle": row.elbow_angle,
            "knee_angle": row.knee_angle,
            "shoulder_angle": row.shoulder_angle,
            "outcome": None,
        }
        for row in angle_rows
    ]
    
    return {"session_id": session.id, "frames": frames}

@router.post("/{session_id}/retry", response_model=SessionResponse)
def retry_session(
    session: SessionModel = Depends(verify_session_ownership),
    db: Session = Depends(get_db),
):
    if session.status != "failed":
        raise HTTPException(
            status_code=409,
            detail="Only failed sessions can be retried",
        )

    session.status = "queued"
    session.completed_at = None
    db.commit()
    db.refresh(session)

    try:
        process_video.delay(session.id)
    except Exception as exc:
        logging.warning("Celery dispatch failed on retry (is Redis running?): %s", exc)
        session.status = "pending"
        db.commit()
        db.refresh(session)

    return session


@router.delete("/{session_id}", status_code=204)
def delete_session(
    session: SessionModel = Depends(verify_session_ownership),
    db: Session = Depends(get_db),
):
    if session.status == "processing":
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a session while it is being processed",
        )

    for path in [session.upload_path, session.output_path, session.report_path]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    db.query(Report).filter(Report.session_id == session.id).delete(synchronize_session=False)
    db.query(ShotEvent).filter(ShotEvent.session_id == session.id).delete(synchronize_session=False)
    db.query(AngleFrame).filter(AngleFrame.session_id == session.id).delete(synchronize_session=False)
    db.delete(session)
    db.commit()
