import os
import shutil
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlalchemy.orm import Session
from app.database import get_db
from app.config import settings
from app.models.user import User
from app.models.session import SessionModel
from app.schemas.session import SessionResponse, SessionListResponse, ShotAnalyticsResponse, AngleDataResponse
from app.core.security import get_current_user

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_MIME_TYPES = {"video/mp4", "video/x-msvideo", "video/quicktime", "video/x-matroska"}


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

    filepath = os.path.join(settings.UPLOAD_DIR, f"{session.id}{ext}")
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    session.status = "queued"
    session.upload_path = filepath
    db.commit()
    db.refresh(session)

    # TODO: Trigger Celery task here
    # from app.tasks.pipeline_task import process_video
    # process_video.delay(session.id)

    return session


@router.get("/", response_model=list[SessionListResponse])
def list_sessions(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    sessions = db.query(SessionModel).filter(SessionModel.user_id == current_user.id).order_by(SessionModel.created_at.desc()).all()
    return sessions


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    session = db.query(SessionModel).filter(SessionModel.id == session_id, SessionModel.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/{session_id}/shots", response_model=ShotAnalyticsResponse)
def get_shots(session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    session = db.query(SessionModel).filter(SessionModel.id == session_id, SessionModel.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # TODO: Query shot_events and build response
    return {"session_id": session_id, "shots": [], "total_shots": 0, "makes": 0, "misses": 0}


@router.get("/{session_id}/angles", response_model=AngleDataResponse)
def get_angles(session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    session = db.query(SessionModel).filter(SessionModel.id == session_id, SessionModel.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # TODO: Query angle_frames and build response
    return {"session_id": session_id, "frames": []}


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    session = db.query(SessionModel).filter(SessionModel.id == session_id, SessionModel.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Remove files
    if session.upload_path and os.path.exists(session.upload_path):
        os.remove(session.upload_path)
    if session.output_path and os.path.exists(session.output_path):
        os.remove(session.output_path)

    db.delete(session)
    db.commit()
