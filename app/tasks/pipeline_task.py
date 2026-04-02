import os
import shutil
import traceback
from celery import Celery
from app.config import settings

celery_app = Celery("swishvision", broker=settings.REDIS_URL)


@celery_app.task(bind=True, max_retries=0)
def process_video(self, session_id: int):
    """Celery task wrapping the CV pipeline for a given session."""
    from app.database import SessionLocal
    from app.models.session import SessionModel

    db = SessionLocal()
    try:
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if not session:
            return

        session.status = "processing"
        db.commit()

        # Copy uploaded video to input_videos/ for pipeline
        input_dir = "input_videos"
        os.makedirs(input_dir, exist_ok=True)
        input_path = os.path.join(input_dir, os.path.basename(session.upload_path))
        shutil.copy2(session.upload_path, input_path)

        # TODO: Import and call the CV pipeline
        # from main import main_pipeline
        # main_pipeline(input_path)

        # TODO: Parse report and persist to DB
        # TODO: Set session.output_path and session.report_path

        session.status = "completed"
        db.commit()

    except Exception:
        traceback.print_exc()
        session.status = "failed"
        db.commit()
    finally:
        db.close()
