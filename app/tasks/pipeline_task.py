from datetime import datetime
import os
import shutil
import traceback
from datetime import datetime, timezone
from celery import Celery
from app.config import settings

celery_app = Celery(
    "swishvision",
    broker=settings.REDIS_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, max_retries=0)
def process_video(self, session_id: int):
    """Celery task wrapping the CV pipeline for a given session."""
    from app.database import SessionLocal
    from app.models.session import SessionModel

    db = SessionLocal()
    session = None
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

        # Run the CV pipeline
        from main import run_pipeline
        output_path, report_path = run_pipeline(input_path, session_id=session_id)

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise FileNotFoundError(f"Output video missing or empty: {output_path}")
        if not os.path.exists(report_path) or os.path.getsize(report_path) == 0:
            raise FileNotFoundError(f"Report file missing or empty: {report_path}")

        session.output_path = output_path
        session.report_path = report_path
        session.status = "completed"
        session.completed_at = datetime.now(timezone.utc)

        db.commit()

    except Exception as exc:
        print("ERROR: Exception during video processing:")
        traceback.print_exc()
        if session is not None:
            session.status = "failed"
            session.error_message = str(exc) or "An unexpected error occurred during processing."
            db.commit()

        if 'input_path' in locals() and os.path.exists(input_path):
            print(input_path)
            os.remove(input_path)

    finally:
        db.close()
