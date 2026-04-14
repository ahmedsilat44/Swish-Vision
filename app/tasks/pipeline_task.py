from datetime import datetime
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

        vid_name = os.path.splitext(os.path.basename(session.upload_path))[0]
        from main import main_pipeline
        main_pipeline(input_path)

        output_vid_path = f"output_videos/output_{vid_name}_processed.avi"
        report_path = f"output_videos/output_{vid_name}_report.txt"

        if not os.path.exists(output_vid_path) or os.path.getsize(output_vid_path) == 0:
            raise FileNotFoundError(f"Output video missing or empty: {output_vid_path}")
        if not os.path.exists(report_path) or not os.path.getsize(report_path) == 0:
            raise FileNotFoundError(f"Report file missing or empty: {report_path}")

        session.output_video_path = output_vid_path
        session.report_path = report_path
        session.status = "completed"
        session.completion_time = datetime.utcnow()
        db.commit()

    except Exception:
        traceback.print_exc()
        session.status = "failed"
        db.commit()
    finally:
        db.close()
