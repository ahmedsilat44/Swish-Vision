from datetime import datetime, timezone
import os
import shutil
import traceback
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

def _cleanup_transient_files():
    """Delete intermediate files written during pipeline inference.
    Safe to call on both success and failure paths.
    Never deletes uploaded videos or output files."""
    transient = ["angs.txt", "xy_coords.txt", "detections.txt", "ball_locl.txt"]
    for fname in transient:
        if os.path.exists(fname):
            try:
                os.remove(fname)
                print(f"Cleaned up transient file: {fname}")
            except OSError as e:
                print(f"Could not delete transient file {fname}: {e}")


@celery_app.task(bind=True, max_retries=0)
def process_video(self, session_id: int):
    """Celery task wrapping the CV pipeline for a given session."""
    from app.database import SessionLocal
    from app.models.session import SessionModel
    from app.models.angle_frame import AngleFrame

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
        output_path, report_path, pipeline_data = run_pipeline(input_path, session_id=session_id)
        print(output_path)
        print(report_path)
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output video missing: {output_path}")
        if os.path.getsize(output_path) == 0:
            raise ValueError(f"Output video is empty: {output_path}")
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Report file missing: {report_path}")
        if os.path.getsize(report_path) == 0:
            raise ValueError(f"Report file is empty: {report_path}")

        # Bulk insert AngleFrame records
        shot_angles = pipeline_data.get("shot_angles", [])
        ball_left_frames = pipeline_data.get("ball_left_frames", [])
        
        print(f"DEBUG: shot_angles count: {len(shot_angles)}, ball_left_frames count: {len(ball_left_frames)}")
        
        if shot_angles and ball_left_frames:
            angles_per_shot = []
            for idx, (frame_num, (elbow_angle, shoulder_angle)) in enumerate(zip(ball_left_frames, shot_angles)):
                angles_per_shot.append({
                    "session_id": session_id,
                    "frame_number": int(frame_num),
                    "elbow_angle": float(elbow_angle) if elbow_angle is not None else None,
                    "knee_angle": None,
                    "shoulder_angle": float(shoulder_angle) if shoulder_angle is not None else None,
                })
            
            if angles_per_shot:
                db.bulk_insert_mappings(AngleFrame, angles_per_shot)
                db.commit()
                print(f"✓ Inserted {len(angles_per_shot)} AngleFrame records for session {session_id}")
        else:
            print(f"WARNING: No shot_angles or ball_left_frames to insert")

        session.output_path = output_path
        session.report_path = report_path
        session.status = "completed"
        session.completed_at = datetime.now(timezone.utc)

        db.commit()

    except Exception:
        print("ERROR: Exception during video processing:")
        traceback.print_exc()
        if session is not None:
            session.status = "failed"
            db.commit()

        if 'input_path' in locals() and os.path.exists(input_path):
            print(input_path)
            os.remove(input_path)

    finally:
        db.close()
        _cleanup_transient_files()

