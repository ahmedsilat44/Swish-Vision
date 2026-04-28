from datetime import datetime, timezone
import os
import shutil
from celery import Celery
from celery.utils.log import get_task_logger

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

logger = get_task_logger(__name__)


def _normalize_shot_result(result) -> str:
    if isinstance(result, str):
        lowered = result.strip().lower()
        if lowered in {"make", "1", "true"}:
            return "make"
        if lowered in {"miss", "0", "false"}:
            return "miss"
    if isinstance(result, bool):
        return "make" if result else "miss"
    if isinstance(result, (int, float)):
        return "make" if int(result) == 1 else "miss"
    return "miss"


def _cleanup_transient_files():
    """Delete intermediate files written during pipeline inference.
    Safe to call on both success and failure paths.
    Never deletes uploaded videos or output files."""
    transient = ["angs.txt", "xy_coords.txt", "detections.txt", "ball_locl.txt"]
    for fname in transient:
        if os.path.exists(fname):
            try:
                os.remove(fname)
                logger.debug("Cleaned up transient file: %s", fname)
            except OSError as e:
                logger.warning("Could not delete transient file %s: %s", fname, e)


@celery_app.task(bind=True, max_retries=0)
def process_video(self, session_id: int):
    """Celery task wrapping the CV pipeline for a given session."""
    from app.database import SessionLocal
    from app.models.session import SessionModel
    from app.models.angle_frame import AngleFrame
    from app.models.shot_event import ShotEvent
    from app.models.report import Report

    db = SessionLocal()
    session = None
    try:
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if not session:
            logger.warning("Session %s not found; skipping pipeline task", session_id)
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
        logger.info("Pipeline produced output_path=%s report_path=%s", output_path, report_path)
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output video missing: {output_path}")
        if os.path.getsize(output_path) == 0:
            raise ValueError(f"Output video is empty: {output_path}")
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Report file missing: {report_path}")
        if os.path.getsize(report_path) == 0:
            raise ValueError(f"Report file is empty: {report_path}")

        # Extract pipeline data
        shot_angles = pipeline_data.get("shot_angles", [])
        shot_starts = pipeline_data.get("shot_strt", [])
        shot_ends = pipeline_data.get("shot_end", [])
        order_shots = pipeline_data.get("order_shots", [])
        total_shots = pipeline_data.get("total_shots", 0)
        made_shots = pipeline_data.get("made_shots", 0)
        missed_shots = pipeline_data.get("missed_shots", 0)

        logger.debug(
            "Counts for session %s: shot_angles=%s shot_starts=%s shot_ends=%s order_shots=%s",
            session_id,
            len(shot_angles),
            len(shot_starts),
            len(shot_ends),
            len(order_shots),
        )
        logger.debug(
            "Summary for session %s: total_shots=%s made_shots=%s missed_shots=%s",
            session_id,
            total_shots,
            made_shots,
            missed_shots,
        )
        
        # Bulk insert AngleFrame records
        
        if shot_angles and shot_starts:
            if len(shot_starts) != len(shot_angles):
                logger.warning(
                    "AngleFrame length mismatch for session %s: shot_starts=%s shot_angles=%s. Using shortest length.",
                    session_id,
                    len(shot_starts),
                    len(shot_angles),
                )

            angles_per_shot = []
            for frame_num, (elbow_angle, shoulder_angle) in zip(shot_starts, shot_angles):
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
                logger.info("Inserted %s AngleFrame records for session %s", len(angles_per_shot), session_id)
        else:
            logger.warning("No shot_angles or shot_starts available for AngleFrame insert (session %s)", session_id)

        # Insert Report record
        report_text = ""
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report_text = f.read()
            except Exception as e:
                logger.warning("Could not read report file %s: %s", report_path, e)
        
        report = Report(
            session_id=session_id,
            raw_text=report_text,
            total_shots=total_shots,
            makes=made_shots,
            misses=missed_shots,
        )
        db.add(report)
        db.commit()
        logger.info("Inserted Report record for session %s", session_id)

        # Bulk insert ShotEvent records
        shot_events_list = []
        
        if shot_starts and shot_ends and shot_angles and order_shots:
            if not (len(shot_starts) == len(shot_ends) == len(shot_angles) == len(order_shots)):
                logger.warning(
                    "ShotEvent length mismatch for session %s: shot_starts=%s shot_ends=%s shot_angles=%s order_shots=%s. Using shortest length.",
                    session_id,
                    len(shot_starts),
                    len(shot_ends),
                    len(shot_angles),
                    len(order_shots),
                )

            for shot_num, (start_frame, release_frame, (elbow_angle, shoulder_angle), result) in enumerate(
                zip(shot_starts, shot_ends, shot_angles, order_shots), 1):

                
                shot_events_list.append({
                    "session_id": session_id,
                    "shot_number": shot_num,
                    "result": _normalize_shot_result(result),
                    "start_frame": int(start_frame),
                    "end_frame": int(release_frame),
                    "elbow_angle": float(elbow_angle) if isinstance(elbow_angle, (int, float)) else None,
                    "shoulder_angle": float(shoulder_angle) if shoulder_angle is not None else None,
                })
            
            if shot_events_list:
                db.bulk_insert_mappings(ShotEvent, shot_events_list)
                db.commit()
                logger.info("Inserted %s ShotEvent records for session %s", len(shot_events_list), session_id)
        else:
            logger.warning("Missing data for ShotEvent insert (session %s)", session_id)

        session.output_path = output_path
        session.report_path = report_path
        session.status = "completed"
        session.completed_at = datetime.now(timezone.utc)

        db.commit()

    except Exception:
        logger.exception("Exception during video processing for session %s", session_id)
        if session is not None:
            session.status = "failed"
            db.commit()

        if 'input_path' in locals() and os.path.exists(input_path):
            logger.debug("Removing copied input file after failure: %s", input_path)
            os.remove(input_path)

    finally:
        db.close()
        _cleanup_transient_files()

