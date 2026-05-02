from datetime import datetime, timezone
import os
import shutil
import traceback
from sqlalchemy import inspect, text
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


def _get_table_columns(db, table_name: str) -> set[str]:
    bind = db.get_bind()
    return {column["name"] for column in inspect(bind).get_columns(table_name)}


def _normalize_shot_result(result) -> str:
    if isinstance(result, str):
        lowered = result.strip().lower()
        if lowered in {"make", "made", "1", "true"}:
            return "make"
        if lowered in {"miss", "missed", "0", "false"}:
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


@celery_app.task(bind=True, max_retries=2, autoretry_for=(Exception,), retry_backoff=True)
def process_video(self, session_id: int):
    """Celery task wrapping the CV pipeline for a given session."""
    from app.database import SessionLocal
    from app.models.session import SessionModel
    from app.models.angle_frame import AngleFrame
    from app.tasks.report_parser import parse_and_persist

    db = SessionLocal()
    session = None
    input_path = None
    copied_input_file = False
    try:
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if not session:
            logger.warning("Session %s not found; skipping pipeline task", session_id)
            return

        session.status = "processing"
        db.commit()

        source_upload_path = os.path.abspath(session.upload_path or "")
        if not source_upload_path or not os.path.exists(source_upload_path):
            raise FileNotFoundError(f"Upload file missing: {session.upload_path}")

        # Copy uploaded video into input_videos only when source is outside input_videos.
        # This avoids deleting the original upload when handling failures.
        input_dir = os.path.abspath("input_videos")
        os.makedirs(input_dir, exist_ok=True)
        source_dir = os.path.dirname(source_upload_path)

        if os.path.normcase(source_dir) == os.path.normcase(input_dir):
            input_path = source_upload_path
            logger.debug("Using upload in input_videos directly for session %s: %s", session_id, input_path)
        else:
            input_filename = f"session_{session_id}_{os.path.basename(source_upload_path)}"
            input_path = os.path.join(input_dir, input_filename)
            shutil.copy2(source_upload_path, input_path)
            copied_input_file = True
            logger.debug("Copied upload for processing for session %s: %s -> %s", session_id, source_upload_path, input_path)

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
        logger.debug(
            "Counts for session %s: shot_angles=%s shot_starts=%s shot_ends=%s order_shots=%s",
            session_id,
            len(shot_angles),
            len(shot_starts),
            len(shot_ends),
            len(order_shots),
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

        parse_and_persist(session_id=session_id, report_path=report_path, db=db)

        # Keep parser as source-of-truth for rows, then enrich with pipeline-only fields.
        shot_event_columns = _get_table_columns(db, "shot_events")
        if not (len(shot_starts) == len(shot_ends) == len(shot_angles) == len(order_shots)):
            logger.warning(
                "ShotEvent enrichment length mismatch for session %s: shot_starts=%s shot_ends=%s shot_angles=%s order_shots=%s. Using shortest length.",
                session_id,
                len(shot_starts),
                len(shot_ends),
                len(shot_angles),
                len(order_shots),
            )

        updates_applied = 0
        for shot_num, start_frame, release_frame, angles, result in zip(
            range(1, min(len(shot_starts), len(shot_ends), len(shot_angles), len(order_shots)) + 1),
            shot_starts,
            shot_ends,
            shot_angles,
            order_shots,
        ):
            elbow_angle, shoulder_angle = angles
            set_clauses = []
            params = {
                "session_id": session_id,
                "shot_number": shot_num,
            }

            if "result" in shot_event_columns:
                set_clauses.append("result = :result")
                params["result"] = _normalize_shot_result(result)
            if "start_frame" in shot_event_columns:
                set_clauses.append("start_frame = :start_frame")
                params["start_frame"] = int(start_frame)
            if "end_frame" in shot_event_columns:
                set_clauses.append("end_frame = :end_frame")
                params["end_frame"] = int(release_frame)
            if "elbow_angle" in shot_event_columns:
                set_clauses.append("elbow_angle = :elbow_angle")
                params["elbow_angle"] = float(elbow_angle) if isinstance(elbow_angle, (int, float)) else None
            if "shoulder_angle" in shot_event_columns:
                set_clauses.append("shoulder_angle = :shoulder_angle")
                params["shoulder_angle"] = float(shoulder_angle) if shoulder_angle is not None else None

            if not set_clauses:
                continue

            db.execute(
                text(
                    f"UPDATE shot_events SET {', '.join(set_clauses)} "
                    "WHERE session_id = :session_id AND shot_number = :shot_number"
                ),
                params,
            )
            updates_applied += 1

        db.commit()
        logger.info("Enriched %s ShotEvent records for session %s", updates_applied, session_id)

        session.output_path = output_path
        session.report_path = report_path
        session.status = "completed"
        session.completed_at = datetime.now(timezone.utc)

        db.commit()


    except Exception as exc:
        print("ERROR: Exception during video processing:")
        traceback.print_exc()
        logger.exception("Exception during video processing for session %s", session_id)
        db.rollback()
        if session is not None:
            session.status = "failed"
            session.error_message = str(exc) or "An unexpected error occurred during processing."
            db.commit()

        if copied_input_file and input_path and os.path.exists(input_path):
            logger.debug("Removing copied input file after failure: %s", input_path)
            os.remove(input_path)

    finally:
        db.close()
        _cleanup_transient_files()

