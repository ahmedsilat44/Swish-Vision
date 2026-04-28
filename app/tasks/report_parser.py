import logging
import re
from sqlalchemy import inspect

from app.models.report import Report
from app.models.shot_event import ShotEvent


logger = logging.getLogger(__name__)


def parse_and_persist(session_id: int, report_path: str, db):
    """Parse HumanTracksDrawer report text and persist Report + ShotEvent rows.

    The parser supports both:
    - legacy free-form report lines (e.g., "Shot 1: made, release=48.3, elbow=92.1")
    - current HumanTracksDrawer.analysis() blocks ("GOOD FORM" / "shot N" + "NEEDS WORK")
    """
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    shot_events = _parse_shot_lines(content)
    if not shot_events:
        shot_events = _parse_analysis_blocks(content)

    makes = sum(1 for event in shot_events if event["result"] == "make")
    misses = sum(1 for event in shot_events if event["result"] == "miss")
    total_shots = makes + misses

    shot_event_columns = _get_table_columns(db, "shot_events")

    report = Report(
        session_id=session_id,
        raw_text=content,
        total_shots=total_shots,
        makes=makes,
        misses=misses,
    )

    try:
        db.add(report)
        db.flush()

        rows = []
        for event in shot_events:
            row = {
                "session_id": session_id,
                "shot_number": event["shot_number"],
                "result": event["result"],
            }
            if "start_frame" in shot_event_columns and event.get("start_frame") is not None:
                row["start_frame"] = event.get("start_frame")
            if "end_frame" in shot_event_columns and event.get("end_frame") is not None:
                row["end_frame"] = event.get("end_frame")
            if "elbow_angle" in shot_event_columns and event.get("elbow_angle") is not None:
                row["elbow_angle"] = event.get("elbow_angle")
            if "shoulder_angle" in shot_event_columns and event.get("shoulder_angle") is not None:
                row["shoulder_angle"] = event.get("shoulder_angle")
            rows.append(row)

        if rows:
            db.bulk_insert_mappings(ShotEvent, rows, render_nulls=False)

        db.commit()
    except Exception:
        db.rollback()
        raise

    logger.info(
        "Session %s: report parsed, %s shots persisted (%s makes, %s misses)",
        session_id,
        total_shots,
        makes,
        misses,
    )


def _get_table_columns(db, table_name: str) -> set[str]:
    bind = db.get_bind()
    return {column["name"] for column in inspect(bind).get_columns(table_name)}


def _normalize_outcome(raw: str | None) -> str:
    normalized = (raw or "").strip().lower()
    if normalized in {"made", "make", "1", "true"}:
        return "make"
    if normalized in {"missed", "miss", "0", "false"}:
        return "miss"
    return "miss"


def _parse_float(line: str, pattern: str):
    match = re.search(pattern, line, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _parse_shot_lines(content: str):
    parsed = []
    for line in content.splitlines():
        match = re.search(r"Shot\s*(\d+)\s*[:\-]?\s*(made|missed|make|miss)\b", line, re.IGNORECASE)
        if not match:
            continue

        shot_number = int(match.group(1))
        result = _normalize_outcome(match.group(2))
        release_angle = _parse_float(line, r"(?:release|shoulder)(?:\s*angle)?\s*[=:]\s*([-+]?\d+(?:\.\d+)?)")
        elbow_angle = _parse_float(line, r"elbow(?:\s*angle)?\s*[=:]\s*([-+]?\d+(?:\.\d+)?)")

        parsed.append(
            {
                "shot_number": shot_number,
                "result": result,
                "elbow_angle": elbow_angle,
                "shoulder_angle": release_angle,
            }
        )

    return parsed


def _parse_analysis_blocks(content: str):
    parsed = []
    blocks = [block.strip() for block in re.split(r"\n\s*\n", content) if block.strip()]

    for index, block in enumerate(blocks, start=1):
        shot_number_match = re.search(r"\bshot\s*(\d+)\b", block, re.IGNORECASE)
        shot_number = int(shot_number_match.group(1)) if shot_number_match else index

        # Current HumanTracksDrawer format marks positive blocks as GOOD FORM.
        result = "make" if re.search(r"\bGOOD\s+FORM\b", block, re.IGNORECASE) else "miss"

        parsed.append(
            {
                "shot_number": shot_number,
                "result": result,
                "elbow_angle": None,
                "shoulder_angle": None,
            }
        )

    return parsed
