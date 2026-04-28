from pathlib import Path

from app.models.report import Report
from app.models.shot_event import ShotEvent
from app.models.session import SessionModel
from app.models.user import User
from app.tasks.report_parser import parse_and_persist


def _make_session(db_session, email="parser@example.com"):
    user = User(name="Parser User", email=email, password_hash="hash")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    session = SessionModel(
        user_id=user.id,
        original_filename="clip.mp4",
        upload_path="uploads/clip.mp4",
        status="completed",
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    return session


class TestReportParser:
    def test_parse_and_persist_with_shot_lines(self, db_session, tmp_path):
        session = _make_session(db_session, email="shot-lines@example.com")
        report_path = Path(tmp_path) / "sample_report.txt"
        report_path.write_text(
            "Shot 1: made, release=48.3, elbow=92.1\n"
            "Shot 2: missed, release=44.0, elbow=88.4\n",
            encoding="utf-8",
        )

        parse_and_persist(session.id, str(report_path), db_session)

        report = db_session.query(Report).filter(Report.session_id == session.id).one()
        assert report.total_shots == 2
        assert report.makes == 1
        assert report.misses == 1
        assert "Shot 1: made" in report.raw_text

        shot_events = (
            db_session.query(ShotEvent)
            .filter(ShotEvent.session_id == session.id)
            .order_by(ShotEvent.shot_number)
            .all()
        )
        assert len(shot_events) == 2
        assert shot_events[0].result == "make"
        assert shot_events[1].result == "miss"
        assert shot_events[0].shoulder_angle == 48.3
        assert shot_events[0].elbow_angle == 92.1

    def test_parse_and_persist_with_analysis_blocks(self, db_session, tmp_path):
        session = _make_session(db_session, email="analysis-blocks@example.com")
        report_path = Path(tmp_path) / "analysis_report.txt"
        report_path.write_text(
            "GOOD FORM\n\n"
            "shot 2\n"
            "NEEDS WORK:\n"
            "Your SEW angle is too low\n",
            encoding="utf-8",
        )

        parse_and_persist(session.id, str(report_path), db_session)

        report = db_session.query(Report).filter(Report.session_id == session.id).one()
        assert report.total_shots == 2
        assert report.makes == 1
        assert report.misses == 1

        shot_events = (
            db_session.query(ShotEvent)
            .filter(ShotEvent.session_id == session.id)
            .order_by(ShotEvent.shot_number)
            .all()
        )
        assert len(shot_events) == 2
        assert shot_events[0].shot_number == 1
        assert shot_events[1].shot_number == 2
        assert shot_events[0].result == "make"
        assert shot_events[1].result == "miss"

    def test_parse_and_persist_zero_shot_report_still_inserts_report(self, db_session, tmp_path):
        session = _make_session(db_session, email="zero-shot@example.com")
        report_path = Path(tmp_path) / "empty_report.txt"
        report_path.write_text("", encoding="utf-8")

        parse_and_persist(session.id, str(report_path), db_session)

        report = db_session.query(Report).filter(Report.session_id == session.id).one()
        assert report.total_shots == 0
        assert report.makes == 0
        assert report.misses == 0

        shot_event_count = db_session.query(ShotEvent).filter(ShotEvent.session_id == session.id).count()
        assert shot_event_count == 0
