from datetime import datetime, timedelta

from app.core.security import create_access_token
from app.models.report import Report
from app.models.session import SessionModel
from app.models.user import User


def _create_user_and_token(db_session, suffix: str = ""):
    user = User(name=f"Dashboard Tester{suffix}", email=f"dashboard{suffix}@example.com", password_hash="hash")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    token = create_access_token({"sub": str(user.id)})
    return user, token


class TestDashboard:
    def test_dashboard_summary_empty_data(self, client, db_session):
        _user, token = _create_user_and_token(db_session, "-empty")

        res = client.get("/api/dashboard/summary", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200

        body = res.json()
        assert body == {
            "total_sessions": 0,
            "completed_sessions": 0,
            "total_shots": 0,
            "total_makes": 0,
            "shot_percentage": None,
        }

    def test_dashboard_summary_completed_sessions_with_reports_and_percentage(self, client, db_session):
        user, token = _create_user_and_token(db_session, "-summary")

        completed_with_report = SessionModel(user_id=user.id, original_filename="s1.mp4", status="completed")
        completed_without_report = SessionModel(user_id=user.id, original_filename="s2.mp4", status="completed")
        processing = SessionModel(user_id=user.id, original_filename="s3.mp4", status="processing")
        db_session.add_all([completed_with_report, completed_without_report, processing])
        db_session.commit()
        db_session.refresh(completed_with_report)

        db_session.add(
            Report(
                session_id=completed_with_report.id,
                total_shots=3,
                makes=2,
                misses=1,
            )
        )
        db_session.commit()

        res = client.get("/api/dashboard/summary", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200

        body = res.json()
        assert body["total_sessions"] == 3
        assert body["completed_sessions"] == 2
        assert body["total_shots"] == 3
        assert body["total_makes"] == 2
        assert body["shot_percentage"] == 66.7

    def test_dashboard_trends_empty_data(self, client, db_session):
        _user, token = _create_user_and_token(db_session, "-trends-empty")

        res = client.get("/api/dashboard/trends", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        assert res.json() == []

    def test_dashboard_trends_orders_by_session_created_at(self, client, db_session):
        user, token = _create_user_and_token(db_session, "-trends")
        base = datetime(2026, 1, 1, 12, 0, 0)

        latest = SessionModel(
            user_id=user.id,
            original_filename="late.mp4",
            status="completed",
            created_at=base + timedelta(days=2),
        )
        earliest = SessionModel(
            user_id=user.id,
            original_filename="early.mp4",
            status="completed",
            created_at=base,
        )
        middle = SessionModel(
            user_id=user.id,
            original_filename="mid.mp4",
            status="completed",
            created_at=base + timedelta(days=1),
        )
        db_session.add_all([latest, earliest, middle])
        db_session.commit()
        db_session.refresh(latest)
        db_session.refresh(earliest)
        db_session.refresh(middle)

        db_session.add_all(
            [
                Report(session_id=latest.id, total_shots=7, makes=4, misses=3),
                Report(session_id=earliest.id, total_shots=5, makes=3, misses=2),
                Report(session_id=middle.id, total_shots=6, makes=4, misses=2),
            ]
        )
        db_session.commit()

        res = client.get("/api/dashboard/trends", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        body = res.json()

        assert [p["session_id"] for p in body] == [earliest.id, middle.id, latest.id]
        assert [p["total_shots"] for p in body] == [5, 6, 7]
        assert [p["makes"] for p in body] == [3, 4, 4]
        assert [p["misses"] for p in body] == [2, 2, 3]