import io

from fastapi.testclient import TestClient

from app.main import app
from app.api import sessions as sessions_api
from app.config import settings
from app.core.security import create_access_token
from app.models.user import User


def get_auth_token(db_session):
    user = User(name="Session Tester", email="sess@example.com", password_hash="hash")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return create_access_token({"sub": str(user.id)})


class TestSessions:
    def test_list_sessions_unauthenticated(self, client):
        res = client.get("/api/sessions/")
        assert res.status_code in (401, 403)

    def test_list_sessions_empty(self, client, db_session):
        token = get_auth_token(db_session)
        res = client.get("/api/sessions/", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        assert res.json() == []

    def test_get_nonexistent_session(self, client, db_session):
        token = get_auth_token(db_session)
        res = client.get("/api/sessions/9999", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 404

    def test_uploads_not_served_as_static_route(self, client):
        res = client.get("/uploads/any_file.mp4")
        assert res.status_code == 404

    def test_lifespan_creates_upload_and_output_directories(self, monkeypatch, tmp_path):
        upload_dir = tmp_path / "uploads"
        output_dir = tmp_path / "output_videos"
        monkeypatch.setattr(settings, "UPLOAD_DIR", str(upload_dir))
        monkeypatch.setattr(settings, "OUTPUT_DIR", str(output_dir))

        assert not upload_dir.exists()
        assert not output_dir.exists()

        with TestClient(app) as local_client:
            res = local_client.get("/api/health")
            assert res.status_code == 200

        assert upload_dir.exists()
        assert output_dir.exists()

    def test_upload_rejects_file_over_max_size_and_deletes_partial(self, client, db_session, monkeypatch, tmp_path):
        token = get_auth_token(db_session)
        monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
        monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 1)

        class DelaySpy:
            def delay(self, _session_id):
                raise AssertionError("Celery dispatch should not run for oversized uploads")

        monkeypatch.setattr(sessions_api, "process_video", DelaySpy())

        oversized_bytes = b"0" * ((1 * 1024 * 1024) + 1)
        res = client.post(
            "/api/sessions/upload",
            files={"file": ("big.mp4", io.BytesIO(oversized_bytes), "video/mp4")},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert res.status_code == 413
        assert "exceeds" in res.json()["detail"].lower()
        assert list(tmp_path.iterdir()) == []

    def test_upload_valid_file_dispatches_celery_and_queues_session(self, client, db_session, monkeypatch, tmp_path):
        token = get_auth_token(db_session)
        monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))

        class DelaySpy:
            called = False
            called_with = None

            def delay(self, session_id):
                self.called = True
                self.called_with = session_id

        spy = DelaySpy()
        monkeypatch.setattr(sessions_api, "process_video", spy)

        res = client.post(
            "/api/sessions/upload",
            files={"file": ("clip.mp4", io.BytesIO(b"small-video"), "video/mp4")},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert res.status_code == 201
        body = res.json()
        assert body["status"] == "queued"
        assert spy.called is True
        assert spy.called_with == body["id"]

        expected_path = tmp_path / f"{body['id']}.mp4"
        assert expected_path.exists()
