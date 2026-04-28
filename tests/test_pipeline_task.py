import sys
import types

from app.models.session import SessionModel
from app.models.user import User
from app.tasks.pipeline_task import process_video


class TestPipelineTask:
    def test_process_video_missing_session_exits_cleanly(self):
        # Should not raise when session_id does not exist.
        process_video.run(999999)

    def test_process_video_sets_failed_on_pipeline_exception(self, db_session):
        user = User(name="Pipeline Tester", email="pipeline@example.com", password_hash="hash")
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        session = SessionModel(
            user_id=user.id,
            original_filename="sample.mp4",
            upload_path="nonexistent_input_video.mp4",
            status="queued",
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        process_video.run(session.id)

        db_session.expire_all()
        refreshed = db_session.query(SessionModel).filter(SessionModel.id == session.id).first()
        assert refreshed is not None
        assert refreshed.status == "failed"

    def test_process_video_failure_does_not_delete_original_input_video(self, db_session, monkeypatch, tmp_path):
        user = User(name="Input Preserve", email="input-preserve@example.com", password_hash="hash")
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        monkeypatch.chdir(tmp_path)
        input_dir = tmp_path / "input_videos"
        input_dir.mkdir(parents=True, exist_ok=True)
        source_video = input_dir / "sample.mp4"
        source_video.write_bytes(b"video-bytes")

        session = SessionModel(
            user_id=user.id,
            original_filename="sample.mp4",
            upload_path=str(source_video),
            status="queued",
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        fake_main = types.ModuleType("main")

        def _raise_pipeline_error(_input_path, session_id):
            raise RuntimeError(f"forced failure for session {session_id}")

        fake_main.run_pipeline = _raise_pipeline_error
        monkeypatch.setitem(sys.modules, "main", fake_main)

        process_video.run(session.id)

        db_session.expire_all()
        refreshed = db_session.query(SessionModel).filter(SessionModel.id == session.id).first()
        assert refreshed is not None
        assert refreshed.status == "failed"
        assert source_video.exists()
