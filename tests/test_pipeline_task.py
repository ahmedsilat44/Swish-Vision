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
