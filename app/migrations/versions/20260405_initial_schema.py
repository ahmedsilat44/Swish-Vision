"""initial schema

Revision ID: 20260405_initial_schema
Revises: 
Create Date: 2026-04-05 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260405_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    op.create_table(
        "sessions",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("original_filename", sa.String(length=500), nullable=False),
        sa.Column("upload_path", sa.String(length=500), nullable=True),
        sa.Column("output_path", sa.String(length=500), nullable=True),
        sa.Column("report_path", sa.String(length=500), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=True, server_default=sa.text("'uploading'")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
    )
    op.create_index(op.f("ix_sessions_user_id"), "sessions", ["user_id"], unique=False)

    op.create_table(
        "reports",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("total_shots", sa.Integer(), nullable=True, server_default=sa.text("0")),
        sa.Column("makes", sa.Integer(), nullable=True, server_default=sa.text("0")),
        sa.Column("misses", sa.Integer(), nullable=True, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"]),
    )
    op.create_index(op.f("ix_reports_session_id"), "reports", ["session_id"], unique=False)

    op.create_table(
        "shot_events",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("shot_number", sa.Integer(), nullable=False),
        sa.Column("result", sa.String(length=10), nullable=False),
        sa.Column("start_frame", sa.Integer(), nullable=True),
        sa.Column("end_frame", sa.Integer(), nullable=True),
        sa.Column("release_angle", sa.Float(), nullable=True),
        sa.Column("elbow_angle_at_release", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"]),
    )
    op.create_index(op.f("ix_shot_events_session_id"), "shot_events", ["session_id"], unique=False)

    op.create_table(
        "angle_frames",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("frame_number", sa.Integer(), nullable=False),
        sa.Column("elbow_angle", sa.Float(), nullable=True),
        sa.Column("knee_angle", sa.Float(), nullable=True),
        sa.Column("shoulder_angle", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"]),
    )
    op.create_index(op.f("ix_angle_frames_session_id"), "angle_frames", ["session_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_angle_frames_session_id"), table_name="angle_frames")
    op.drop_table("angle_frames")

    op.drop_index(op.f("ix_shot_events_session_id"), table_name="shot_events")
    op.drop_table("shot_events")

    op.drop_index(op.f("ix_reports_session_id"), table_name="reports")
    op.drop_table("reports")

    op.drop_index(op.f("ix_sessions_user_id"), table_name="sessions")
    op.drop_table("sessions")

    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
