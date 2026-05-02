"""add elbow_angle and shoulder_angle to shot_events

Revision ID: 20260428_add_shot_event_angles
Revises: 20260425_add_error_message
Create Date: 2026-04-28 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260428_add_shot_event_angles"
down_revision = "20260425_add_error_message"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_cols = [c["name"] for c in inspector.get_columns("shot_events")]

    if "elbow_angle" not in existing_cols:
        op.add_column("shot_events", sa.Column("elbow_angle", sa.Float(), nullable=True))

    if "shoulder_angle" not in existing_cols:
        op.add_column("shot_events", sa.Column("shoulder_angle", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("shot_events", "shoulder_angle")
    op.drop_column("shot_events", "elbow_angle")
