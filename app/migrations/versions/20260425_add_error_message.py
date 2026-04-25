"""add error_message to sessions

Revision ID: 20260425_add_error_message
Revises: 20260405_initial_schema
Create Date: 2026-04-25 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "20260425_add_error_message"
down_revision = "20260405_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("sessions", sa.Column("error_message", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("sessions", "error_message")
