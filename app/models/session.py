from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, text
from sqlalchemy.sql import func
from app.database import Base


class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    original_filename = Column(String(500), nullable=False)
    upload_path = Column(String(500), nullable=True)
    output_path = Column(String(500), nullable=True)
    report_path = Column(String(500), nullable=True)
    status = Column(String(50), default="uploading", server_default=text("'uploading'"))  # uploading | queued | processing | completed | failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
