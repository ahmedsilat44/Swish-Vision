from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from app.database import Base


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)
    raw_text = Column(Text, nullable=True)
    total_shots = Column(Integer, default=0)
    makes = Column(Integer, default=0)
    misses = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
