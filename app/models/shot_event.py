from sqlalchemy import Column, Integer, String, Float, ForeignKey
from app.database import Base


class ShotEvent(Base):
    __tablename__ = "shot_events"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)
    shot_number = Column(Integer, nullable=False)
    result = Column(String(10), nullable=False)  # "make" or "miss"
    start_frame = Column(Integer, nullable=True)
    end_frame = Column(Integer, nullable=True)
    elbow_angle = Column(Float, nullable=True)
    shoulder_angle = Column(Float, nullable=True)
