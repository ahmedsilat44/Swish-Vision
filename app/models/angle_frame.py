from sqlalchemy import Column, Integer, Float, ForeignKey
from app.database import Base


class AngleFrame(Base):
    __tablename__ = "angle_frames"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    elbow_angle = Column(Float, nullable=True)
    knee_angle = Column(Float, nullable=True)
    shoulder_angle = Column(Float, nullable=True)
