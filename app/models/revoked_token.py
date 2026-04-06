from sqlalchemy import Column, Integer, String, DateTime, Index
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from app.database import Base


class RevokedToken(Base):
    __tablename__ = "revoked_tokens"

    id = Column(Integer, primary_key=True, index=True)
    jti = Column(String(36), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)

    # Index supports fast range-deletes during cleanup.
    __table_args__ = (Index("ix_revoked_tokens_expires_at", "expires_at"),)

    @staticmethod
    def delete_expired(db: Session) -> int:
        """Delete all rows whose token has already expired. Returns the number of rows removed."""
        cutoff = datetime.now(tz=timezone.utc)
        deleted = (
            db.query(RevokedToken)
            .filter(RevokedToken.expires_at < cutoff)
            .delete(synchronize_session=False)
        )
        db.commit()
        return deleted
