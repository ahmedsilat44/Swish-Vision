import uuid
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.config import settings
from app.database import get_db
from app.models.session import SessionModel
from app.models.user import User
from app.models.revoked_token import RevokedToken

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security_scheme = HTTPBearer()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_minutes: int = 60) -> str:
    payload = data.copy()
    payload["jti"] = str(uuid.uuid4())
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


def _user_from_token(token: str, db: Session) -> User:
    """Decode JWT, enforce revocation checks, and return the authenticated user."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        jti = payload.get("jti")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = int(user_id)
    except (JWTError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid token")

    if jti is not None and db.query(RevokedToken).filter(RevokedToken.jti == jti).first():
        raise HTTPException(status_code=401, detail="Token has been revoked")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    db: Session = Depends(get_db),
):
    return _user_from_token(credentials.credentials, db)

def get_current_user_token(token: str, db: Session) -> User:
    """Decode a raw JWT string (e.g. from a query parameter) and return the user."""
    return _user_from_token(token, db)


def get_session_or_403(session_id: int, current_user: User, db: Session):
    """Return the session if it exists and belongs to the current user."""
    session = db.query(SessionModel).filter(
        SessionModel.id == session_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    return session