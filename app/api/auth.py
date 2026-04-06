from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from jose import jwt, JWTError
from datetime import datetime, timezone
from app.database import get_db
from app.models.user import User
from app.models.revoked_token import RevokedToken
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from app.core.security import hash_password, verify_password, create_access_token, get_current_user, security_scheme
from app.config import settings
import re

router = APIRouter(prefix="/api", tags=["auth"])


def validate_password_strength(password: str) -> bool:
    return len(password) >= 8 and bool(re.search(r"[A-Za-z]", password)) and bool(re.search(r"\d", password))


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", payload.email):
        raise HTTPException(status_code=422, detail="Invalid email format")
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    if not validate_password_strength(payload.password):
        raise HTTPException(status_code=422, detail="Password too weak")

    user = User(name=payload.name, email=payload.email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}


@router.post("/logout")
def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    jti = payload.get("jti")
    exp_ts = payload.get("exp")
    if not isinstance(jti, str) or not jti:
        raise HTTPException(status_code=401, detail="Invalid token")
    if not isinstance(exp_ts, (int, float)):
        raise HTTPException(status_code=401, detail="Invalid token")

    expires_at = datetime.fromtimestamp(exp_ts, tz=timezone.utc)

    try:
        db.add(RevokedToken(jti=jti, expires_at=expires_at))
        db.commit()
    except IntegrityError:
        # Another concurrent request already revoked this token — still idempotent.
        db.rollback()
    return {"message": "Logged out successfully"}
