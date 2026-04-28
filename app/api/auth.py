from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func
from jose import jwt, JWTError
from datetime import datetime, timezone
from typing import Optional
import secrets
from app.database import get_db
from app.models.user import User
from app.models.revoked_token import RevokedToken
from app.schemas.auth import RegisterRequest, LoginRequest, ResetPasswordRequest, AdminForceResetPasswordRequest, TokenResponse
from app.core.security import hash_password, verify_password, create_access_token, get_current_user, security_scheme
from app.config import settings
import re

router = APIRouter(prefix="/api/auth", tags=["auth"])


def validate_password_strength(password: str) -> bool:
    return len(password) >= 8 and bool(re.search(r"[A-Za-z]", password)) and bool(re.search(r"\d", password))


def normalize_email(email: str) -> str:
    return email.strip().lower()


def require_admin_reset_key(
    x_admin_reset_key: Optional[str] = Header(default=None, alias="X-Admin-Reset-Key"),
):
    configured_key = (settings.ADMIN_RESET_KEY or "").strip()
    if not configured_key:
        raise HTTPException(status_code=503, detail="Admin reset endpoint is not configured")

    if not x_admin_reset_key or not secrets.compare_digest(x_admin_reset_key, configured_key):
        raise HTTPException(status_code=403, detail="Forbidden")


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    email = normalize_email(payload.email)
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise HTTPException(status_code=422, detail="Invalid email format")
    if db.query(User).filter(func.lower(User.email) == email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    if not validate_password_strength(payload.password):
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters and contain both letters and a number.")

    user = User(name=payload.name, email=email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = normalize_email(payload.email)
    user = db.query(User).filter(func.lower(User.email) == email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}


@router.post("/reset-password")
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    email = normalize_email(payload.email)
    user = db.query(User).filter(func.lower(User.email) == email).first()
    if not user or not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if payload.current_password == payload.new_password:
        raise HTTPException(status_code=422, detail="New password must be different from the current password")

    if not validate_password_strength(payload.new_password):
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters and contain both letters and a number.")

    user.password_hash = hash_password(payload.new_password)
    db.commit()

    return {"message": "Password updated successfully. Please sign in with your new password."}


@router.post("/admin/force-reset-password")
def admin_force_reset_password(
    payload: AdminForceResetPasswordRequest,
    _admin_ok: None = Depends(require_admin_reset_key),
    db: Session = Depends(get_db),
):
    email = normalize_email(payload.email)
    user = db.query(User).filter(func.lower(User.email) == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not validate_password_strength(payload.new_password):
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters and contain both letters and a number.")

    if verify_password(payload.new_password, user.password_hash):
        raise HTTPException(status_code=422, detail="New password must be different from current password")

    user.password_hash = hash_password(payload.new_password)
    db.commit()
    return {"message": "Password reset successfully"}


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
