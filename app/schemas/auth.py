from pydantic import BaseModel


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ResetPasswordRequest(BaseModel):
    email: str
    current_password: str
    new_password: str


class AdminForceResetPasswordRequest(BaseModel):
    email: str
    new_password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
