from pydantic import BaseModel, UUID4
from typing import Optional


class GoogleTokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: int
    scope: str
    token_type: str
    id_token: str


class EstablishSessionRequest(BaseModel):
    user_id: str


class EstablishSessionResponse(BaseModel):
    success: bool


class AuthStatusResponse(BaseModel):
    authenticated: bool
    user_id: Optional[UUID4] = None


class UserResponse(BaseModel):
    user_id: UUID4
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None
    authenticated: bool = True

    class Config:
        from_attributes = True
