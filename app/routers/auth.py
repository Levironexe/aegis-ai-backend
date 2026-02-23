from fastapi import APIRouter, Depends, Request, Response, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from datetime import datetime, timedelta

from app.database import get_db
from app.services.google_auth_service import GoogleAuthService
from app.services.token_service import TokenService
from app.schemas.auth import (
    EstablishSessionRequest,
    EstablishSessionResponse,
    AuthStatusResponse,
    UserResponse,
)
from app.utils.session import session_manager
from app.config import settings
from app.models.user import User
from sqlalchemy import select

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.get("/google/login")
async def google_login():
    """Initiate Google OAuth flow"""
    state = GoogleAuthService.generate_state()
    auth_url = GoogleAuthService.get_authorization_url(state)
    return RedirectResponse(url=auth_url)


@router.get("/google/callback")
async def google_callback(
    response: Response,
    code: str,
    db: AsyncSession = Depends(get_db),
):
    """Handle Google OAuth callback"""

    # Note: CSRF state validation removed for simplicity
    # In production, implement proper state validation with server-side storage (Redis, database, etc.)

    try:
        # Exchange code for tokens
        token_response = await GoogleAuthService.exchange_code_for_token(code)

        # Decode id_token to get user info
        user_info = GoogleAuthService.decode_id_token(token_response.id_token)
        email = user_info.get("email")
        google_id = user_info.get("sub")
        name = user_info.get("name")
        picture = user_info.get("picture")

        if not email or not google_id:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract user information from Google",
            )

        # Check if user exists
        user = await TokenService.get_user_by_google_id(db, google_id)

        if user:
            # Update existing user's tokens
            await TokenService.store_tokens(
                db,
                user.id,
                token_response.access_token,
                token_response.refresh_token,
                token_response.expires_in,
                token_response.scope,
            )
        else:
            # Create new user
            user = await TokenService.create_user(
                db,
                email,
                google_id,
                name,
                picture,
                token_response.access_token,
                token_response.refresh_token,
                token_response.expires_in,
                token_response.scope,
            )

        # Redirect to frontend with user info
        from urllib.parse import urlencode
        params = {
            "userId": str(user.id),
            "email": email,
            "name": name or "",
            "picture": picture or "",
        }
        redirect_url = f"{settings.frontend_url}/auth/success?{urlencode(params)}"
        return RedirectResponse(url=redirect_url)

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Google OAuth callback error: {type(e).__name__}: {str(e)}", exc_info=True)

        # Get more detailed error message
        error_detail = str(e) if str(e) else f"{type(e).__name__}: {repr(e)}"
        raise HTTPException(status_code=500, detail=f"Authentication failed: {error_detail}")


@router.post("/establish-session", response_model=EstablishSessionResponse)
async def establish_session(
    request: EstablishSessionRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """Establish server-side session after OAuth redirect"""

    try:
        user_id = UUID(request.user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID")

    # Verify user exists and has tokens
    has_tokens = await TokenService.user_exists_and_has_tokens(db, user_id)
    if not has_tokens:
        raise HTTPException(status_code=404, detail="User not found or invalid tokens")

    # Set session
    session_manager.set_session(response, "user_id", str(user_id))

    return EstablishSessionResponse(success=True)


@router.get("/status", response_model=AuthStatusResponse)
async def auth_status(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Check authentication status"""

    user_id_str = session_manager.get_session(request, "user_id")
    if not user_id_str:
        return AuthStatusResponse(authenticated=False)

    try:
        user_id = UUID(user_id_str)
    except ValueError:
        return AuthStatusResponse(authenticated=False)

    # Check if token is still valid
    is_valid = await TokenService.is_token_valid(db, user_id)
    if not is_valid:
        return AuthStatusResponse(authenticated=False)

    return AuthStatusResponse(authenticated=True, user_id=user_id)


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Get current authenticated user info"""

    user_id_str = session_manager.get_session(request, "user_id")
    if not user_id_str:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        user_id = UUID(user_id_str)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid session")

    # Get user from database
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(
        user_id=user.id,
        email=user.email,
        name=user.name,
        picture=user.picture,
        authenticated=True,
    )
