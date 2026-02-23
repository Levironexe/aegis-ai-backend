import secrets
import base64
from typing import Optional
from urllib.parse import urlencode
import httpx
from jose import jwt
from app.config import settings
from app.schemas.auth import GoogleTokenResponse


class GoogleAuthService:
    """Service for handling Google OAuth 2.0 authentication"""

    AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    SCOPES = [
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ]

    @staticmethod
    def generate_state() -> str:
        """Generate a cryptographically secure random state token"""
        random_bytes = secrets.token_bytes(32)
        return base64.urlsafe_b64encode(random_bytes).decode('utf-8')

    @classmethod
    def get_authorization_url(cls, state: str) -> str:
        """
        Construct Google OAuth authorization URL

        Args:
            state: CSRF token for security

        Returns:
            Full authorization URL to redirect user to
        """
        params = {
            "client_id": settings.google_client_id,
            "redirect_uri": settings.google_redirect_uri,
            "response_type": "code",
            "scope": " ".join(cls.SCOPES),
            "state": state,
            "access_type": "offline",  # Get refresh token
            "prompt": "consent",  # Force consent to ensure refresh token
        }
        return f"{cls.AUTHORIZATION_URL}?{urlencode(params)}"

    @classmethod
    async def exchange_code_for_token(cls, code: str) -> GoogleTokenResponse:
        """
        Exchange authorization code for access and refresh tokens

        Args:
            code: Authorization code from Google OAuth callback

        Returns:
            GoogleTokenResponse with accessToken, refreshToken, etc.

        Raises:
            httpx.HTTPError: If token exchange fails
        """
        data = {
            "code": code,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uri": settings.google_redirect_uri,
            "grant_type": "authorization_code",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(cls.TOKEN_URL, data=data)

            if response.status_code != 200:
                error_body = response.text
                raise Exception(f"Google token exchange failed ({response.status_code}): {error_body}")

            token_data = response.json()

        return GoogleTokenResponse(**token_data)

    @classmethod
    async def refresh_accessToken(cls, refreshToken: str) -> GoogleTokenResponse:
        """
        Refresh an expired access token using refresh token

        Args:
            refreshToken: The refresh token to use

        Returns:
            GoogleTokenResponse with new accessToken

        Raises:
            httpx.HTTPError: If refresh fails
        """
        data = {
            "refreshToken": refreshToken,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "grant_type": "refreshToken",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(cls.TOKEN_URL, data=data)
            response.raise_for_status()
            token_data = response.json()

        return GoogleTokenResponse(**token_data)

    @staticmethod
    def decode_id_token(id_token: str) -> dict:
        """
        Decode Google JWT id_token to extract user information

        Args:
            id_token: JWT token from Google

        Returns:
            Dict with user info (email, sub/googleId, name, picture)

        Note:
            This does NOT verify the token signature. For production,
            you should verify using Google's public keys.
        """
        # Decode without verification (for simplicity, similar to C# version)
        # In production, use jwt.decode() with proper verification
        decoded = jwt.get_unverified_claims(id_token)
        return decoded
