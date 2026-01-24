from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.services.google_auth_service import GoogleAuthService


class TokenService:
    """Service for managing user OAuth tokens"""

    TOKEN_EXPIRY_BUFFER_MINUTES = 5

    @staticmethod
    async def get_valid_accessToken(db: AsyncSession, user_id: UUID) -> Optional[str]:
        """
        Get a valid access token for user, refreshing if necessary

        Args:
            db: Database session
            user_id: User's UUID

        Returns:
            Valid access token or None if user not found

        Note:
            Automatically refreshes token if it expires within 5 minutes
        """
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user or not user.accessToken:
            return None

        # Check if token is expiring soon (within 5 minutes)
        if user.tokenExpiresAt:
            expiry_threshold = datetime.utcnow() + timedelta(
                minutes=TokenService.TOKEN_EXPIRY_BUFFER_MINUTES
            )
            if user.tokenExpiresAt <= expiry_threshold:
                # Token is expiring soon, refresh it
                if user.refreshToken:
                    try:
                        token_response = await GoogleAuthService.refresh_accessToken(
                            user.refreshToken
                        )

                        # Update user's tokens
                        user.accessToken = token_response.accessToken
                        user.tokenExpiresAt = datetime.utcnow() + timedelta(
                            seconds=token_response.expires_in
                        )

                        # Only update refreshToken if a new one is provided
                        if token_response.refreshToken:
                            user.refreshToken = token_response.refreshToken

                        if token_response.scope:
                            user.scopes = token_response.scope

                        await db.commit()
                        await db.refresh(user)

                    except Exception as e:
                        print(f"Failed to refresh token for user {user_id}: {e}")
                        return None

        return user.accessToken

    @staticmethod
    async def store_tokens(
        db: AsyncSession,
        user_id: UUID,
        accessToken: str,
        refreshToken: Optional[str],
        expires_in: int,
        scopes: Optional[str] = None,
    ) -> None:
        """
        Store or update OAuth tokens for a user

        Args:
            db: Database session
            user_id: User's UUID
            accessToken: OAuth access token
            refreshToken: OAuth refresh token (optional)
            expires_in: Token expiry in seconds
            scopes: Granted OAuth scopes
        """
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            return

        user.accessToken = accessToken
        user.tokenExpiresAt = datetime.utcnow() + timedelta(seconds=expires_in)

        # Only update refreshToken if provided
        if refreshToken:
            user.refreshToken = refreshToken

        if scopes:
            user.scopes = scopes

        await db.commit()

    @staticmethod
    async def is_token_valid(db: AsyncSession, user_id: UUID) -> bool:
        """
        Check if user's access token is still valid

        Args:
            db: Database session
            user_id: User's UUID

        Returns:
            True if token is valid (not expired), False otherwise
        """
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user or not user.accessToken or not user.tokenExpiresAt:
            return False

        return user.tokenExpiresAt > datetime.utcnow()

    @staticmethod
    async def revoke_tokens(db: AsyncSession, user_id: UUID) -> None:
        """
        Clear all OAuth tokens for a user (logout)

        Args:
            db: Database session
            user_id: User's UUID
        """
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            return

        user.accessToken = None
        user.refreshToken = None
        user.tokenExpiresAt = None
        user.scopes = None

        await db.commit()

    @staticmethod
    async def get_user_by_google_id(db: AsyncSession, googleId: str) -> Optional[User]:
        """
        Find user by Google ID

        Args:
            db: Database session
            googleId: Google user identifier

        Returns:
            User object or None if not found
        """
        result = await db.execute(select(User).where(User.googleId == googleId))
        return result.scalar_one_or_none()

    @staticmethod
    async def create_user(
        db: AsyncSession,
        email: str,
        googleId: str,
        name: Optional[str],
        picture: Optional[str],
        accessToken: str,
        refreshToken: Optional[str],
        expires_in: int,
        scopes: Optional[str],
    ) -> User:
        """
        Create a new user with OAuth tokens

        Args:
            db: Database session
            email: User's email
            googleId: Google user identifier
            name: User's display name
            picture: Profile picture URL
            accessToken: OAuth access token
            refreshToken: OAuth refresh token
            expires_in: Token expiry in seconds
            scopes: Granted OAuth scopes

        Returns:
            Newly created User object
        """
        user = User(
            email=email,
            googleId=googleId,
            name=name,
            picture=picture,
            accessToken=accessToken,
            refreshToken=refreshToken,
            tokenExpiresAt=datetime.utcnow() + timedelta(seconds=expires_in),
            scopes=scopes,
        )

        db.add(user)
        await db.commit()
        await db.refresh(user)

        return user

    @staticmethod
    async def user_exists_and_has_tokens(db: AsyncSession, user_id: UUID) -> bool:
        """
        Check if user exists and has valid OAuth tokens

        Args:
            db: Database session
            user_id: User's UUID

        Returns:
            True if user exists and has tokens, False otherwise
        """
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        return bool(
            user and user.accessToken and user.refreshToken and user.tokenExpiresAt
        )
