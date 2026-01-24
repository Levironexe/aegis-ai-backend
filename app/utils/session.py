from typing import Optional
from fastapi import Request, Response
from itsdangerous import URLSafeTimedSerializer, BadSignature
from app.config import settings


class SessionManager:
    """Simple session management using signed cookies"""

    def __init__(self):
        self.serializer = URLSafeTimedSerializer(settings.secret_key)
        self.session_cookie_name = "session"
        self.max_age = 86400  # 24 hours

    def set_session(self, response: Response, key: str, value: str) -> None:
        """Set a session value in a signed cookie"""
        session_data = {key: value}
        signed_data = self.serializer.dumps(session_data)
        response.set_cookie(
            key=self.session_cookie_name,
            value=signed_data,
            httponly=True,
            samesite="lax",
            secure=False,  # Set to True in production with HTTPS
            max_age=self.max_age,
        )

    def get_session(self, request: Request, key: str) -> Optional[str]:
        """Get a session value from signed cookie"""
        cookie = request.cookies.get(self.session_cookie_name)
        if not cookie:
            return None

        try:
            session_data = self.serializer.loads(cookie, max_age=self.max_age)
            return session_data.get(key)
        except BadSignature:
            return None

    def clear_session(self, response: Response) -> None:
        """Clear the session cookie"""
        response.delete_cookie(key=self.session_cookie_name)


session_manager = SessionManager()
