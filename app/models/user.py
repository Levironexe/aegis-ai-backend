from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from app.database import Base


class User(Base):
    __tablename__ = "User"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    email = Column(String(64), unique=True, nullable=False, index=True)
    password = Column(String(64), nullable=True)  # Not used with OAuth

    # Google OAuth fields
    googleId = Column(String(255), unique=True, nullable=True, index=True)
    name = Column(String(255), nullable=True)
    picture = Column(Text, nullable=True)

    # OAuth tokens (stored server-side)
    accessToken = Column(Text, nullable=True)
    refreshToken = Column(Text, nullable=True)
    tokenExpiresAt = Column(DateTime, nullable=True)
    scopes = Column(Text, nullable=True)


    # Timestamps
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, googleId={self.googleId})>"
