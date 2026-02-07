from sqlalchemy import Column, String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from app.database import Base


class Chat(Base):
    __tablename__ = "Chat"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    title = Column(Text, nullable=False)
    userId = Column(UUID(as_uuid=True), ForeignKey("User.id"), nullable=False)
    visibility = Column(String(10), nullable=False, default="private")  # enum: public, private
    lastContext = Column(JSONB, nullable=True)

    # Relationships
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")
    votes = relationship("Vote", back_populates="chat", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Chat(id={self.id}, title={self.title}, userId={self.userId})>"


class Message(Base):
    __tablename__ = "Message_v2"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chatId = Column(UUID(as_uuid=True), ForeignKey("Chat.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    parts = Column(JSON, nullable=False)  # Array of message parts
    attachments = Column(JSON, nullable=False)  # Array of attachments
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    chat = relationship("Chat", back_populates="messages")
    votes = relationship("Vote", back_populates="message", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Message(id={self.id}, chatId={self.chatId}, role={self.role})>"


class Vote(Base):
    __tablename__ = "Vote_v2"

    chatId = Column(UUID(as_uuid=True), ForeignKey("Chat.id"), primary_key=True, nullable=False)
    messageId = Column(UUID(as_uuid=True), ForeignKey("Message_v2.id"), primary_key=True, nullable=False)
    isUpvoted = Column(String(10), nullable=False)  # Store as boolean

    # Relationships
    chat = relationship("Chat", back_populates="votes")
    message = relationship("Message", back_populates="votes")

    def __repr__(self):
        return f"<Vote(chatId={self.chatId}, messageId={self.messageId}, isUpvoted={self.isUpvoted})>"
