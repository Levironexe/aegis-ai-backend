from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Boolean, ForeignKeyConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from app.database import Base


class Document(Base):
    __tablename__ = "Document"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    createdAt = Column(DateTime, primary_key=True, nullable=False, default=datetime.utcnow)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=True)
    kind = Column(String(10), nullable=False, default="text")  # enum: text, code, image, sheet
    userId = Column(UUID(as_uuid=True), ForeignKey("User.id"), nullable=False)

    def __repr__(self):
        return f"<Document(id={self.id}, title={self.title}, kind={self.kind})>"


class Suggestion(Base):
    __tablename__ = "Suggestion"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    documentId = Column(UUID(as_uuid=True), nullable=False)
    documentCreatedAt = Column(DateTime, nullable=False)
    originalText = Column(Text, nullable=False)
    suggestedText = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    isResolved = Column(Boolean, nullable=False, default=False)
    userId = Column(UUID(as_uuid=True), ForeignKey("User.id"), nullable=False)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Foreign key constraint for composite key
    __table_args__ = (
        ForeignKeyConstraint(
            ["documentId", "documentCreatedAt"],
            ["Document.id", "Document.createdAt"],
            name="fk_suggestion_document"
        ),
    )

    def __repr__(self):
        return f"<Suggestion(id={self.id}, documentId={self.documentId}, isResolved={self.isResolved})>"
