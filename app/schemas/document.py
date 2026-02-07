from pydantic import BaseModel, UUID4
from typing import Optional, List
from datetime import datetime


class DocumentCreate(BaseModel):
    title: str
    content: Optional[str] = None
    kind: str = "text"  # text, code, image, sheet


class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None


class DocumentResponse(BaseModel):
    id: UUID4
    createdAt: datetime
    title: str
    content: Optional[str]
    kind: str
    userId: UUID4

    class Config:
        from_attributes = True


class SuggestionCreate(BaseModel):
    documentId: str
    documentCreatedAt: datetime
    originalText: str
    suggestedText: str
    description: Optional[str] = None


class SuggestionResponse(BaseModel):
    id: UUID4
    documentId: UUID4
    documentCreatedAt: datetime
    originalText: str
    suggestedText: str
    description: Optional[str]
    isResolved: bool
    userId: UUID4
    createdAt: datetime

    class Config:
        from_attributes = True


class SuggestionListResponse(BaseModel):
    suggestions: List[SuggestionResponse]
