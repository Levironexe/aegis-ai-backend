from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from uuid import UUID
from datetime import datetime

from app.database import get_db
from app.schemas.document import (
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    SuggestionCreate,
    SuggestionResponse,
    SuggestionListResponse
)
from app.models.document import Document, Suggestion
from app.models.user import User
from app.routers.chat import get_current_user

router = APIRouter(prefix="/api/document", tags=["documents"])


@router.post("", response_model=DocumentResponse)
async def create_document(
    document: DocumentCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Create a new document"""

    new_document = Document(
        title=document.title,
        content=document.content,
        kind=document.kind,
        userId=user.id,
        createdAt=datetime.utcnow()
    )

    db.add(new_document)
    await db.commit()
    await db.refresh(new_document)

    return DocumentResponse.from_orm(new_document)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    created_at: datetime,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Get a specific document"""

    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.createdAt == created_at
        )
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.userId != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this document")

    return DocumentResponse.from_orm(document)


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: UUID,
    created_at: datetime,
    document_update: DocumentUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Update a document"""

    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.createdAt == created_at
        )
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.userId != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to update this document")

    if document_update.title is not None:
        document.title = document_update.title
    if document_update.content is not None:
        document.content = document_update.content

    await db.commit()
    await db.refresh(document)

    return DocumentResponse.from_orm(document)


@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    created_at: datetime,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Delete a document"""

    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.createdAt == created_at
        )
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.userId != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this document")

    await db.delete(document)
    await db.commit()

    return {"success": True}


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """List all documents for the current user"""

    result = await db.execute(
        select(Document)
        .where(Document.userId == user.id)
        .order_by(desc(Document.createdAt))
    )
    documents = result.scalars().all()

    return [DocumentResponse.from_orm(doc) for doc in documents]


# Suggestions endpoints
@router.post("/suggestions", response_model=SuggestionResponse)
async def create_suggestion(
    suggestion: SuggestionCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Create a new suggestion for a document"""

    # Verify document exists and user has access
    result = await db.execute(
        select(Document).where(
            Document.id == UUID(suggestion.documentId),
            Document.createdAt == suggestion.documentCreatedAt
        )
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.userId != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to create suggestions for this document")

    new_suggestion = Suggestion(
        documentId=UUID(suggestion.documentId),
        documentCreatedAt=suggestion.documentCreatedAt,
        originalText=suggestion.originalText,
        suggestedText=suggestion.suggestedText,
        description=suggestion.description,
        userId=user.id,
        createdAt=datetime.utcnow()
    )

    db.add(new_suggestion)
    await db.commit()
    await db.refresh(new_suggestion)

    return SuggestionResponse.from_orm(new_suggestion)


@router.get("/suggestions/{document_id}", response_model=SuggestionListResponse)
async def get_suggestions(
    document_id: UUID,
    created_at: datetime,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Get all suggestions for a document"""

    # Verify document exists and user has access
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.createdAt == created_at
        )
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.userId != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access suggestions for this document")

    result = await db.execute(
        select(Suggestion).where(
            Suggestion.documentId == document_id,
            Suggestion.documentCreatedAt == created_at
        )
    )
    suggestions = result.scalars().all()

    return SuggestionListResponse(
        suggestions=[SuggestionResponse.from_orm(sug) for sug in suggestions]
    )
