from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID

from app.database import get_db
from app.schemas.chat import VoteRequest, VoteResponse
from app.models.chat import Vote, Message, Chat
from app.models.user import User
from app.routers.chat import get_current_user, get_current_user_or_guest

router = APIRouter(prefix="/api/vote", tags=["votes"])


@router.post("", response_model=VoteResponse)
async def create_or_update_vote(
    vote_request: VoteRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Create or update a vote on a message"""

    chat_id = UUID(vote_request.chatId)
    message_id = UUID(vote_request.messageId)

    # Verify chat exists and user has access
    result = await db.execute(select(Chat).where(Chat.id == chat_id))
    chat = result.scalar_one_or_none()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.userId != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to vote on this chat")

    # Verify message exists
    result = await db.execute(select(Message).where(Message.id == message_id))
    message = result.scalar_one_or_none()

    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    # Check if vote already exists
    result = await db.execute(
        select(Vote).where(
            Vote.chatId == chat_id,
            Vote.messageId == message_id
        )
    )
    existing_vote = result.scalar_one_or_none()

    if existing_vote:
        # Update existing vote
        existing_vote.isUpvoted = str(vote_request.isUpvoted)
        await db.commit()
        await db.refresh(existing_vote)
        return VoteResponse.from_orm(existing_vote)
    else:
        # Create new vote
        new_vote = Vote(
            chatId=chat_id,
            messageId=message_id,
            isUpvoted=str(vote_request.isUpvoted)
        )
        db.add(new_vote)
        await db.commit()
        await db.refresh(new_vote)
        return VoteResponse.from_orm(new_vote)


@router.get("/{chat_id}", response_model=list[VoteResponse])
async def get_votes_by_chat(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Get all votes for a chat"""

    # Verify chat exists and user has access
    result = await db.execute(select(Chat).where(Chat.id == chat_id))
    chat = result.scalar_one_or_none()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.userId != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this chat")

    # Get all votes for this chat
    result = await db.execute(
        select(Vote).where(Vote.chatId == chat_id)
    )
    votes = result.scalars().all()

    return [VoteResponse.from_orm(vote) for vote in votes]


@router.get("/{chat_id}/{message_id}", response_model=VoteResponse)
async def get_vote(
    chat_id: UUID,
    message_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Get a specific vote"""

    # Verify chat exists and user has access
    result = await db.execute(select(Chat).where(Chat.id == chat_id))
    chat = result.scalar_one_or_none()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.userId != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this chat")

    # Get vote
    result = await db.execute(
        select(Vote).where(
            Vote.chatId == chat_id,
            Vote.messageId == message_id
        )
    )
    vote = result.scalar_one_or_none()

    if not vote:
        raise HTTPException(status_code=404, detail="Vote not found")

    return VoteResponse.from_orm(vote)
