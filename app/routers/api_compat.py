"""
API Compatibility Router
Provides compatibility endpoints for the frontend that expects certain paths
"""
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from uuid import UUID

from app.database import get_db
from app.models.chat import Chat
from app.models.user import User
from app.routers.chat import get_current_user, get_current_user_or_guest
from app.schemas.chat import ChatListResponse, ChatResponse

router = APIRouter(prefix="/api", tags=["compatibility"])


@router.get("/history")
async def get_history_compat(
    limit: int = 20,
    ending_before: str = None,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_or_guest)
):
    """Get chat history - compatibility endpoint for /api/history"""

    query = select(Chat).where(Chat.userId == user.id)

    # If pagination cursor provided, only get chats before this ID
    if ending_before:
        # Get the createdAt timestamp of the cursor chat
        cursor_result = await db.execute(
            select(Chat).where(Chat.id == UUID(ending_before))
        )
        cursor_chat = cursor_result.scalar_one_or_none()
        if cursor_chat:
            query = query.where(Chat.createdAt < cursor_chat.createdAt)

    query = query.order_by(desc(Chat.createdAt)).limit(limit + 1)

    result = await db.execute(query)
    chats = result.scalars().all()

    # Check if there are more results
    has_more = len(chats) > limit
    if has_more:
        chats = chats[:limit]

    return {
        "chats": [
            {
                "id": str(chat.id),
                "title": chat.title,
                "createdAt": chat.createdAt.isoformat(),
                "userId": str(chat.userId),
                "visibility": chat.visibility,
            }
            for chat in chats
        ],
        "hasMore": has_more
    }


@router.delete("/history")
async def delete_history_compat(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_or_guest)
):
    """Delete all chat history - compatibility endpoint"""

    result = await db.execute(
        select(Chat).where(Chat.userId == user.id)
    )
    chats = result.scalars().all()

    for chat in chats:
        await db.delete(chat)

    await db.commit()

    return {"success": True}


@router.get("/auth/session")
async def get_session_compat(request: Request, db: AsyncSession = Depends(get_db)):
    """Session endpoint for NextAuth compatibility"""

    try:
        # Try to get current user or guest
        user = await get_current_user_or_guest(request, db)
        return {
            "user": {
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "image": user.picture,
            }
        }
    except Exception:
        # Should not happen with get_current_user_or_guest, but handle gracefully
        return {}
