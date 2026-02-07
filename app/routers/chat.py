from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from uuid import UUID
import json
import asyncio
from datetime import datetime
from typing import AsyncGenerator, List, Dict, Any

from app.database import get_db
from app.schemas.chat import ChatRequest, ChatResponse, ChatListResponse, MessageResponse
from app.models.chat import Chat, Message
from app.models.user import User
from app.utils.session import session_manager
from app.config import settings
from app.ai.gateway_client import claude_client

import logging

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TITLE_PROMPT = """Generate a short chat title (2-5 words) summarizing the user's message.

Output ONLY the title text. No prefixes, no formatting.

Examples:
- "what's the weather in nyc" → Weather in NYC
- "help me write an essay about space" → Space Essay Help
- "hi" → New Conversation
- "debug my python code" → Python Debugging

Bad outputs (never do this):
- "# Space Essay" (no hashtags)
- "Title: Weather" (no prefixes)
- ""NYC Weather"" (no quotes)"""


async def generate_chat_title(user_message_text: str) -> str:
    """Generate a concise title from the user's first message"""
    try:
        # Use Claude to generate title
        messages = [
            {"role": "system", "content": TITLE_PROMPT},
            {"role": "user", "content": user_message_text}
        ]

        full_title = ""
        async for chunk in claude_client.stream_chat_completion(
            model=settings.default_title_model,
            messages=messages,
            temperature=0.7
        ):
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    full_title += content

        # Clean up the title (remove quotes, hashtags, prefixes)
        import re
        title = re.sub(r'^[#*"\s]+', '', full_title)
        title = re.sub(r'["]+$', '', title)
        title = title.strip()
        return title if title else "New Conversation"

    except Exception as e:
        logger.error(f"Error generating title: {e}")
        return "New Conversation"


async def get_current_user_or_guest(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    """Get current authenticated user or create/return guest user"""
    user_id_str = session_manager.get_session(request, "user_id")

    # If user has session, return authenticated user
    if user_id_str:
        try:
            user_id = UUID(user_id_str)
            result = await db.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()

            if user:
                return user
        except (ValueError, Exception):
            pass  # Fall through to guest creation

    # Create or get guest user
    # Check if there's a guest_id in cookies
    guest_id_str = request.cookies.get("guest_id")

    if guest_id_str:
        try:
            guest_id = UUID(guest_id_str)
            result = await db.execute(
                select(User).where(User.id == guest_id, User.email.like("guest_%"))
            )
            guest_user = result.scalar_one_or_none()

            if guest_user:
                return guest_user
        except (ValueError, Exception):
            pass

    # Create new guest user
    import uuid
    guest_uuid = uuid.uuid4()
    guest_user = User(
        id=guest_uuid,
        email=f"guest_{guest_uuid}@aegis.local",
        name="Guest User",
        picture=None,
        createdAt=datetime.utcnow()
    )
    db.add(guest_user)
    await db.commit()
    await db.refresh(guest_user)

    return guest_user


async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    """Get current authenticated user from session (strict, no guest)"""
    user_id_str = session_manager.get_session(request, "user_id")
    if not user_id_str:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        user_id = UUID(user_id_str)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid session")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


async def stream_chat_response(
    chat_request: ChatRequest,
    chat_id: UUID,
    user: User,
    db: AsyncSession
) -> AsyncGenerator[str, None]:
    """Stream chat responses using LiteLLM"""

    try:
        logger.info(f"Starting stream_chat_response for chat_id={chat_id}, user={user.email}")

        # Normalize messages - handle both single message and messages array
        message_list = chat_request.messages if chat_request.messages else ([chat_request.message] if chat_request.message else [])
        logger.info(f"Processing {len(message_list)} messages")

        # Convert messages to LiteLLM format
        messages = []
        for msg in message_list:
            content_parts = []
            for part in msg.parts:
                if part.type == "text" and part.text:
                    content_parts.append({"type": "text", "text": part.text})
                elif part.type == "image" and part.image:
                    content_parts.append({"type": "image_url", "image_url": {"url": part.image}})

            if content_parts:
                messages.append({
                    "role": msg.role,
                    "content": content_parts if len(content_parts) > 1 else content_parts[0].get("text", "")
                })

        # Add system message
        system_message = {
            "role": "system",
            "content": "You are a helpful AI assistant."
        }
        messages.insert(0, system_message)
        logger.info(f"Total messages after system prompt: {len(messages)}")

        # Determine the model to use - default to Gemini (matches frontend)
        model = chat_request.selectedChatModel or chat_request.modelId or settings.default_chat_model
        logger.info(f"Using model: {model}")

        # Stream from Claude
        logger.info(f"Calling Claude with model={model}, stream=True")

        # Generate message ID upfront
        import uuid
        message_id = str(uuid.uuid4())

        # Send message creation event first (AI SDK expects this)
        message_annotation = {
            "id": message_id,
            "role": "assistant",
            "content": [],
            "createdAt": datetime.utcnow().isoformat()
        }
        yield f"2:{json.dumps(message_annotation)}\n"

        full_content = ""
        chunk_count = 0
        text_started = False

        async for chunk in claude_client.stream_chat_completion(
            model=model,
            messages=messages,
            temperature=0.7
        ):
            chunk_count += 1

            # Parse OpenAI-compatible chunk format
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")

                if content:
                    # Send text-start event on first chunk
                    if not text_started:
                        start_event = {"type": "text-start", "id": "t1"}
                        yield f"0:{json.dumps(start_event)}\n"
                        text_started = True

                    full_content += content

                    # Stream in AI SDK format (text-delta event with id)
                    event_data = {
                        "type": "text-delta",
                        "id": "t1",
                        "delta": content
                    }
                    yield f"0:{json.dumps(event_data)}\n"

        # Send text-end event
        if text_started:
            end_event = {"type": "text-end", "id": "t1"}
            yield f"0:{json.dumps(end_event)}\n"

        logger.info(f"Streaming complete. Processed {chunk_count} chunks, total content length: {len(full_content)}")

        # Save assistant message to database
        assistant_message = Message(
            chatId=chat_id,
            role="assistant",
            parts=[{"type": "text", "text": full_content}],
            attachments=[],
            createdAt=datetime.utcnow()
        )
        db.add(assistant_message)
        await db.commit()
        await db.refresh(assistant_message)

        logger.info(f"Saved assistant message with id={assistant_message.id}")

        # Generate title if this is the first message
        result = await db.execute(select(Chat).where(Chat.id == chat_id))
        current_chat = result.scalar_one_or_none()

        if current_chat and current_chat.title == "New Chat":
            # Get the user's first message text
            user_message_text = ""
            for msg in message_list:
                if msg.role == "user":
                    for part in msg.parts:
                        if part.type == "text" and part.text:
                            user_message_text = part.text
                            break
                    break

            if user_message_text:
                logger.info("Generating title for new chat")
                title = await generate_chat_title(user_message_text)
                current_chat.title = title
                await db.commit()
                logger.info(f"Updated chat title to: {title}")

        # Send finish event in AI SDK format
        finish_event = {
            "type": "finish",
            "finishReason": "stop",
            "usage": {
                "promptTokens": 0,
                "completionTokens": len(full_content.split())
            }
        }
        yield f"d:{json.dumps(finish_event)}\n"

    except Exception as e:
        logger.error(f"Error in stream_chat_response: {type(e).__name__}: {str(e)}", exc_info=True)
        error_event = {
            "type": "error",
            "error": str(e)
        }
        yield f"3:{json.dumps(error_event)}\n"


@router.post("")
async def chat(
    chat_request: ChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_or_guest)
):
    """Stream chat responses"""

    # Get or create chat
    chat_id = None
    if chat_request.id:
        try:
            chat_id = UUID(chat_request.id)
        except (ValueError, AttributeError):
            # Invalid UUID, treat as new chat
            chat_id = None

    if chat_id:
        result = await db.execute(select(Chat).where(Chat.id == chat_id))
        chat = result.scalar_one_or_none()

        if not chat:
            # Chat ID provided but doesn't exist - create new chat with this ID
            visibility = chat_request.selectedVisibilityType or "private"
            chat = Chat(
                id=chat_id,
                title="New Chat",
                userId=user.id,
                visibility=visibility,
                createdAt=datetime.utcnow()
            )
            db.add(chat)
            await db.commit()
            await db.refresh(chat)
        elif chat.userId != user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this chat")
    else:
        # Create new chat
        visibility = chat_request.selectedVisibilityType or "private"
        chat = Chat(
            title="New Chat",
            userId=user.id,
            visibility=visibility,
            createdAt=datetime.utcnow()
        )
        db.add(chat)
        await db.commit()
        await db.refresh(chat)
        chat_id = chat.id

    # Save user message - handle both message formats
    user_message = None
    if chat_request.messages and chat_request.messages[-1].role == "user":
        user_message = chat_request.messages[-1]
    elif chat_request.message and chat_request.message.role == "user":
        user_message = chat_request.message

    if user_message:
        message = Message(
            chatId=chat_id,
            role=user_message.role,
            parts=[part.dict() for part in user_message.parts],
            attachments=[att.dict() for att in user_message.attachments],
            createdAt=datetime.utcnow()
        )
        db.add(message)
        await db.commit()

    # Return streaming response with guest cookie if user is a guest
    # AI SDK expects text/plain not text/event-stream
    response = StreamingResponse(
        stream_chat_response(chat_request, chat_id, user, db),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Vercel-AI-Data-Stream": "v1"
        }
    )

    # Set guest cookie if this is a guest user
    if user.email.startswith("guest_"):
        response.set_cookie(
            key="guest_id",
            value=str(user.id),
            httponly=True,
            samesite="lax",
            max_age=86400 * 30  # 30 days
        )

    return response


@router.post("/{chat_id}/stream")
async def stream_chat(
    chat_id: str,
    chat_request: ChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_or_guest)
):
    """Resume streaming for an existing chat"""

    # Get existing chat
    result = await db.execute(select(Chat).where(Chat.id == UUID(chat_id)))
    chat = result.scalar_one_or_none()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.userId != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this chat")

    # Save user message if provided
    user_message = None
    if chat_request.messages and chat_request.messages[-1].role == "user":
        user_message = chat_request.messages[-1]
    elif chat_request.message and chat_request.message.role == "user":
        user_message = chat_request.message

    if user_message:
        message = Message(
            chatId=UUID(chat_id),
            role=user_message.role,
            parts=[part.dict() for part in user_message.parts],
            attachments=[att.dict() for att in user_message.attachments],
            createdAt=datetime.utcnow()
        )
        db.add(message)
        await db.commit()

    # Return streaming response with guest cookie if user is a guest
    # AI SDK expects text/plain not text/event-stream
    response = StreamingResponse(
        stream_chat_response(chat_request, UUID(chat_id), user, db),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Vercel-AI-Data-Stream": "v1"
        }
    )

    # Set guest cookie if this is a guest user
    if user.email.startswith("guest_"):
        response.set_cookie(
            key="guest_id",
            value=str(user.id),
            httponly=True,
            samesite="lax",
            max_age=86400 * 30  # 30 days
        )

    return response


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(
    chat_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_or_guest)
):
    """Get a specific chat by ID"""

    result = await db.execute(
        select(Chat).where(Chat.id == UUID(chat_id))
    )
    chat = result.scalar_one_or_none()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.userId != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this chat")

    return ChatResponse.from_orm(chat)


@router.get("/history", response_model=ChatListResponse)
async def get_chat_history(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_or_guest)
):
    """Get user's chat history"""

    result = await db.execute(
        select(Chat)
        .where(Chat.userId == user.id)
        .order_by(desc(Chat.createdAt))
    )
    chats = result.scalars().all()

    return ChatListResponse(chats=[ChatResponse.from_orm(chat) for chat in chats])


@router.delete("/history")
async def delete_all_chats(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_or_guest)
):
    """Delete all chats for the current user"""

    result = await db.execute(
        select(Chat).where(Chat.userId == user.id)
    )
    chats = result.scalars().all()

    for chat in chats:
        await db.delete(chat)

    await db.commit()

    return {"success": True, "deleted_count": len(chats)}


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_or_guest)
):
    """Delete a chat"""

    result = await db.execute(select(Chat).where(Chat.id == chat_id))
    chat = result.scalar_one_or_none()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.userId != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this chat")

    await db.delete(chat)
    await db.commit()

    return {"success": True}


@router.get("/{chat_id}/messages")
async def get_chat_messages(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_or_guest)
):
    """Get messages for a specific chat"""

    # Verify chat ownership
    result = await db.execute(select(Chat).where(Chat.id == chat_id))
    chat = result.scalar_one_or_none()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.userId != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this chat")

    # Get messages
    result = await db.execute(
        select(Message)
        .where(Message.chatId == chat_id)
        .order_by(Message.createdAt)
    )
    messages = result.scalars().all()

    return {"messages": [MessageResponse.from_orm(msg) for msg in messages]}
