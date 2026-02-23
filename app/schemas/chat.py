from pydantic import BaseModel, UUID4
from typing import List, Optional, Any, Dict
from datetime import datetime


class MessagePart(BaseModel):
    type: str
    text: Optional[str] = None
    image: Optional[str] = None
    # File attachment fields
    url: Optional[str] = None
    name: Optional[str] = None
    mediaType: Optional[str] = None
    # Tool call fields
    toolCallId: Optional[str] = None
    toolName: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None


class MessageAttachment(BaseModel):
    name: str
    contentType: str
    url: str


class MessageRequest(BaseModel):
    id: Optional[str] = None
    role: str
    parts: List[MessagePart]
    attachments: List[MessageAttachment] = []


class ChatRequest(BaseModel):
    id: Optional[str] = None
    messages: Optional[List[MessageRequest]] = None
    message: Optional[MessageRequest] = None
    modelId: Optional[str] = None
    selectedChatModel: Optional[str] = None
    selectedVisibilityType: Optional[str] = None


class MessageResponse(BaseModel):
    id: UUID4
    chatId: UUID4
    role: str
    parts: List[Dict[str, Any]]
    attachments: List[Dict[str, Any]]
    createdAt: datetime

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    id: UUID4
    title: str
    createdAt: datetime
    userId: UUID4
    visibility: str

    class Config:
        from_attributes = True


class ChatListResponse(BaseModel):
    chats: List[ChatResponse]


class VoteRequest(BaseModel):
    chatId: str
    messageId: str
    isUpvoted: bool


class VoteResponse(BaseModel):
    chatId: UUID4
    messageId: UUID4
    isUpvoted: bool

    class Config:
        from_attributes = True
