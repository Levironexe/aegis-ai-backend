# Image Upload and Vision Processing Fix

**Date:** February 23, 2026

## Problem Summary

Uploaded images were successfully stored on the backend but LLMs (both Claude models and LangGraph agent) could not see or process them. Users could upload images through the frontend, but when querying the AI, the models would not acknowledge or analyze the image content.

---

## Root Causes

### 1. Backend Message Conversion Missing File Parts
**Location:** `app/routers/chat.py` lines 186-195

**Issue:** The message conversion logic only handled `type == "text"` and `type == "image"` parts, but the frontend was sending `type == "file"` for uploaded attachments.

**Impact:** Image attachments were silently dropped during message processing, never reaching the LLM.

### 2. Single-Part Messages Losing Image Data
**Location:** `app/routers/chat.py` line 201

**Issue:** Logic simplified single-part messages to extract only text content:
```python
content_parts if len(content_parts) > 1 else content_parts[0].get("text", "")
```

**Impact:** When a message contained only an image (no text), the image part was lost.

### 3. Claude API HTTPS-Only Requirement
**Location:** `app/ai/llms/claude_client.py`

**Issue:** Claude's API rejects HTTP URLs for images (returns error: "Only HTTPS URLs are supported"), but local development uses `http://localhost:8000/api/files/...`.

**Error:**
```
BadRequestError: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Only HTTPS URLs are supported.'}}
```

**Impact:** Images uploaded in local development could not be processed by Claude.

### 4. LangGraph Agent Not Processing Images
**Location:** `app/ai/langgraph_agent.py` lines 788-794

**Issue:** The agent's message conversion stripped out image parts, only extracting text:
```python
if isinstance(content, list):
    text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
    content = " ".join(text_parts)
```

**Impact:** The agent/cyber-analyst model could not process any images.

---

## Solutions Implemented

### 1. Added File Part Schema Support
**File:** `app/schemas/chat.py`

Added file attachment fields to `MessagePart`:
```python
class MessagePart(BaseModel):
    type: str
    text: Optional[str] = None
    image: Optional[str] = None
    # File attachment fields
    url: Optional[str] = None
    name: Optional[str] = None
    mediaType: Optional[str] = None
```

### 2. Updated Message Conversion to Handle File Parts
**File:** `app/routers/chat.py` lines 190-196

Added file part handling that converts image files to `image_url` format:
```python
elif part.type == "file" and part.url:
    # Handle file attachments - check if it's an image by mediaType
    if part.mediaType and part.mediaType.startswith("image"):
        content_parts.append({"type": "image_url", "image_url": {"url": part.url}})
    else:
        # For non-image files, just mention the filename in text
        content_parts.append({"type": "text", "text": f"[File: {part.name or 'attachment'}]"})
```

Changed to always use array format for multimodal support:
```python
messages.append({
    "role": msg.role,
    "content": content_parts  # Always use array format
})
```

### 3. HTTP Image Fetching and Base64 Conversion
**File:** `app/ai/llms/claude_client.py` lines 77-122

Added logic to fetch HTTP images and convert to base64 inline:
```python
if not image_url.startswith("https://"):
    # Fetch HTTP image
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(image_url)
        image_data = response.content
        media_type = response.headers.get("content-type", "image/png")

        # Convert to base64
        base64_data = base64.b64encode(image_data).decode("utf-8")

        claude_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_data
            }
        })
else:
    # For HTTPS URLs, use URL directly
    claude_content.append({
        "type": "image",
        "source": {
            "type": "url",
            "url": image_url
        }
    })
```

**Added imports:**
```python
import base64
import httpx
```

### 4. Implemented Image Processing in LangGraph Agent
**File:** `app/ai/langgraph_agent.py` lines 793-843

Copied the exact same image processing logic from `claude_client.py` to handle multimodal content:
- Detects `image_url` parts in message content
- Fetches HTTP images and converts to base64
- Uses Claude's image format: `{"type": "image", "source": {...}}`
- Preserves multimodal content structure for LangChain messages

**Added imports:**
```python
import base64
import httpx
```

---

## Testing Results

### Before Fix
- ❌ Regular Claude models: Could not see images
- ❌ Agent/cyber-analyst: Could not see images
- ❌ Error: "Only HTTPS URLs are supported"

### After Fix
- ✅ Regular Claude models (anthropic/claude-haiku-4.5): Can process images
- ✅ Agent/cyber-analyst (agent/cyber-analyst): Can process images
- ✅ HTTP images automatically converted to base64
- ✅ HTTPS images used directly

---

## Files Modified

1. `app/schemas/chat.py` - Added file attachment fields
2. `app/routers/chat.py` - Updated message conversion logic
3. `app/ai/llms/claude_client.py` - Added HTTP image fetching and base64 conversion
4. `app/ai/langgraph_agent.py` - Implemented image processing for LangGraph agent

---

## Key Learnings

1. **Frontend-Backend Contract:** Always verify the exact structure being sent from frontend before implementing backend processing
2. **Multimodal Support:** Always use array format for message content, even for single parts, to maintain consistency
3. **Local Development Constraints:** Claude API's HTTPS requirement means local development needs base64 conversion for images
4. **Consistency Across Models:** Both direct Claude API calls and LangGraph agents need identical image processing logic

---

## Production Deployment Notes

For production deployment with HTTPS URLs (e.g., Vercel Blob Storage):
- Images can be passed as URLs directly without base64 conversion
- Performance benefit: No need to fetch and encode images
- The current implementation handles both cases automatically
