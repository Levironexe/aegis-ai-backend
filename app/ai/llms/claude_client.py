import logging
import base64
import httpx
from typing import AsyncGenerator, List, Dict, Any
from anthropic import AsyncAnthropic
from app.config import settings

logger = logging.getLogger(__name__)

class ClaudeClient:
    """Client for Anthropic Claude API with streaming support"""

    def __init__(self, api_key: str = None):
        # Anthropic API key
        self.api_key = api_key or settings.anthropic_api_key or settings.ai_gateway_api_key

        # Create Anthropic async client (will raise error when actually used if key is missing)
        self.client = AsyncAnthropic(api_key=self.api_key) if self.api_key else None

        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not configured - Claude client will fail if used")

    async def stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat completion from Claude

        Args:
            model: Model name (e.g., "claude-haiku-4-5", "claude-haiku-4-5", "claude-opus-4-6")
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature

        Yields:
            Chunks in OpenAI-compatible format for consistency
        """

        # Strip provider prefix if present (e.g., "anthropic/claude-3-5-sonnet-20241022" -> "claude-3-5-sonnet-20241022")
        if "/" in model:
            model = model.split("/", 1)[1]

        # Map common model names to Claude API names (2025 models)
        model_mapping = {
            "claude-haiku-4.5": "claude-haiku-4-5",
            # "claude-sonnet-4.5": "claude-haiku-4-5",
            # "claude-opus-4.5": "claude-opus-4-6",
            # # Legacy mappings
            # "claude-3-5-haiku-20241022": "claude-haiku-4-5",
            # "claude-3-5-sonnet-20241022": "claude-haiku-4-5",
        }
        claude_model = model_mapping.get(model, model)

        logger.info(f"Streaming from Claude: model={claude_model}, messages={len(messages)}")

        try:
            # Convert messages to Claude format
            claude_messages = []
            system_message = None

            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")

                # Handle content as string or list
                if isinstance(content, list):
                    # Convert to Claude's content format
                    claude_content = []
                    for part in content:
                        if part.get("type") == "text":
                            claude_content.append({
                                "type": "text",
                                "text": part.get("text", "")
                            })
                        elif part.get("type") == "image_url":
                            # Claude expects image format
                            image_url = part.get("image_url", {}).get("url", "")
                            logger.info(f"Processing image URL: {image_url}")
                            if image_url:
                                # For non-HTTPS URLs, fetch and convert to base64
                                if not image_url.startswith("https://"):
                                    try:
                                        logger.info(f"Fetching HTTP image: {image_url}")
                                        async with httpx.AsyncClient() as http_client:
                                            response = await http_client.get(image_url)
                                            response.raise_for_status()
                                            image_data = response.content

                                            # Detect media type from response headers or URL
                                            media_type = response.headers.get("content-type", "image/png")
                                            if not media_type.startswith("image/"):
                                                media_type = "image/png"

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
                                            logger.info(f"Successfully converted image to base64, size: {len(image_data)} bytes")
                                    except Exception as e:
                                        logger.error(f"Failed to fetch image from {image_url}: {e}")
                                        # Skip this image if we can't fetch it
                                        pass
                                else:
                                    # For HTTPS URLs, use URL directly
                                    logger.info(f"Using HTTPS URL directly: {image_url}")
                                    claude_content.append({
                                        "type": "image",
                                        "source": {
                                            "type": "url",
                                            "url": image_url
                                        }
                                    })
                    content = claude_content if claude_content else ""

                # Separate system messages
                if role == "system":
                    # System messages must be strings
                    if isinstance(content, list):
                        text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                        system_message = " ".join(text_parts)
                    else:
                        system_message = content
                elif role in ["user", "assistant"]:
                    claude_messages.append({
                        "role": role,
                        "content": content
                    })

            # Log what we're sending to Claude
            logger.info(f"Sending {len(claude_messages)} messages to Claude")
            for i, msg in enumerate(claude_messages):
                content = msg.get("content", "")
                if isinstance(content, list):
                    logger.info(f"Message {i}: role={msg.get('role')}, content_parts={len(content)}")
                    for j, part in enumerate(content):
                        logger.info(f"  Part {j}: type={part.get('type')}")
                else:
                    logger.info(f"Message {i}: role={msg.get('role')}, content_type=str")

            # Stream response from Claude
            stream = await self.client.messages.create(
                model=claude_model,
                max_tokens=4096,
                temperature=temperature,
                system=system_message if system_message else "You are a helpful AI assistant.",
                messages=claude_messages,
                stream=True
            )

            # Yield chunks in OpenAI-compatible format
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        yield {
                            "choices": [{
                                "delta": {
                                    "content": event.delta.text
                                }
                            }]
                        }

            logger.info("Claude streaming complete")

        except Exception as e:
            logger.error(f"Claude streaming error: {type(e).__name__}: {str(e)}", exc_info=True)
            raise

