"""
Claude AI Client
Direct integration with Anthropic's Claude API
"""
import logging
from typing import AsyncGenerator, List, Dict, Any
from anthropic import AsyncAnthropic
from app.config import settings

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Client for Anthropic Claude API with streaming support"""

    def __init__(self, api_key: str = None):
        # Anthropic API key
        self.api_key = api_key or settings.anthropic_api_key or settings.ai_gateway_api_key
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY or AI_GATEWAY_API_KEY is required")

        # Create Anthropic async client
        self.client = AsyncAnthropic(api_key=self.api_key)

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
            "claude-sonnet-4.5": "claude-haiku-4-5",
            "claude-opus-4.5": "claude-opus-4-6",
            # Legacy mappings
            "claude-3-5-haiku-20241022": "claude-haiku-4-5",
            "claude-3-5-sonnet-20241022": "claude-haiku-4-5",
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
                    # Extract text from content parts
                    text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    content = " ".join(text_parts)

                # Separate system messages
                if role == "system":
                    system_message = content
                elif role in ["user", "assistant"]:
                    claude_messages.append({
                        "role": role,
                        "content": content
                    })

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


# Singleton instance
claude_client = ClaudeClient()
