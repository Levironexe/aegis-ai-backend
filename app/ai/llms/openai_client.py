import logging
from typing import AsyncGenerator, List, Dict, Any
from openai import AsyncOpenAI
from app.config import settings

logger = logging.getLogger(__name__)

class OpenAIClient:
    """Client for OpenAI API with streaming support"""

    def __init__(self, api_key: str = None):
        # OpenAI API key
        self.api_key = api_key or settings.openai_api_key or settings.ai_gateway_api_key

        # Create OpenAI async client (will raise error when actually used if key is missing)
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not configured - OpenAI client will fail if used")

    async def stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat completion from OpenAI

        Args:
            model: Model name (e.g., "gpt-4.1-mini", "gpt-5.2")
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature

        Yields:
            Chunks in OpenAI-compatible format
        """

        # Strip provider prefix if present (e.g., "openai/gpt-4.1-mini" -> "gpt-4.1-mini")
        if "/" in model:
            model = model.split("/", 1)[1]

        logger.info(f"Streaming from OpenAI: model={model}, messages={len(messages)}")

        try:
            # Stream response from OpenAI
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True
            )

            # Yield chunks in OpenAI format (already compatible)
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield {
                            "choices": [{
                                "delta": {
                                    "content": delta.content
                                }
                            }]
                        }

            logger.info("OpenAI streaming complete")

        except Exception as e:
            logger.error(f"OpenAI streaming error: {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Non-streaming chat completion from OpenAI

        Args:
            model: Model name
            messages: List of message dicts
            temperature: Sampling temperature

        Returns:
            Response dict in OpenAI format
        """
        # Strip provider prefix if present
        if "/" in model:
            model = model.split("/", 1)[1]

        logger.info(f"Calling OpenAI: model={model}, messages={len(messages)}")

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False
            )

            return {
                "choices": [{
                    "message": {
                        "role": response.choices[0].message.role,
                        "content": response.choices[0].message.content
                    }
                }],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }

        except Exception as e:
            logger.error(f"OpenAI error: {type(e).__name__}: {str(e)}", exc_info=True)
            raise
