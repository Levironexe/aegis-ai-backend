import logging
from typing import AsyncGenerator, List, Dict, Any
import google.generativeai as genai
from app.config import settings

logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for Google Gemini API with streaming support"""

    def __init__(self, api_key: str = None):
        # Google Gemini API key
        self.api_key = api_key or settings.google_api_key or settings.ai_gateway_api_key

        # Configure Google Generative AI (will fail when actually used if key is missing)
        if self.api_key:
            genai.configure(api_key=self.api_key)
        else:
            logger.warning("GOOGLE_API_KEY not configured - Gemini client will fail if used")

    async def stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
    ):
        # Strip provider prefix
        if "/" in model:
            model = model.split("/", 1)[1]

        logger.info(f"Streaming from Gemini: model={model}, messages={len(messages)}")

        # Convert OpenAI-style messages → Gemini format
        contents = []
        for msg in messages:
            role = msg["role"]

            if role == "system":
                continue  # Gemini doesn't support system role

            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "")
                    for part in content
                    if part.get("type") == "text"
                )

            contents.append({
                "role": role,
                "parts": [{"text": content}]
            })

        try:
            # Create model instance
            gemini_model = genai.GenerativeModel(model)

            # Generate content stream
            stream = gemini_model.generate_content(
                contents=contents,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": 4096,
                },
                stream=True,
            )

            for chunk in stream:
                if chunk.text:
                    yield {
                        "choices": [{
                            "delta": {
                                "content": chunk.text
                            }
                        }]
                    }

            logger.info("Gemini streaming complete")

        except Exception as e:
            logger.error(
                f"Gemini streaming error: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise
