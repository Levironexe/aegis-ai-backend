# app/ai/gateway_client.py

import logging
from app.ai.llms.claude_client import ClaudeClient
from app.ai.llms.gemini_client import GeminiClient
from app.ai.langgraph_agent import LangGraphAgent
from app.tools.example_ioc_tool import ExampleIOCTool

logger = logging.getLogger(__name__)


class GatewayClient:
    def __init__(self):
        self.claude = ClaudeClient()
        self.gemini = GeminiClient()

        # Initialize LangGraph agent with example tools
        self.agent = LangGraphAgent()
        example_tool = ExampleIOCTool()
        self.agent.register_tools([example_tool.to_langchain_tool()])

        logger.info("Gateway initialized with Claude, Gemini, and LangGraph agent")

    def get_client(self, model: str):
        m = model.lower().strip()

        if "/" in m:
            provider, _ = m.split("/", 1)
        else:
            provider = m

        # Route agent requests to LangGraph agent
        if provider == "agent":
            return self.agent

        if provider in ("anthropic", "claude"):
            return self.claude

        if provider in ("google", "gemini"):
            return self.gemini

        if provider in ("openai", "gpt"):
            return self.openai

        raise ValueError(f"Unsupported model: {model}")


    async def stream_chat_completion(self, model, messages, **kwargs):
        m = model.lower().strip()

        if "/" in m:
            provider, raw_model = m.split("/", 1)
        else:
            provider = m
            raw_model = model

        client = self.get_client(model)

        async for chunk in client.stream_chat_completion(
            model=raw_model,
            messages=messages,
            **kwargs
        ):
            yield chunk


    async def chat_completion(self, model, messages, **kwargs):
        client = self.get_client(model)
        return await client.chat_completion(
            model=model,
            messages=messages,
            **kwargs
        )


# Singleton gateway instance
gateway_client = GatewayClient()
