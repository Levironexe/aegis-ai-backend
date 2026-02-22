"""Base Tool Class for AI Agent Tools"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel


class BaseTool(ABC):
    """Base class for all AI agent tools."""

    def __init__(self):
        self.name = self.__class__.__name__.replace("Tool", "").lower()
        self.needs_approval = False

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the LLM."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> type[BaseModel]:
        """Pydantic model for tool input validation."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with validated inputs."""
        pass

    def to_langchain_tool(self):
        """Convert to LangChain tool format."""
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            coroutine=self.execute,  # Use coroutine for async functions
            name=self.name,
            description=self.description,
            args_schema=self.input_schema,
        )
