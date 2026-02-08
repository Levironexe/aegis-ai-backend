from .base import BaseTool
from pydantic import BaseModel, Field

class AInput(BaseModel):
    """Input schema for Tool A."""
    name: str
    age: int
    
class ATool(BaseTool):
    """Tool A does something interesting."""

    def __init__(self):
        super().__init__()
        self.name = "toolA"
        self.needs_approval = True

    @property
    def description(self) -> str:
        return "Tool A processes a name and age."

    @property
    def input_schema(self) -> type[BaseModel]:
        return AInput

    async def execute(self, name: str, age: int) -> dict:
        """Execute Tool A logic."""
        return {"message": f"Processed {name}, age {age}."}
    