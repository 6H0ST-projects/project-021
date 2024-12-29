"""
Base agent class and common functionality for Sidewinder agents.
"""

from typing import Dict, Any, TypeVar, Generic
from pydantic import BaseModel, Field

AgentState = TypeVar("AgentState", bound=BaseModel)


class BaseAgentState(BaseModel):
    """Base state for all agents."""
    messages: list[str] = Field(default_factory=list)
    error: str | None = Field(default=None)
    completed: bool = Field(default=False)


class BaseAgent(Generic[AgentState]):
    """Base agent class with common functionality."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    async def run(self, state: AgentState) -> AgentState:
        """
        Run the agent's main logic.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        raise NotImplementedError("Agents must implement run method") 