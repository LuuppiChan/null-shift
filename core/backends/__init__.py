from abc import ABC, abstractmethod
import logging
from typing import AsyncIterator, Optional

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel
from core.registry import LLMTool

from core.config import manager


class LLMBackend(ABC):
    """LLM backend base class."""

    @abstractmethod
    def stream(
        self, messages: list[BaseMessage], tools: Optional[list[LLMTool]]
    ) -> AsyncIterator[AIMessage]:
        """Stream a response from the LLM.

        Args:
            messages: Conversation history in langchain message format.
            tools: List of BaseTool from langchain.

        Yields:
            AIMessage: Incremental text delta or a complete tool call.
        """

    @abstractmethod
    def structured_output[T: BaseModel](
        self, messages: list[BaseMessage], structure: type[T]
    ) -> T:
        """Returns structured output as defined by the structure parameter."""

    @abstractmethod
    def reset_customs(self):
        """Reset custom configuration for the back-end to config values."""

    @abstractmethod
    def set_model(self, model: str):
        """Set custom model."""

    @abstractmethod
    def set_temperature(self, temperature: float):
        """Set custom temperature."""

    @abstractmethod 
    def set_thinking(self, level: str | None):
        """Set custom thinking level"""


def get_backend() -> LLMBackend:
    """
    Get LLM backend based on the current profile.
    """

    logger = logging.getLogger(__name__)

    match manager.get_config().llm_provider:
        case "openai":
            return _manager.openai
        case "vertexai":
            return _manager.vertexai
        case "litellm":
            return _manager.litellm

    logger.error("Invalid back-end name, defaulting to OpenAI")
    return _manager.openai


class BackendManager:
    def __init__(self) -> None:
        from core.backends.litellm import LiteLLMBackend
        from core.backends.openai import OpenAIBackend
        from core.backends.vertexai import VertexAIBackend

        self.openai = OpenAIBackend()
        self.vertexai = VertexAIBackend()
        self.litellm = LiteLLMBackend()


_manager = BackendManager()
