import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from core.config import ModelInfo, manager
from core.registry import LLMTool


class LLMBackend(ABC):
    """LLM backend base class."""

    @abstractmethod
    def __init__(self, model: ModelInfo) -> None: ...

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


def get_backend(model: ModelInfo) -> LLMBackend:
    """
    Get LLM backend based on the current profile.
    """

    logger = logging.getLogger(__name__)

    match model.provider:
        case "openai":
            from core.backends.openai import OpenAIBackend

            return OpenAIBackend(model)
        case "vertexai":
            from core.backends.vertexai import VertexAIBackend

            return VertexAIBackend(model)
        case "litellm":
            from core.backends.litellm import LiteLLMBackend

            return LiteLLMBackend(model)
        case "openrouter":
            from core.backends.openrouter import OpenRouterBackend

            return OpenRouterBackend(model)

    logger.error("Invalid back-end name, defaulting to OpenAI")
    return OpenAIBackend(model)
