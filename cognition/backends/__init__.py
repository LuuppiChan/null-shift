"""
LLM backend system for the Cognition node.

``LLMBackend`` is the abstract base class. Concrete implementations live in
sibling modules:

- :mod:`cognition.backends.openai` — OpenAI-compatible endpoints
- :mod:`cognition.backends.vertexai` — Google VertexAI / Gemini

The :func:`build_backend` factory selects the right implementation from
``config.llm_provider``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator

from cognition.types import StreamChunk

if TYPE_CHECKING:
    from cognition.config import CognitionConfig


class LLMBackend(ABC):
    """Abstract provider wrapper.

    One concrete subclass per LLM provider. Subclasses adapt a LangChain
    chat model's streaming interface to produce :class:`~cognition.types.StreamChunk`
    objects so the :class:`~cognition.vector.Vector` orchestrator stays
    provider-agnostic.
    """

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response from the LLM.

        Args:
            messages: Conversation history in OpenAI message format. Media
                parts are in the internal mime_type format — each backend
                converts them to its native format before sending.
            tools: OpenAI function-calling schemas for available tools.

        Yields:
            StreamChunk: Incremental text delta or a complete tool call.
        """
        ...  # pragma: no cover


def build_backend(config: CognitionConfig) -> LLMBackend:
    """Instantiate the correct :class:`LLMBackend` from ``config.llm_provider``.

    Args:
        config: Resolved :class:`~cognition.config.CognitionConfig`.

    Returns:
        LLMBackend: The constructed backend instance.

    Raises:
        ValueError: If ``config.llm_provider`` is not a known provider.
    """
    if config.llm_provider == "openai":
        from cognition.backends.openai import OpenAIBackend
        return OpenAIBackend(config)

    if config.llm_provider == "vertexai":
        from cognition.backends.vertexai import VertexAIBackend
        return VertexAIBackend(config)

    raise ValueError(
        f"Unknown LLM provider '{config.llm_provider}'. "
        f"Valid options: 'openai', 'vertexai'."
    )
