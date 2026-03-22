"""
Google VertexAI / Gemini LLM backend.

Wraps ``langchain-google-genai`` with VertexAI authentication.
Requires a GCP project with the Generative AI API enabled and
``llm_vertex_project`` / ``llm_vertex_location`` set in config.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator

from cognition.backends import LLMBackend
from cognition.media import convert_content_list_vertexai
from cognition.types import StreamChunk, ToolCall

if TYPE_CHECKING:
    from cognition.config import CognitionConfig

logger = logging.getLogger(__name__)


class VertexAIBackend(LLMBackend):
    """LangChain ``ChatGoogleGenerativeAI`` streaming adapter.

    Args:
        config: Resolved :class:`~cognition.config.CognitionConfig`.
    """

    def __init__(self, config: "CognitionConfig") -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._llm = ChatGoogleGenerativeAI(
            model=config.llm_model,
            vertexai=True,
            project=config.llm_vertex_project,
            location=config.llm_vertex_location,
            temperature=config.llm_temperature,
        )

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> AsyncIterator[StreamChunk]:
        from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

        lc_messages: list[Any] = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content") or ""

            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                if isinstance(content, list):
                    lc_messages.append(
                        HumanMessage(content=convert_content_list_vertexai(content))
                    )
                else:
                    lc_messages.append(HumanMessage(content=content))

        llm = self._llm.bind_tools(tools) if tools else self._llm

        async for chunk in llm.astream(lc_messages):
            if not isinstance(chunk, AIMessageChunk):
                continue
            if chunk.content and isinstance(chunk.content, str):
                yield StreamChunk(delta_text=chunk.content)
            for tc in getattr(chunk, "tool_calls", []):
                yield StreamChunk(
                    tool_call=ToolCall(
                        call_id=tc.get("id", str(uuid.uuid4())),
                        name=tc["name"],
                        args=tc.get("args", {}),
                    )
                )

        yield StreamChunk(is_done=True)
