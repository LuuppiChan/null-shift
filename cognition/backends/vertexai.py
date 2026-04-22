"""
Google VertexAI / Gemini LLM backend.

Wraps ``langchain-google-genai`` with VertexAI authentication.
Requires a GCP project with the Generative AI API enabled and
``llm_vertex_project`` / ``llm_vertex_location`` set in config.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator

from backends import LLMBackend
from custom_types import StreamChunk, ToolCall
from langchain_core.messages import BaseMessage
from media import convert_content_list_vertexai

if TYPE_CHECKING:
    from config import CognitionConfig

logger = logging.getLogger(__name__)


class VertexAIBackend(LLMBackend):
    """LangChain ``ChatGoogleGenerativeAI`` streaming adapter.

    Args:
        config: Resolved :class:`~cognition.config.CognitionConfig`.
    """

    def __init__(self, config: "CognitionConfig") -> None:
        from google.auth import default

        self._config = config
        self._credentials, _project = default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )  # (credentials, project) we want only the credentials
        self._reload_backend()

    def _reload_backend(self):
        """Refreshes the backend based on current configuration and keeps the api key fresh."""
        from google.auth.transport.requests import Request
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not self._credentials.valid:
            logger.info("VertexAI Credentials refreshed")
            self._credentials.refresh(Request())

        self._llm = ChatGoogleGenerativeAI(
            model=self._config.llm_model,
            vertexai=True,
            project=self._config.llm_vertex_project,
            location=self._config.llm_vertex_location,
            credentials=self._credentials,
            temperature=self._config.llm_temperature,
        )

    async def stream(
        self,
        messages: list[BaseMessage],
        tools: list[Any],
    ) -> AsyncIterator[StreamChunk]:
        from langchain_core.messages import (
            AIMessageChunk,
            HumanMessage,
        )

        # refresh the backend
        self._reload_backend()

        lc_messages = []
        for m in messages:
            if isinstance(m, HumanMessage) and isinstance(m.content, list):
                lc_messages.append(
                    HumanMessage(content=convert_content_list_vertexai(m.content))
                )
            else:
                lc_messages.append(m)

        llm = self._llm.bind_tools(tools) if tools else self._llm
        if tools:
            logger.info("Binding %d tools to VertexAI LLM.", len(tools))
            llm = self._llm.bind_tools(tools)
        else:
            llm = self._llm

        accumulated_tool_calls: dict[str, dict] = {}
        finish_reason = None

        async for chunk in llm.astream(lc_messages):
            if not isinstance(chunk, AIMessageChunk):
                continue

            if chunk.response_metadata and chunk.response_metadata.get("finish_reason"):
                raw_reason = str(chunk.response_metadata.get("finish_reason")).upper()
                if "STOP" in raw_reason:
                    finish_reason = "STOP"
                elif "TOOL_CALL" in raw_reason or "FUNCTION_CALL" in raw_reason:
                    finish_reason = "TOOL_CALL"
                elif "MAX_TOKENS" in raw_reason or "LENGTH" in raw_reason:
                    finish_reason = "LENGTH"
                else:
                    finish_reason = raw_reason

            if chunk.content and isinstance(chunk.content, str):
                yield StreamChunk(delta_text=chunk.content)

            for tc in chunk.tool_call_chunks or []:
                key = str(tc.get("index", 0))
                if key not in accumulated_tool_calls:
                    accumulated_tool_calls[key] = {
                        "id": tc.get("id", ""),
                        "name": tc.get("name", ""),
                        "args_str": "",
                    }
                accumulated_tool_calls[key]["args_str"] += tc.get("args", "") or ""
                if tc.get("id"):
                    accumulated_tool_calls[key]["id"] = tc["id"]
                if tc.get("name"):
                    accumulated_tool_calls[key]["name"] = tc["name"]

        for tc_data in accumulated_tool_calls.values():
            try:
                args = json.loads(tc_data["args_str"] or "{}")
            except json.JSONDecodeError:
                args = {}
            yield StreamChunk(
                tool_call=ToolCall(
                    call_id=tc_data["id"] or str(uuid.uuid4()),
                    name=tc_data["name"],
                    args=args,
                )
            )

        yield StreamChunk(is_done=True, finish_reason=finish_reason)
