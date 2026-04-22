"""
OpenAI-compatible LLM backend.

Works with any endpoint that speaks OpenAI's API, including:
- OpenAI (gpt-4o, gpt-4o-mini, o3-mini, ...)
- Ollama  (local models via OpenAI shim)
- LM Studio, vLLM, etc.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator

from backends import LLMBackend
from custom_types import StreamChunk, ToolCall
from langchain_core.messages import BaseMessage
from media import convert_content_list_openai

if TYPE_CHECKING:
    from config import CognitionConfig

logger = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    """LangChain ``ChatOpenAI`` streaming adapter.

    Args:
        config: Resolved :class:`~cognition.config.CognitionConfig`.
    """

    def __init__(self, config: "CognitionConfig") -> None:
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=config.llm_model,
            api_key=config.llm_api_key,  # type: ignore[arg-type]
            base_url=config.llm_base_url,
            temperature=config.llm_temperature,
            streaming=True,
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

        lc_messages = []
        for m in messages:
            if isinstance(m, HumanMessage) and isinstance(m.content, list):
                lc_messages.append(
                    HumanMessage(content=convert_content_list_openai(m.content))
                )
            else:
                lc_messages.append(m)

        if tools:
            logger.info("Binding %d tools to OpenAI LLM.", len(tools))
            llm = self._llm.bind_tools(tools)
        else:
            llm = self._llm
        accumulated_tool_calls: dict[str, dict] = {}

        finish_reason = None
        async for chunk in llm.astream(lc_messages):
            if not isinstance(chunk, AIMessageChunk):
                continue

            if chunk.response_metadata and chunk.response_metadata.get("finish_reason"):
                finish_reason = chunk.response_metadata.get("finish_reason").upper()

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

        if finish_reason == "TOOL_CALLS":
            finish_reason = "TOOL_CALL"

        yield StreamChunk(is_done=True, finish_reason=finish_reason)
