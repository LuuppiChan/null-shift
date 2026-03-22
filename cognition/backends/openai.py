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

from cognition.backends import LLMBackend
from cognition.media import convert_content_list_openai
from cognition.types import StreamChunk, ToolCall

if TYPE_CHECKING:
    from cognition.config import CognitionConfig

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
        messages: list[dict],
        tools: list[dict],
    ) -> AsyncIterator[StreamChunk]:
        from langchain_core.messages import (
            AIMessage,
            AIMessageChunk,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        lc_messages: list[Any] = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content") or ""

            if role == "system":
                lc_messages.append(SystemMessage(content=content))

            elif role == "user":
                if isinstance(content, list):
                    lc_messages.append(
                        HumanMessage(content=convert_content_list_openai(content))
                    )
                else:
                    lc_messages.append(HumanMessage(content=content))

            elif role == "assistant":
                tc = m.get("tool_calls")
                if tc:
                    lc_messages.append(
                        AIMessage(
                            content=content or "",
                            tool_calls=[
                                {
                                    "id": t["id"],
                                    "name": t["function"]["name"],
                                    "args": json.loads(
                                        t["function"].get("arguments", "{}")
                                    ),
                                }
                                for t in tc
                            ],
                        )
                    )
                else:
                    lc_messages.append(AIMessage(content=content))

            elif role == "tool":
                lc_messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=m.get("tool_call_id", ""),
                    )
                )

        llm = self._llm.bind_tools(tools) if tools else self._llm
        accumulated_tool_calls: dict[str, dict] = {}

        async for chunk in llm.astream(lc_messages):
            if not isinstance(chunk, AIMessageChunk):
                continue

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

        yield StreamChunk(is_done=True)
