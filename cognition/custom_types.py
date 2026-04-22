"""
Shared primitive dataclasses for the Cognition node.

These are kept separate so both backends and the Vector orchestrator can
import them without creating circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ToolCall:
    """A tool invocation requested by the LLM.

    Attributes:
        call_id: Unique ID for correlating results.
        name: Tool function name.
        args: Decoded argument dictionary.
    """

    call_id: str
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class StreamChunk:
    """A single chunk produced by the LLM streaming interface.

    Attributes:
        delta_text: Incremental text token, or ``None`` if this chunk carries
            a tool call.
        tool_call: A complete tool-call request, or ``None`` for text chunks.
        is_done: ``True`` signals end of the stream.
        finish_reason: Upstream API reason for stopping (e.g. \"STOP\", \"TOOL_CALL\").
    """

    delta_text: str | None = None
    tool_call: ToolCall | None = None
    is_done: bool = False
    finish_reason: str | None = None
