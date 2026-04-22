"""
Conversation history manager.

Stores the conversation as LangChain typed message objects (SystemMessage,
HumanMessage, AIMessage, ToolMessage).

Multimodal content is stored using the **mime_type** convention understood by
both VertexAI and the V1 assistant. LLM backends are responsible for
converting these parts to their provider-specific format when sending
messages. Format for media parts::

    {"mime_type": "image/png", "data": "<base64>"}

Text parts use::

    {"type": "text", "text": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from config import CognitionConfig
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)

logger = logging.getLogger(__name__)


class HistoryManager:
    """Manages the raw conversation history list.

    Args:
        config: Resolved :class:`~cognition.config.CognitionConfig` instance.
    """

    def __init__(self, config: CognitionConfig) -> None:
        self._config = config
        self._messages: list[BaseMessage] = []

    # ------------------------------------------------------------------
    # Append helpers
    # ------------------------------------------------------------------

    def append_system(self, text: str) -> None:
        """Replace or set the leading system message.

        Always kept at index 0. Calling this a second time replaces the
        previous system message so the history stays well-formed.

        Args:
            text: System prompt text.
        """
        if self._messages and isinstance(self._messages[0], SystemMessage):
            self._messages[0] = SystemMessage(content=text)
        else:
            self._messages.insert(0, SystemMessage(content=text))

    def append_user(
        self,
        text: str,
        media_parts: list[dict] | None = None,
    ) -> None:
        """Append a user message, optionally with multimodal media parts.

        Media parts use the mime_type format::

            [{"mime_type": "image/png", "data": "<base64>"}, ...]

        If ``media_parts`` is provided the final content is a list where the
        text is appended as the last element in the standard text-part format::

            [...media_parts, {"type": "text", "text": text}]

        Backends translate mime_type parts to their provider format on send.

        Args:
            text: The plain-text body of the user's message.
            media_parts: Optional list of media dicts with ``mime_type`` and
                ``data`` fields.
        """
        if media_parts:
            content: Any = list(media_parts) + [{"type": "text", "text": text}]
        else:
            content = text

        self._messages.append(HumanMessage(content=content))

    def append_assistant(self, text: str) -> None:
        """Append an assistant (LLM) response message.

        Args:
            text: The full text of the assistant's response.
        """
        self._messages.append(AIMessage(content=text))

    def append_tool_call(
        self,
        call_id: str,
        name: str,
        args: dict,
        content: str | None = None,
    ) -> None:
        """Append the assistant's tool-call request to history.

        In V1 logs, tool-call messages sometimes carry spoken text alongside
        the tool invocation. The ``content`` field preserves that behaviour —
        it can be ``None`` (the common case) or the text spoken while the
        tool was requested.

        Args:
            call_id: Unique identifier for this tool invocation.
            name: Name of the function being called.
            args: Dict of arguments being passed.
            content: Optional text content spoken alongside the tool call.
        """
        self._messages.append(
            AIMessage(
                content=content or "",
                tool_calls=[
                    {
                        "id": call_id,
                        "name": name,
                        "args": args,
                    }
                ],
            )
        )

    def append_tool_result(self, call_id: str, result: str) -> None:
        """Append the result of a tool call.

        Args:
            call_id: Must match the ``call_id`` from :meth:`append_tool_call`.
            result: String result returned by the tool or Action Node.
        """
        self._messages.append(
            ToolMessage(
                tool_call_id=call_id,
                content=result,
            )
        )

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def trim(self) -> None:
        """Trim the history to ``llm_max_history`` messages, preserving pairs.

        The system message at index 0 is always kept. Tool call / result
        pairs are never split across the trim boundary.
        """
        max_messages = self._config.llm_max_history
        if len(self._messages) <= max_messages:
            return

        system_msg = (
            self._messages[0]
            if self._messages and isinstance(self._messages[0], SystemMessage)
            else None
        )
        rest = self._messages[1:] if system_msg else list(self._messages)

        remove_count = len(self._messages) - max_messages
        cutoff = remove_count

        while cutoff < len(rest):
            msg = rest[cutoff]
            is_tool_result = isinstance(msg, ToolMessage)
            prev_has_tool_call = (
                cutoff > 0
                and isinstance(rest[cutoff - 1], AIMessage)
                and rest[cutoff - 1].tool_calls
            )
            if is_tool_result or prev_has_tool_call:
                cutoff += 1
            else:
                break

        self._messages = ([system_msg] if system_msg else []) + rest[cutoff:]
        logger.info("Trimmed history to %d messages.", len(self._messages))

    def sanitize(self) -> None:
        """Remove structurally invalid message sequences.

        Rules enforced:
        - Only one system message, always at index 0.
        - No consecutive user messages (last one wins).
        - History after the system message must start with a user message.
        """
        if not self._messages:
            return

        sanitized: list[BaseMessage] = []
        for i, msg in enumerate(self._messages):
            if i == 0:
                sanitized.append(msg)
                continue

            if isinstance(msg, SystemMessage):
                logger.debug("Dropping mid-history system message at index %d.", i)
                continue

            prev = sanitized[-1] if sanitized else None

            if (
                prev
                and isinstance(msg, HumanMessage)
                and isinstance(prev, HumanMessage)
            ):
                logger.debug("Merging consecutive user messages at index %d.", i)
                sanitized[-1] = msg
                continue

            sanitized.append(msg)

        system_end = 1 if (sanitized and isinstance(sanitized[0], SystemMessage)) else 0
        while len(sanitized) > system_end and not isinstance(
            sanitized[system_end], HumanMessage
        ):
            logger.debug(
                "Removing leading '%s' message to enforce system→user ordering.",
                sanitized[system_end].__class__.__name__,
            )
            sanitized.pop(system_end)

        self._messages = sanitized

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def save(self) -> None:
        """Persist the history to disk as JSON (async, non-blocking).

        Also calls :meth:`sanitize` before writing so the file is always
        well-formed, even if a tool has manipulated the history mid-turn.
        The system message is excluded from the saved file — it is
        regenerated fresh on the next startup.
        """
        self.sanitize()

        path = Path(self._config.path_history)
        path.parent.mkdir(parents=True, exist_ok=True)

        to_save = [m for m in self._messages if not isinstance(m, SystemMessage)]

        try:
            await asyncio.to_thread(
                path.write_text,
                json.dumps(
                    [message_to_dict(m) for m in to_save], indent=2, ensure_ascii=False
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed to save history: %s", exc)

    def load(self) -> None:
        """Load persisted history from disk.

        The system message is not stored on disk — callers should call
        :meth:`append_system` after loading to inject the current prompt.
        """
        path = Path(self._config.path_history)
        if not path.exists():
            return

        try:
            raw: list[dict] = json.loads(path.read_text(encoding="utf-8"))

            loaded = messages_from_dict(raw)

            self._messages = [m for m in loaded if not isinstance(m, SystemMessage)]
            logger.info("Loaded %d messages from history.", len(self._messages))
        except Exception as exc:
            logger.error("Failed to load history from %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def to_messages(self) -> list[BaseMessage]:
        """Return a shallow copy of the current message list.

        Returns:
            list[BaseMessage]: Messages ready to pass to an LLM backend.
        """
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)
