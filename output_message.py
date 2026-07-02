from enum import StrEnum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ValidationError

from global_types import BusMessage, MessageTopic


class Event(StrEnum):
    ABORT = "abort"


class OutputMessage(BaseModel):
    """Output stream message."""

    text: str | None = None
    reasoning: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_result: str | dict[str, Any] | list[dict[str, Any]] | list[str] | None = None
    tool_args: dict[str, Any] | None = None
    stream_id: str | Literal["default"] = "default"
    event: Event | str | None = None
    full: bool = False

    def to_bus(self, topic: MessageTopic | str) -> BusMessage:
        return BusMessage(
            topic=topic,
            payload=self.model_dump(exclude_none=True, exclude_defaults=True),
        )

    @staticmethod
    def from_bus(message: BusMessage) -> Optional["OutputMessage"]:
        try:
            out = OutputMessage.model_validate(message.payload)
            if message.topic == MessageTopic.FULL:
                out.full = True
            return out
        except ValidationError:
            return None

    def is_stream(self) -> bool:
        """Whether the message is part of an LLM text or reasoning, but not full."""
        return (self.text is not None or self.reasoning is not None) and not self.full

    def is_tool_call(self) -> bool:
        """Whether this message is a tool call (not a tool result)"""
        return (
            self.tool_name is not None
            or self.tool_call_id is not None
            or self.tool_args is not None
        )

    def is_tool_result(self) -> bool:
        """Whether this message is a tool result (not a tool call)"""
        return self.tool_result is not None

    def __add__(self, other: "OutputMessage") -> "OutputMessage":
        new = self.model_copy(deep=True)
        new.text = (
            None
            if self.text is None and other.text is None
            else (self.text or "") + (other.text or "")
        )
        new.reasoning = (
            None
            if self.reasoning is None and other.reasoning is None
            else (self.reasoning or "") + (other.reasoning or "")
        )
        return new

    def __iadd__(self, other: "OutputMessage") -> "OutputMessage":
        self.text = (
            None
            if self.text is None and other.text is None
            else (self.text or "") + (other.text or "")
        )
        self.reasoning = (
            None
            if self.reasoning is None and other.reasoning is None
            else (self.reasoning or "") + (other.reasoning or "")
        )
        return self
