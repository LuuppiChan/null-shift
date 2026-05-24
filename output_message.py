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

    def to_bus(self, topic: MessageTopic | str) -> BusMessage:
        return BusMessage(
            topic=topic,
            payload=self.model_dump(exclude_none=True, exclude_defaults=True),
        )

    @staticmethod
    def from_bus(message: BusMessage) -> Optional["OutputMessage"]:
        try:
            return OutputMessage.model_validate(message.payload)
        except ValidationError:
            return None
