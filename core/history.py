import json
import logging
from pathlib import Path
from typing import Iterable, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)

from core.config import manager

logger = logging.getLogger(__name__)


class History:
    def __init__(self) -> None:
        # history without
        self.messages: list[BaseMessage] = []

    def with_system_message(self, system_message: SystemMessage) -> list[BaseMessage]:
        """Get the history with your system message."""
        # Shallow list in python, cheap
        return [system_message] + self.messages

    def append(self, message: BaseMessage):
        """Append a message to the history."""
        self.messages.append(message)

    def extend(self, messages: Iterable[BaseMessage]):
        """Extend the history with these messages."""
        self.messages.extend(messages)

    def save(self):
        """Save history to the file."""
        config = manager.get_config()
        path = Path(config.core_history_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

        history_data = messages_to_dict(self.messages)

        path.write_text(json.dumps(history_data, indent=2), encoding="utf-8")

    def load(self):
        """Loads history from the file."""
        config = manager.get_config()
        path = Path(config.core_history_path).expanduser().resolve()
        if not path.exists():
            logger.warning(
                "Cannot load history: it doesn't exist at %s. Loading empty history.",
                path,
            )
            self.messages = []
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.messages = messages_from_dict(data)
            logger.info("Loaded %s messages from history.", len(self.messages))
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(
                "Failed to parse history file: %s. Initializing empty history.", e
            )
            self.messages = []

    def validate_history(self):
        """
        Should be called when the last message is an LLM message.
        Validates and fixes errors in the history by removing messages.

        Currently first candidate wins.
        So for example there are 2 human messages the first one will be added.
        """
        validated: list[BaseMessage] = []
        msgs = self.messages
        expected_tool_calls: set[str] = set()
        for i, msg in enumerate(msgs):
            last = validated[-1] if validated else None

            match msg:
                case SystemMessage():
                    valid = last is None
                    if valid:
                        logger.warning("System message in conversation.")
                        # No because the system message is dynamic and doesn't belong to the messages.
                        # validated.append(msg)
                    else:
                        logger.warning(
                            "System message in the middle of the conversation at %s, dropping.",
                            i,
                        )

                case HumanMessage():
                    valid = last is None or isinstance(last, (SystemMessage, AIMessage))
                    if valid:
                        validated.append(msg)
                    else:
                        logger.warning("Invalid HumanMessage at %s", i)

                case AIMessage():
                    valid = isinstance(last, (ToolMessage, HumanMessage))
                    if valid:
                        if expected_tool_calls:
                            logger.warning(
                                "Dangling tool calls: %s", expected_tool_calls
                            )
                            for missing_id in expected_tool_calls:
                                validated.append(
                                    ToolMessage(
                                        content="Error: Tool execution was interrupted or result was lost.",
                                        tool_call_id=missing_id,
                                    )
                                )
                            # no clearing needed as it's overwritten by the statement under this

                        expected_tool_calls = (
                            {cast(str, tc["id"]) for tc in msg.tool_calls}
                            if msg.tool_calls
                            else set()
                        )

                        validated.append(msg)
                    else:
                        logger.warning("Invalid AIMessage at %s", i)

                case ToolMessage(tool_call_id=tid):
                    valid = (
                        isinstance(last, (AIMessage, ToolMessage))
                        and tid in expected_tool_calls
                    )
                    if valid:
                        validated.append(msg)
                        expected_tool_calls.remove(tid)
                    else:
                        logger.warning("Invalid ToolMessage at %s", i)

                case _:
                    logger.warning(
                        "Unknown message type found at %s: %s, dropping.", i, type(msg)
                    )

        while validated and not isinstance(validated[-1], AIMessage):
            validated.pop()

        self.messages = validated

    def trim_history(self, length: int | None = None):
        """
        Trims the history by removing old messages based on either the config or an input argument.
        First message will either be a ToolMessage or HumanMessage.
        """
        if length is None:
            length = manager.get_config().core_history_length

        logger.info("History length: %s/%s", len(self.messages), length)
        if len(self.messages) < length:
            logger.info("No history trimming needed.")
            return

        # Get all human messages
        human_messages = 0
        positions: list[int] = []
        for i, msg in enumerate(self.messages):
            if isinstance(msg, HumanMessage):
                human_messages += 1
                positions.append(i)

        # Pop the last so that we don't accidentally delete all human messages
        if positions:
            positions.pop()

        # VertexAI is strict about human messages.
        # It crashes if the first message isn't human message.
        if human_messages <= 1:
            logger.info(
                "Cannot trim due to the shortage of human messages (%s).",
                human_messages,
            )
            return

        ok_pos = 0
        # Now we trim
        for pos in positions:
            # instead of actually creating lists let's use pure numbers
            candidate = len(self.messages) - pos
            # If we're shorter than target we break
            if candidate < length:
                # Do the actual history change
                self.messages = self.messages[pos:]
                break

            ok_pos = pos
        else:
            self.messages = self.messages[ok_pos:]

        logger.info("History trimmed %s/%s", len(self.messages), length)


def test():
    """Test the history validation"""
    history = History()
    messages = [
        HumanMessage("Hi"),
        AIMessage(
            "Idk need tool", tool_calls=[ToolCall(name="foo", args=dict(), id="123")]
        ),
        ToolMessage("Polite greeting", tool_call_id="123"),
        AIMessage("Hi"),
    ]
    history.extend(messages.copy())
    history.validate_history()
    assert history.messages == messages

    messages = [
        HumanMessage("Hi"),
        ToolMessage("Polite greeting", tool_call_id="123"),
        AIMessage("Hi"),
    ]
    valid_messages = [
        HumanMessage("Hi"),
        AIMessage("Hi"),
    ]

    history.messages = messages.copy()
    history.validate_history()
    assert history.messages == valid_messages

    messages = [
        HumanMessage("Hi"),
        SystemMessage("Be nise"),
        AIMessage(
            "Idk need tool", tool_calls=[ToolCall(name="foo", args=dict(), id="123")]
        ),
        ToolMessage("Polite greeting", tool_call_id="123"),
        AIMessage("Hi"),
        HumanMessage("Hi"),
        AIMessage("Hi"),
        ToolMessage("Polite greeting", tool_call_id="foo"),
        ToolMessage("Polite greeting", tool_call_id="123"),
        AIMessage("Hi"),
        HumanMessage("Hi"),
        AIMessage(
            "Idk need tool",
            tool_calls=[
                ToolCall(name="foo", args=dict(), id="-1"),
                ToolCall(name="foo", args=dict(), id="0"),
                ToolCall(name="foo", args=dict(), id="1"),
            ],
        ),
        ToolMessage("Polite greeting", tool_call_id="-1"),
        ToolMessage("Polite greeting", tool_call_id="1"),
        AIMessage("Hi"),
    ]
    valid_messages = [
        HumanMessage("Hi"),
        AIMessage(
            "Idk need tool", tool_calls=[ToolCall(name="foo", args=dict(), id="123")]
        ),
        ToolMessage("Polite greeting", tool_call_id="123"),
        AIMessage("Hi"),
        HumanMessage("Hi"),
        AIMessage("Hi"),
        HumanMessage("Hi"),
        AIMessage(
            "Idk need tool",
            tool_calls=[
                ToolCall(name="foo", args=dict(), id="-1"),
                ToolCall(name="foo", args=dict(), id="0"),
                ToolCall(name="foo", args=dict(), id="1"),
            ],
        ),
        ToolMessage("Polite greeting", tool_call_id="-1"),
        ToolMessage("Polite greeting", tool_call_id="1"),
        ToolMessage(
            "Error: Tool execution was interrupted or result was lost.",
            tool_call_id="0",
        ),
        AIMessage("Hi"),
    ]

    history.messages = messages
    history.validate_history()
    assert history.messages == valid_messages

    messages = [
        HumanMessage("Hi"),
        AIMessage("Hi"),
        HumanMessage("Hi"),
        HumanMessage("Hi"),
        HumanMessage("Hi"),
        HumanMessage("Hi"),
    ]
    valid_messages = [
        HumanMessage("Hi"),
        AIMessage("Hi"),
    ]
    history.messages = messages
    history.validate_history()
    assert history.messages == valid_messages

    messages = [
        HumanMessage("Hi"),
        AIMessage("Hi"),
        HumanMessage("Hi"),
        AIMessage("Hi"),
        AIMessage("Hi"),
        AIMessage("Hi"),
        AIMessage("Hi"),
        AIMessage("Hi"),
        AIMessage("Hi"),
        AIMessage("Hi"),
        AIMessage("Hi"),
        AIMessage("Hi"),
        AIMessage("Hi"),
        AIMessage("Hi"),
    ]
    valid_messages = [
        HumanMessage("Hi"),
        AIMessage("Hi"),
        HumanMessage("Hi"),
        AIMessage("Hi"),
    ]
    history.messages = messages
    history.validate_history()
    assert history.messages == valid_messages


if __name__ == "__main__":
    test()
