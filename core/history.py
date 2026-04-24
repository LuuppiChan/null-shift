import json
import logging
from pathlib import Path
from typing import Any, Iterable, cast

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)

from core.config import manager
from core.backends import get_backend

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
                    valid = last is None or isinstance(
                        last, (SystemMessage, AIMessage, AIMessageChunk)
                    )
                    if valid:
                        validated.append(msg)
                    else:
                        logger.warning("Invalid HumanMessage at %s", i)

                case AIMessageChunk() | AIMessage():
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
                        isinstance(last, (AIMessage, AIMessageChunk, ToolMessage))
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

        if expected_tool_calls:
            for missing_id in expected_tool_calls:
                validated.append(
                    ToolMessage(
                        content="Error: Tool execution was interrupted or result was lost.",
                        tool_call_id=missing_id,
                    )
                )

        if validated and not isinstance(validated[-1], (AIMessage, AIMessageChunk)):
            logger.info("Keeping history. Appending an AIMessage to close the turn.")
            recovery_prompt = (
                "[System Notification: The previous workflow was abruptly interrupted. "
                "Any pending actions were aborted. "
                "Please seamlessly process the user's newest request without apologizing for the interruption.]"
            )
            validated.append(AIMessage(content=recovery_prompt))

        # Alternatively we could be aggressive, but the dangling tools suck ass so it's better to just add an AI message
        # while validated and not isinstance(validated[-1], (AIMessage, AIMessageChunk)):
        #     popped = validated.pop()
        #     logger.info("Popping a message: %s", type(popped))

        logger.info("History validated.")

        self.messages = validated

    async def trim_history(self, length: int | None = None):
        """
        Trims the history by removing old messages based on either the config or an input argument.
        First message will either be a ToolMessage or HumanMessage.
        """
        cfg = manager.get_config()

        # Determine our threshold and target lengths based on compression settings
        if length is None:
            if cfg.core_history_compression:
                length_threshold = cfg.core_history_compression_threshold
                target_length = cfg.core_history_compression_target_length
            else:
                length_threshold = cfg.core_history_length
                target_length = cfg.core_history_length
        else:
            length_threshold = length
            target_length = length

        logger.info("History length: %s/%s", len(self.messages), length_threshold)
        if len(self.messages) < length_threshold:
            logger.info("No history trimming needed.")
            return

        # Get all possible split positions
        positions: list[int] = []
        for i, msg in enumerate(self.messages):
            if isinstance(msg, HumanMessage):
                positions.append(i)
            # This fucked the system
            # Also adding this feature WILL ALSO degrade the answer quality and make possible infinite loops
            # Future me, trust the past me when I realized the agent was looping for no reason for 30 minutes.
            # elif isinstance(msg, (AIMessage, AIMessageChunk)) and cfg.core_history_compression:
            #     positions.append(i)

        # removing this also fucked the trimming
        if positions:
            positions.pop()

        # VertexAI is strict about human messages.
        # It crashes if the first message isn't human message.
        if not positions:
            logger.info(
                "Cannot trim due to the shortage of human messages (%s).",
                len(positions),
            )
            return

        ok_pos = 0
        # Iterate through HumanMessage positions from oldest to newest.
        # We want to find the first cut-off point that leaves us with a history length
        # smaller than our target `length`.
        for pos in positions:
            # instead of actually creating lists let's use
            # candidate represents how many messages would be left if we split at `pos` pure numbers
            candidate = len(self.messages) - pos
            # If we're shorter than target we break
            if candidate <= target_length:
                # The trimming was way too harsh
                # I hope this makes it trim less hard.
                continue
                # Do the actual history change
                popped_messages = self.messages[:pos]
                self.messages = self.messages[pos:]
                break  # Exiting here skips the `else` block below
            # Keep track of the last valid cut-off point we checked
            ok_pos = pos
        else:
            popped_messages = self.messages[:ok_pos]
            # The `else` block on a for-loop executes ONLY if the loop finishes without hitting `break`.
            # This happens if a single back-and-forth (e.g., lots of tool calls)
            # is longer than our target `length`. We couldn't get it under the limit,
            # so we just trim as much as we safely can using the last known valid position.
            self.messages = self.messages[ok_pos:]

        # Skip if there's nothing to compress
        if not popped_messages:
            logger.info("No messages were popped.")
            return

        # Prevent pointless API calls if compression is impossible
        #  Since summarize_history returns at least 1 (often 2) messages,
        # popping <= 2 messages will not shrink the history.
        if len(popped_messages) <= 2 and cfg.core_history_compression:
            self.messages = popped_messages + self.messages
            logger.warning(
                "Cannot trim history: The sequence of recent tool/AI messages is too long to safely compress."
            )
            return

        # Skip if compression is disabled in config
        if not cfg.core_history_compression:
            logger.info(
                "History trimmed %s/%s (Compression Disabled)",
                len(self.messages),
                length,
            )
            return

        if cfg.core_history_compression:
            try:
                summary = await summarize_history(popped_messages, self.messages[0])
                self.messages = summary + self.messages
            except Exception as e:
                logger.warning("Didn't compress the history: %s", e)
                # Because I don't want to lose my history on an error.
                self.messages = popped_messages + self.messages

        # Move this here because it's more logical to be here instead of before pop messages return
        logger.info("History trimmed %s/%s", len(self.messages), length_threshold)


async def summarize_history(
    messages: list[BaseMessage], message_after_summary: BaseMessage
) -> list[BaseMessage]:
    """Returns the some messages with the summary ready to be appended to the messages."""
    cfg = manager.get_config()
    md = messages_to_md(messages)
    system = cfg.core_history_compression_prompt
    human = HumanMessage(md)
    msgs = [SystemMessage(system), human]
    llm = get_backend()
    llm.set_model(cfg.core_history_compression_model)
    llm.set_thinking(None)
    full = None
    logger.info("Summarizing history...")
    for _ in range(3):
        try:
            async for chunk in llm.stream(msgs, None):
                full = chunk if full is None else cast(AIMessage, full + chunk)
            break
        except Exception as e:
            logger.error("Creating summary failed: %s", e)

    llm.reset_customs()

    if full and content_exists(full.content):
        summary = HumanMessage(full.content)
        if messages and isinstance(message_after_summary, (AIMessage, AIMessageChunk)):
            return [summary]
        else:
            ai = AIMessage("Summary acknowledged.")
            return [summary, ai]
    else:
        logger.error("Error creating summary message. Keeping history.")
        raise Exception("Summary exception")


def messages_to_md(messages: list[BaseMessage]) -> str:
    parts = []
    for msg in messages:
        parts.append(message_to_md(msg))

    return "\n\n\n".join(parts)


def message_to_md(message: BaseMessage) -> str:
    """Return empty string if unknown message. (such as SystemMessage)"""
    match message:
        case AIMessage() | AIMessageChunk():
            return f"# AI Message\n{msg_dict_to_msg(message.content)}"
        case HumanMessage():
            return f"# Human Message\n{msg_dict_to_msg(message.content)}"
        case ToolMessage():
            return f"# Tool Message\n{msg_dict_to_msg(message.content)}"
        case _:
            return ""


def msg_dict_to_msg(msg: str | list[str | dict[str, Any]]) -> str:
    if isinstance(msg, str):
        return msg

    cumulated = []
    for block in msg:
        if isinstance(block, str):
            cumulated.append(block)
            continue

        msg_type = block.get("type")
        text = block.get("text")
        mtype = block.get("mime_type")
        cumulated.append(f"## Content type: {msg_type}")
        cumulated.append(
            f"{text or mtype or 'Cannot show this type of content in text'}\n"
        )

    return "\n".join(cumulated)


def content_exists(msg: str | list[str | dict[str, Any]]) -> bool:
    """Whether content has stuff in it."""
    if isinstance(msg, str):
        return bool(msg)

    for block in msg:
        if isinstance(block, str):
            return True

        text = block.get("text", block.get("reasoning"))
        if text:
            return True

    return False


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
