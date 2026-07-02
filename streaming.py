"""
Functions for easier interaction with the core.
"""

import logging
from typing import AsyncGenerator, AsyncIterator, cast
import zmq.asyncio
from core.config import manager
from global_types import BusMessage, InputMessage, MessageTopic
from output_message import OutputMessage

logger = logging.getLogger(__name__)


class CoreStream:
    """
    Listen to the core and send messages to the core.
    """

    def __init__(self) -> None:
        ctx = zmq.asyncio.Context()
        cfg = manager.get_config()
        self.ctx = ctx
        self.sock_in = ctx.socket(zmq.PUSH)
        self.sock_in.bind(cfg.socket.input)
        self.sock_out = ctx.socket(zmq.SUB)
        self.sock_out.bind(cfg.socket.output)

    def __aiter__(self) -> AsyncIterator[BusMessage]:
        return self

    async def __anext__(self) -> BusMessage:
        while True:
            msg = BusMessage.decoded(await self.sock_out.recv_multipart())
            if msg is None:
                logger.error("Core sent an invalid bus message, ignoring.")
                continue
            return msg

    async def send(self, message: BusMessage):
        """Send message to the core."""
        await self.sock_in.send_multipart(message.encoded())

    def destroy(self) -> None:
        """
        Call when you don't need this object anymore.
        Not calling will cause the zmq context to still linger.
        """
        self.ctx.destroy()


async def send_message(
    message: InputMessage | str,
    finish_reason: list[MessageTopic] | None = None,
    send_full: bool = False,
) -> AsyncGenerator[OutputMessage, None]:
    """
    Send a message to the LLM and stream the answer.

    Args:
        finish_reason:
            If you want a to know why the stream ended you must pass a reference of a list.
            This function will append the finish reason to it.
        send_full:
            Whether to send a full answer once one has appeared from the core.
            Good for knowing when a new message starts without having to rely on tool calls.
    """
    stream = CoreStream()
    try:
        if isinstance(message, str):
            message = InputMessage(body=message)

        await stream.send(message.to_bus())
        async for msg in stream:
            if msg.topic in [MessageTopic.FINISHED, MessageTopic.ABORT]:
                if finish_reason is not None:
                    finish_reason.append(cast(MessageTopic, msg.topic))
                break

            output = OutputMessage.from_bus(msg)
            if output is None:
                logger.debug("Not a stream message, ignoring: %s", msg)
                continue

            if output.full and not send_full:
                logger.debug("Ignoring full message as send_full is False")
                continue

            yield output
    finally:
        stream.destroy()
