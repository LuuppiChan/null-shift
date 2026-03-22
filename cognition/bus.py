"""
ZMQ bus abstraction for the Cognition node.

Cognition **binds** both sockets. All other nodes connect to these addresses.

- PULL socket: receives all inbound messages (inputs, action results, commands)
- PUB socket: broadcasts all outbound messages (tokens, state events, action requests)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator

import zmq
import zmq.asyncio

from cognition.config import CognitionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BusMessage:
    """A single message received on the input PULL socket."""

    topic: str
    payload: dict


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------


class CognitionBus:
    """
    Owns and manages both ZMQ sockets on behalf of the Cognition node.

    The bus uses the multipart-frame convention for the input PULL socket:
    ``[topic_bytes, json_bytes]``. The output PUB socket uses the same
    framing so that Synthesis nodes can use ZMQ topic-prefix filtering.
    """

    def __init__(self, config: CognitionConfig) -> None:
        self._config = config
        self._ctx: zmq.asyncio.Context = zmq.asyncio.Context()

        # PULL — inbound
        self._pull: zmq.asyncio.Socket = self._ctx.socket(zmq.PULL)
        self._pull.bind(config.zmq_input_bind)
        logger.info("PULL socket bound to %s", config.zmq_input_bind)

        # PUB — outbound
        self._pub: zmq.asyncio.Socket = self._ctx.socket(zmq.PUB)
        self._pub.bind(config.zmq_output_bind)
        logger.info("PUB socket bound to %s", config.zmq_output_bind)

    # ------------------------------------------------------------------
    # Inbound
    # ------------------------------------------------------------------

    async def listen(self) -> AsyncIterator[BusMessage]:
        """Yield :class:`BusMessage` objects as they arrive on the PULL socket.

        Messages must be sent as two-frame multipart messages:
        ``[topic: bytes, payload: utf-8-encoded JSON bytes]``.

        Yields:
            BusMessage: The decoded topic and payload dict.
        """
        while True:
            try:
                frames = await self._pull.recv_multipart()
            except zmq.ZMQError as exc:
                if exc.errno == zmq.ETERM:
                    logger.info("ZMQ context terminated — stopping listen loop.")
                    return
                logger.error("ZMQ recv error: %s", exc)
                await asyncio.sleep(0.1)
                continue

            if len(frames) != 2:
                logger.warning("Malformed message: expected 2 frames, got %d.", len(frames))
                continue

            topic_bytes, payload_bytes = frames
            topic = topic_bytes.decode("utf-8", errors="replace")

            try:
                payload: dict = json.loads(payload_bytes)
            except json.JSONDecodeError as exc:
                logger.warning("Could not decode payload for topic %r: %s", topic, exc)
                continue

            yield BusMessage(topic=topic, payload=payload)

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def publish(self, topic: str, payload: dict) -> None:
        """Publish a message to the PUB socket.

        Args:
            topic: Dot-namespaced topic string (e.g. ``"assistant.stream"``).
            payload: JSON-serialisable dict to send as the message body.
        """
        try:
            await self._pub.send_multipart(
                [
                    topic.encode("utf-8"),
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                ]
            )
        except zmq.ZMQError as exc:
            logger.error("ZMQ publish error on topic %r: %s", topic, exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Gracefully close both sockets and terminate the ZMQ context."""
        logger.info("Closing ZMQ bus...")
        self._pull.close(linger=0)
        self._pub.close(linger=0)
        self._ctx.term()
        logger.info("ZMQ bus closed.")
