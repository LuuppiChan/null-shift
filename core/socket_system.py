import asyncio
import json
import logging
import zmq.asyncio
from zmq.asyncio import Socket

from global_types import BusMessage
from core.config import manager

logger = logging.getLogger(__name__)


class SocketOut:
    """Wrapper for output socket."""

    def __init__(self) -> None:
        self._ctx = zmq.asyncio.Context()
        self._socket: Socket = self._ctx.socket(zmq.PUB)
        self._socket.bind(manager.get_config().zmq_output_bind)

        self._lock = asyncio.Lock()  # Prevent race conditions during rebind
        self._current_bind = manager.get_config().zmq_output_bind

    async def _ensure_rebind(self):
        new_bind = manager.get_config().zmq_output_bind
        # Double-checked locking: The first check avoids lock overhead during
        # normal operation; the second check (inside the lock) prevents
        # race conditions if multiple tasks trigger a rebind simultaneously.
        if new_bind != self._current_bind:
            async with self._lock:
                new_bind = manager.get_config().zmq_output_bind
                if new_bind != self._current_bind:
                    try:
                        logger.info("Socket updated, rebinding...")
                        self._socket.unbind(self._current_bind)
                        await asyncio.sleep(0.1)
                        self._socket.bind(new_bind)
                        self._current_bind = new_bind
                        logger.info("Rebound output socket to %s", new_bind)
                    except zmq.ZMQError as e:
                        logger.critical("Failed to rebind socket: %s", e)

    async def send(self, message: BusMessage):
        """Send a message to the out socket."""
        await self._ensure_rebind()
        logger.info("Sending message to topic: %s", message.topic)
        logger.debug("Full BusMessage object: %s", message)
        await self._socket.send_multipart(
            [
                message.topic.encode(),
                json.dumps(message.payload).encode(),
            ]
        )


socket_out = SocketOut()
