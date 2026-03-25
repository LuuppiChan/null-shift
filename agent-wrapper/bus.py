import asyncio
import json
import logging
from typing import AsyncGenerator, Tuple

import zmq
import zmq.asyncio

from .config import AgentConfig

logger = logging.getLogger(__name__)

class WrapperBus:
    """Manages the 4 ZMQ sockets for the Agent Wrapper."""
    
    def __init__(self, config: AgentConfig):
        self._ctx = zmq.asyncio.Context()
        self._config = config
        
        # 1. Core Input (PUSH)
        self.core_push = self._ctx.socket(zmq.PUSH)
        self.core_push.connect(config.zmq_core_input)
        
        # 2. Core Output (SUB)
        self.core_sub = self._ctx.socket(zmq.SUB)
        self.core_sub.connect(config.zmq_core_output)
        self.core_sub.subscribe(b"assistant.stream.done")
        
        # 3. Native Tool Signals (PULL)
        self.signal_pull = self._ctx.socket(zmq.PULL)
        self.signal_pull.bind(config.zmq_signals_bind)
        
        # 4. UI/Perception Input (PULL)
        self.ui_pull = self._ctx.socket(zmq.PULL)
        self.ui_pull.bind(config.zmq_ui_bind)

        logger.info(f"Bus initialized. UI listening on {config.zmq_ui_bind}, Signals on {config.zmq_signals_bind}")

    async def listen_ui(self) -> AsyncGenerator[dict, None]:
        """Yield parsed JSON messages from the UI/Perception socket."""
        while True:
            try:
                # Expecting bipartite or single frame depending on UI sender. Handle both.
                frames = await self.ui_pull.recv_multipart()
                payload_bytes = frames[-1]
                payload = json.loads(payload_bytes)
                yield payload
            except Exception as e:
                logger.error(f"Error parsing UI input: {e}")

    async def listen_signals(self) -> AsyncGenerator[dict, None]:
        """Yield parsed JSON messages from the Native Tool Signals socket."""
        while True:
            try:
                frames = await self.signal_pull.recv_multipart()
                payload_bytes = frames[-1]
                payload = json.loads(payload_bytes)
                yield payload
            except Exception as e:
                logger.error(f"Error parsing Signal input: {e}")

    async def listen_core_sub(self) -> AsyncGenerator[Tuple[str, dict], None]:
        """Yield (topic, payload) messages from the Cognition Core SUB socket."""
        while True:
            try:
                frames = await self.core_sub.recv_multipart()
                if len(frames) == 2:
                    topic = frames[0].decode('utf-8')
                    payload = json.loads(frames[1])
                    yield topic, payload
            except Exception as e:
                logger.error(f"Error parsing Core SUB input: {e}")

    async def send_to_core(self, topic: str, payload: dict) -> None:
        """Send a formatted multipart event to the Cognition Core PUSH socket."""
        try:
            await self.core_push.send_multipart([
                topic.encode("utf-8"),
                json.dumps(payload, ensure_ascii=False).encode("utf-8")
            ])
            logger.debug(f"Pushed {topic} to core.")
        except Exception as e:
            logger.error(f"Error sending to Core PUSH: {e}")

    async def close(self):
        self.core_push.close(linger=0)
        self.core_sub.close(linger=0)
        self.signal_pull.close(linger=0)
        self.ui_pull.close(linger=0)
        self._ctx.term()
