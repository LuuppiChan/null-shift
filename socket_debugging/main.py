import asyncio
import json
import logging
import sys
from typing import Any, Optional
import readline

import zmq
import zmq.asyncio

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("simple-text")


class SimpleTextFrontend:
    """A minimal text-based frontend for interacting with the core engine."""

    def __init__(
        self,
        input_url: str = "tcp://127.0.0.1:5555",
        output_url: str = "tcp://127.0.0.1:5556",
    ) -> None:
        self.input_url = input_url
        self.output_url = output_url
        self.ctx = zmq.asyncio.Context()
        self.push_socket: zmq.asyncio.Socket = self.ctx.socket(zmq.PUSH)
        self.sub_socket: zmq.asyncio.Socket = self.ctx.socket(zmq.SUB)
        self._receive_task: Optional[asyncio.Task[None]] = None
        self._running: bool = False

    async def start(self) -> None:
        """Starts the frontend connection and background tasks."""
        self.push_socket.connect(self.input_url)
        self.sub_socket.connect(self.output_url)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

        self._running = True
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info(
            "Connected to core. PUSH: %s | SUB: %s", self.input_url, self.output_url
        )

        await self._input_loop()

    async def stop(self) -> None:
        """Stops the frontend and cleans up resources."""
        self._running = False
        if self._receive_task:
            self._receive_task.cancel()
        
        self.push_socket.close()
        self.sub_socket.close()
        self.ctx.term()
        logger.info("Frontend stopped.")

    async def _receive_loop(self) -> None:
        """Background task that asynchronously receives messages from the core."""
        while self._running:
            try:
                frames: list[bytes] = await self.sub_socket.recv_multipart()
                if len(frames) == 2:
                    topic: str = frames[0].decode(errors="replace")
                    payload_base: str = frames[1].decode(errors="replace")
                    try:
                        payload: dict[str, Any] = json.loads(payload_base)
                        self._handle_incoming_message(topic, payload)
                    except json.JSONDecodeError:
                        self._handle_incoming_message(topic, {"raw": payload_base})
                else:
                    logger.warning("Received malformed message: %s", frames)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in receive loop: %s", e, exc_info=True)

    def _handle_incoming_message(self, topic: str, payload: dict[str, Any]) -> None:
        """Handles parsed incoming messages and outputting them to the screen."""
        # Simple logging strategy for testing purposes.
        # Can be expanded for individual topic handling.
        print(f"\n[CORE:{topic}] {payload}")
        # Reprint the prompt prefix dynamically after an asynchronous log
        sys.stdout.write("> ")
        sys.stdout.flush()

    async def _send_input(self, text: str) -> None:
        """Sends a standard text input payload to the core via the PUSH socket."""
        topic: bytes = b"input"
        payload_dict: dict[str, Any] = {"type": "instant", "body": text}
        payload: bytes = json.dumps(payload_dict).encode()
        await self.push_socket.send_multipart([topic, payload])

    async def _send_command(self, cmd: str, args: list[str]) -> None:
        """Sends a specific command payload to the core via the PUSH socket."""
        topic: bytes = b"command"
        payload_dict: dict[str, Any] = {"command": cmd, "args": args}
        payload: bytes = json.dumps(payload_dict).encode()
        await self.push_socket.send_multipart([topic, payload])

    async def _input_loop(self) -> None:
        """Main loop that continuously asks for user input without blocking the event loop."""
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        print("Ready for input. Type '/quit' to exit.")

        while self._running:
            try:
                sys.stdout.write("> ")
                sys.stdout.flush()

                # Await input on a background thread
                line: str = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                if line == "/quit":
                    break
                elif line.startswith("/"):
                    parts: list[str] = line[1:].split(" ", 1)
                    cmd: str = parts[0]
                    args: list[str] = parts[1].split() if len(parts) > 1 else []
                    await self._send_command(cmd, args)
                else:
                    await self._send_input(line)

            except asyncio.CancelledError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("Error sending input: %s", e)


async def main() -> None:
    frontend = SimpleTextFrontend()
    try:
        await frontend.start()
    finally:
        await frontend.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
