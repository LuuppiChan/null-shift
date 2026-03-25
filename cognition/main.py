"""
Cognition node entry point.

Wires all components together and drives the main async event loop.
Input events are queued and processed **sequentially** — a new turn only
starts after the previous one completes. This prevents race conditions when
two input events arrive simultaneously.

Usage::

    python -m cognition          # uses cognition/cognition.toml
    python -m cognition --config /path/to/config.toml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal

from cognition.bus import BusMessage, CognitionBus
from cognition.config import CognitionConfig, load_config
from cognition.context import ContextAssembler
from cognition.history import HistoryManager
from cognition.prompt import PromptAssembler
from cognition.registry import ToolRegistry
from cognition.state import RuntimeState
from cognition.backends import build_backend
from cognition.vector import Vector


def _setup_logging(level: str) -> None:
    """Configure structured logging for the Cognition node.

    Args:
        level: Log level string (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


async def run(config: CognitionConfig) -> None:
    """Initialise all components and run the main event loop.

    The event loop has two concurrent coroutines:

    - ``_listen_loop``: Reads from the PULL socket. Abort/action-result/command
      messages are handled immediately. All other turn events are pushed onto
      ``input_queue``.
    - ``_consumer_loop``: Drains ``input_queue`` one event at a time,
      ``await``-ing each turn to completion before starting the next.

    Args:
        config: Fully resolved :class:`~cognition.config.CognitionConfig`.
    """
    bus = CognitionBus(config)
    state = RuntimeState()
    registry = ToolRegistry()
    context = ContextAssembler(config)
    prompt = PromptAssembler(config)
    history = HistoryManager(config)
    backend = build_backend(config)
    vector = Vector(config, bus, state, registry, context, prompt, history, backend)

    history.load()

    logger = logging.getLogger("main")
    logger.info("Cognition node started.")
    logger.info("Input  → %s", config.zmq_input_bind)
    logger.info("Output → %s", config.zmq_output_bind)

    # Sequential turn queue — prevents simultaneous turns.
    input_queue: asyncio.Queue[BusMessage] = asyncio.Queue()
    shutdown_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: (
                logger.info("Signal %s — shutting down.", s.name),
                shutdown_event.set(),
            ),
        )

    async def _listen_loop() -> None:
        """Read socket messages and route them appropriately."""
        async for event in bus.listen():
            if shutdown_event.is_set():
                break

            topic = event.topic

            # Abort — handled immediately, bypass queue.
            if topic == "input.abort":
                await vector.abort()
                continue

            # Action Node result — fed directly into Vector's pending queue.
            if topic == "action.result":
                await vector.feed_action_result(event)
                continue

            # Commands — dispatch_batched goes through turn queue to avoid
            # conflicting with an in-progress turn.
            if topic == "input.command":
                cmd = event.payload.get("cmd")
                if cmd == "dispatch_batched":
                    await input_queue.put(event)
                elif cmd == "poll_state":
                    reply_topic = event.payload.get("reply_topic", "state.reply")
                    state_dict = vector._state._to_dict()
                    await bus.publish(reply_topic, state_dict)
                else:
                    logger.warning("Unknown command: %r", cmd)
                continue

            # Batched input — enqueue body for later dispatch.
            if topic == "input.batched":
                vector.enqueue_batched(event)
                continue

            # User / instant — push onto sequential turn queue.
            await input_queue.put(event)

    async def _consumer_loop() -> None:
        """Process turn events one at a time from the input queue."""
        while not shutdown_event.is_set():
            try:
                event = await asyncio.wait_for(input_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            try:
                if (
                    event.topic == "input.command"
                    and event.payload.get("cmd") == "dispatch_batched"
                ):
                    await vector.dispatch_batched()
                else:
                    await vector.process_turn(event)
            finally:
                input_queue.task_done()

    listen_task = asyncio.create_task(_listen_loop(), name="listen")
    consumer_task = asyncio.create_task(_consumer_loop(), name="consumer")

    await shutdown_event.wait()

    logger.info("Shutting down — draining queue (%d item(s))...", input_queue.qsize())
    listen_task.cancel()

    # Abort any in-flight turn, then wait for the consumer to finish.
    await vector.abort()
    try:
        await asyncio.wait_for(consumer_task, timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning("Consumer did not finish in time — cancelling.")
        consumer_task.cancel()

    await bus.close()
    logger.info("Cognition node stopped.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Null-Shift Cognition Node")
    parser.add_argument(
        "--config",
        default="cognition/cognition.toml",
        help="Path to cognition.toml (default: cognition/cognition.toml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    _setup_logging(config.log_level)

    try:
        asyncio.run(run(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
