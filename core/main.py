import asyncio
import json
import logging
import signal

import zmq
import zmq.asyncio

import core.config
import core.log_manager
from core.config import CoreConfig, manager, state
from global_types import BusMessage
from core.vector import Vector

logger = logging.getLogger(__name__)

ctx = zmq.asyncio.Context()


def handle_signals():
    loop = asyncio.get_running_loop()

    def sigterm():
        logger.info("Signal %s – shutting down.", signal.SIGTERM.name)
        state.shutdown_event.set()

    loop.add_signal_handler(signal.SIGTERM, sigterm)

    def sigint():
        logger.info("Signal %s – shutting down.", signal.SIGINT.name)
        state.shutdown_event.set()

    loop.add_signal_handler(signal.SIGINT, sigint)


async def listener_loop(input_queue: asyncio.Queue[BusMessage]):
    logger = logging.getLogger("listener_loop")
    sock = ctx.socket(zmq.PULL)
    current_bind = manager.get_config().zmq_input_bind
    sock.bind(current_bind)
    needs_rebind = False

    def trigger_rebind(new_config: CoreConfig):
        nonlocal needs_rebind
        new_bind = new_config.zmq_input_bind
        if new_bind != current_bind:
            needs_rebind = True

    manager.config_updated.connect(trigger_rebind)

    try:
        while not state.shutdown_event.is_set():
            if needs_rebind:
                try:
                    logger.info("Socket updated, rebinding...")
                    new_bind = manager.get_config().zmq_input_bind
                    sock.unbind(current_bind)
                    await asyncio.sleep(0.1)
                    sock.bind(new_bind)
                    current_bind = new_bind
                    needs_rebind = False
                    logger.info("Rebound input socket to %s", new_bind)
                except zmq.ZMQError as e:
                    logger.critical("Failed to rebind socket: %s", e)

            try:
                frames = await asyncio.wait_for(sock.recv_multipart(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            msg = BusMessage.decoded(frames)
            logger.debug("Received message: %s", msg)
            if msg is None:
                logger.warning("Skipping message as decode failed.")
                continue

            logger.info("Received message from topic %s", msg.topic)

            await input_queue.put(msg)
    finally:
        sock.close()


async def main():
    handle_signals()

    input_queue: asyncio.Queue[BusMessage] = asyncio.Queue()
    vector = Vector(input_queue)

    logger.info("Starting listener")
    listen_task = asyncio.create_task(listener_loop(input_queue), name="listen")
    consumer_task = asyncio.create_task(vector.consumer_loop(), name="consumer")
    message_task = asyncio.create_task(vector.message_loop(), name="message")

    logger.info("All set")
    await state.shutdown_event.wait()

    listen_task.cancel()
    consumer_task.cancel()
    message_task.cancel()
