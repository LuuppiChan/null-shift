"""
Global types and random helpers that are useful for any sub-module.
"""

import asyncio
import json
import logging
import multiprocessing
from asyncio import Task, sleep
from datetime import datetime, timedelta
from enum import StrEnum
from multiprocessing.connection import Connection, Pipe
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Self, cast

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MessageTopic(StrEnum):
    INPUT = "input"
    COMMAND = "command"
    STREAM = "out.stream"
    TOOL_CALL = "out.tool_call"
    TOOL_RESULT = "out.tool_result"
    FULL = "out.full"
    FINISHED = "event.finished"
    STARTED = "event.started"
    ABORT = "event.abort"
    COMMAND_RESPONSE = "response"


class MessageType(StrEnum):
    INSTANT = "instant"
    BATCHED = "batched"


class Commands(StrEnum):
    FLUSH_BATCH = "flush_batch"
    ABORT = "abort"
    IS_RUNNING = "is_running"


class Difficulty(StrEnum):
    SIMPLE = "simple"
    TOOL_ASSISTED = "tool_assisted"
    AUTONOMOUS_TRAJECTORY = "autonomous_trajectory"
    AUTONOMOUS_STRICT = "autonomous_strict"
    INFER = "infer"


def is_autonomous(diff: Difficulty | Any) -> bool:
    """Whether the difficulty is autonomous or not."""
    return (
        diff == Difficulty.AUTONOMOUS_STRICT or diff == Difficulty.AUTONOMOUS_TRAJECTORY
    )


class BusMessage(BaseModel):
    """Arbitrary bus message."""

    topic: str
    payload: dict[str, Any] = Field(default_factory=dict)

    def encoded(self) -> list[bytes]:
        """Return a ready-to-send multipart message based on current data."""
        return [self.topic.encode(), json.dumps(self.payload).encode()]

    @classmethod
    def decoded(cls, frames: list[bytes]) -> Self | None:
        """Tries to decode a bus message and either returns a successful BusMessage or None if it failed."""
        if len(frames) != 2:
            logger.error("Malformed message: expected 2 frames, got %s", len(frames))
            return None

        topic_bytes, payload_bytes = frames
        topic = topic_bytes.decode(errors="replace")

        try:
            payload: dict = json.loads(payload_bytes)
        except json.JSONDecodeError as e:
            logger.warning(
                "Error decoding payload for topic %s: %s", topic, e, exc_info=True
            )
            return None

        msg = cls(topic=topic, payload=payload)
        return msg


class InputMessage(BaseModel):
    """Typed input message"""

    type: str = "instant"
    title: Optional[str] = None
    body: str
    media: Optional[list[dict[str, str]]] = None
    difficulty: Optional[Difficulty] = None
    goal: Optional[str] = None
    context: Optional[str] = None

    def to_bus(self, topic: MessageTopic = MessageTopic.INPUT) -> BusMessage:
        return BusMessage(
            topic=topic,
            payload=self.model_dump(
                exclude_defaults=True, exclude_unset=True, exclude_none=True
            ),
        )


class InputCommand(BaseModel):
    """Typed command message"""

    command: str
    args: list[Any] = Field(default_factory=list)


def command_response(**values: Any) -> dict[str, Any]:
    """Helper for building command responses."""
    return values


class CachedFile:
    """
    A class representing a single file.
    It caches the content re-reads the file only when necessary.
    This is hashable based on the file hash.
    """

    def __init__(self, file: Path) -> None:
        self.file: Path = file
        self.mtime: float = 0
        self.cache: str = ""

    def needs_refresh(self) -> bool:
        """
        Whether the file needs to be refreshed or not.
        Returns true if file doesn't exist.
        """
        if not self.file.exists():
            return True
        new_mtime = self.file.stat().st_mtime
        return new_mtime != self.mtime

    def get_contents(self) -> Optional[str]:
        """
        Retrieves this file's content.

        Returns None if an error occurs.

        Should not raise exceptions in any normal circumstance.
        If this raises an exception there's something extremely wrong.
        """
        if not self.file.exists():
            return None

        try:
            if self.needs_refresh():
                self.cache = self.file.read_text()
                new_mtime = self.file.stat().st_mtime
                self.mtime = new_mtime
        except Exception as e:
            logger.error("Could not get file contents from file %s: %s", self.file, e)
            return None

        return self.cache

    def __hash__(self) -> int:
        return self.file.__hash__()


def convert_to_langchain_media(
    media: list[dict[str, str]],
) -> list[dict[str, str | dict[str, str]]]:
    content: list[dict[str, str | dict[str, str]]] = []
    for item in media:
        mtype = item.get("mime_type", None)
        data = item.get("data", None)
        if mtype is None:
            logger.error("No 'mime_type' field in the media.")
            continue

        if data is None:
            logger.error("No 'data' field in the media.")
            continue

        if mtype.startswith("image/"):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mtype};base64,{data}"},
                }
            )
        elif mtype.startswith("video/"):
            content.append(
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": data,
                    "mime_type": mtype,
                }
            )
        elif mtype.startswith("audio/"):
            content.append(
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": data,
                    "mime_type": mtype,
                }
            )
        else:
            content.append({"data": data, "mime_type": mtype})

    return content


async def run_batch_ordered[T](
    functions: Iterable[Callable[[], T]], timeout: float = 1.0
) -> list[Optional[T]]:
    """
    Runs a batch of functions on another threads and returns their result.
    Timeout only means it returns when the given time reaches.
    Some processes may still linger as zombies if they hang.
    But they will at some point end unless they're stuck on an infinite loop.
    """
    results: list[Optional[T]] = []

    tasks: list[Task[T]] = []
    for func in functions:
        task = asyncio.create_task(
            asyncio.wait_for(asyncio.to_thread(func), timeout=timeout),
            name=func.__name__,
        )
        tasks.append(task)

    while not all(map(lambda t: t.done(), tasks)):
        await sleep(0.05)

    for task in tasks:
        try:
            e = task.exception()
            if e is None:
                result = task.result()
            else:
                logger.warning(
                    "Task %s returned an exception: %s",
                    task.get_name(),
                    e,
                    exc_info=True,
                )
                result = None
            results.append(result)
        except (
            asyncio.TimeoutError,
            asyncio.CancelledError,
            asyncio.InvalidStateError,
        ) as e:
            logger.info("Task %s timed out: %s", task.get_name(), e)
            results.append(None)

    return results


async def run_killables[T](
    functions: Iterable[Callable[[], T]], timeout: float = 1.0
) -> list[Optional[T]]:
    """
    Runs a batch of blocking tasks with a timeout.
    Returns the results in the same order as the task list.
    Result can either be the function result or None if it timed out or raised an exception.
    """
    tasks: list[Task[Optional[T]]] = []
    for fn in functions:
        tasks.append(asyncio.create_task(run_killable(fn, timeout)))

    while not all(map(lambda t: t.done(), tasks)):
        await sleep(0.05)

    return [t.result() for t in tasks]


async def run_killable[T](
    function: Callable[[], T],
    timeout: float = 1.0,
) -> Optional[T]:
    """
    Runs a blocking function on a separate thread.
    Returns the result of this function.
    Violently kills the process if it isn't finished and time exceeds timeout.
    Timeout resolution is 0.05 seconds.

    Returns None if timeout happens and the process is violently killed.
    """
    parent_conn, child_conn = Pipe()
    p = multiprocessing.Process(target=_worker, args=(function, child_conn))
    p.start()
    end = datetime.now() + timedelta(seconds=timeout)
    while p.is_alive() and datetime.now() < end:
        await sleep(0.05)

    if p.is_alive():
        p.terminate()
        logger.warning("Worker process %s timed out.", function.__name__)
        # I think this leaves the process hanging,
        # but it's better than freezing the whole program.
        p.join(1)
        parent_conn.close()
        return None

    p.join()
    result = None
    try:
        if parent_conn.poll():
            data: dict[str, bool | T | Exception] = parent_conn.recv()
            if data.get("finished"):
                result = cast(T, data.get("value"))
            else:
                logger.error(
                    "Worker %s crashed: %s", function.__name__, data.get("value")
                )
        else:
            logger.warning("Worker %s didn't return any data.", function.__name__)
    except Exception as e:
        logger.warning("Unexpected error when getting worker result: %s", e)
    finally:
        parent_conn.close()

    return result


def _worker[T](func: Callable[[], T], connection: Connection):
    """Worker processing the function in the separate thread."""
    try:
        result = func()
        connection.send({"finished": True, "value": result})
    except Exception as e:
        connection.send({"finished": False, "value": e})
    finally:
        connection.close()
