"""
Some global tools useful for any sub-module.
"""

import asyncio
import logging
from inspect import iscoroutinefunction
from pathlib import Path
from typing import Any, Awaitable, Callable

import tomllib
from pydantic import BaseModel, ValidationError

type Connection[T, U] = Callable | Callable[[T | U], Awaitable]


logger = logging.getLogger(__name__)


def with_overrides[T: BaseModel](cls: type[T], config_path: Path) -> T | None:
    """Get the config with overrides."""
    overrides = {}

    path = Path(config_path)
    if path.exists():
        try:
            overrides = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as e:
            logger.error("(%s): Config parsing error: %s", config_path, e)
            return None

    try:
        model = cls(**overrides)
        return model
    except ValidationError as e:
        logger.error("Config validation error: %s", e)
        return None


class ConfigManager[T: BaseModel]:
    def __init__(self, config_path: Path, config: T) -> None:
        self.path = config_path
        self._config: T = config
        self._last_mtime: float = 0.0
        self.config_updated: Signal[T, Any] = Signal()

    def get_config(self) -> T:
        """Get the current program configuration."""
        # Calling this is apparently quite expensive on system calls
        # This is also currently blocking.
        # Another solution would be to spin up an async background task to watch the file.
        if self.path.exists():
            current_mtime = self.path.stat().st_mtime
            if current_mtime > self._last_mtime:
                new = with_overrides(type(self._config), self.path)
                if new is not None:
                    self._config = new
                    logger.info(
                        "Config was changed %s -> %s. Reloaded.",
                        self._last_mtime,
                        current_mtime,
                    )
                    self._last_mtime = current_mtime
                    self.config_updated.emit(self._config)
                else:
                    logger.info("Config not updated because of an error.")
        else:
            logger.warning("Config file doesn't exist: %s", self.path.resolve())

        return self._config


class Signal[T, U]:
    """
    Simple async/sync event signals.
    Both work the same way.
    An advantage of using async functions is that they can block like other async functions.
    They are spawned as tasks.
    """

    def __init__(self, *args: type[T], **kwargs: type[U]) -> None:
        self.args = args
        self.kwargs = kwargs
        self.connections: list[Connection[T, U]] = []

    def connect(self, callable: Connection):
        """Connect a function."""
        self.connections.append(callable)

    def disconnect(self, callable: Connection):
        """
        Disconnect a function.

        Raises:
            ValueError if callable is not connected.
        """
        self.connections.remove(callable)

    def is_connected(self, callable: Connection) -> bool:
        """Whether the callable is connected or not."""
        return callable in self.connections

    def emit(self, *args: T, **kwargs: U):
        """
        Emit a signal and call all the connections.
        """
        for connection in self.connections:
            if iscoroutinefunction(connection):
                asyncio.create_task(connection(*args, **kwargs))
            else:
                connection(*args, **kwargs)
