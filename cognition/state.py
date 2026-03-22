"""
Runtime state for the Cognition node.

State is never queried via a synchronous socket — instead, ``Vector``
publishes a ``state.changed`` event on the output PUB socket every time a
field changes. Other nodes subscribe to this topic and react reactively
rather than polling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RuntimeState:
    """Mutable runtime state of the Cognition node.

    All mutations should go through the setter methods so that callers can
    attach a publish callback (``on_change``) without coupling this class to
    the bus directly.

    The ``on_change`` callback receives the serialised state dict and is
    responsible for publishing it on the bus. It is set by ``Vector`` during
    initialisation and called after every field mutation.
    """

    is_busy: bool = False
    turn_id: str | None = None
    is_aborting: bool = False
    tool_active: str | None = None
    at_finished: float = field(default_factory=time.time)

    # Injected by Vector — not serialised.
    _on_change: object = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of the current state.

        Returns:
            dict: Snapshot of all public fields.
        """
        return {
            "is_busy": self.is_busy,
            "turn_id": self.turn_id,
            "is_aborting": self.is_aborting,
            "tool_active": self.tool_active,
            "at_finished": self.at_finished,
        }

    # ------------------------------------------------------------------
    # Mutations — each calls _notify() so state.changed is auto-published
    # ------------------------------------------------------------------

    def set_busy(self, turn_id: str) -> None:
        """Mark the node as busy and record the active turn ID.

        Args:
            turn_id: Unique identifier for the turn being processed.
        """
        self.is_busy = True
        self.turn_id = turn_id
        self.is_aborting = False
        self.tool_active = None
        self._notify()

    def set_idle(self) -> None:
        """Mark the node as idle after a turn completes."""
        self.is_busy = False
        self.turn_id = None
        self.is_aborting = False
        self.tool_active = None
        self.at_finished = time.time()
        self._notify()

    def set_tool_active(self, tool_name: str) -> None:
        """Record which tool is currently executing.

        Args:
            tool_name: Name of the tool being executed.
        """
        self.tool_active = tool_name
        self._notify()

    def clear_tool(self) -> None:
        """Clear the active tool field after execution completes."""
        self.tool_active = None
        self._notify()

    def request_abort(self) -> None:
        """Signal that the current turn should abort at the next safe checkpoint."""
        self.is_aborting = True
        self._notify()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _notify(self) -> None:
        """Invoke the registered change callback, if any."""
        if callable(self._on_change):
            self._on_change(self.to_dict())  # type: ignore[call-arg]
