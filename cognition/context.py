"""
Async, hot-reloadable context assembly.

Context is gathered every turn from a set of **context plugins** — small
``.py`` files in ``path_context_plugins``, each exposing an
``async def collect() -> str | None`` function. Plugins that return ``None``
or an empty string are silently skipped.

Plugins are hot-reloaded by mtime on every call to ``assemble()``, so new
or updated context sources are picked up without a restart.

Example plugin ``context_plugins/current_time.py``::

    from datetime import datetime

    async def collect() -> str | None:
        return f"Time: {datetime.now().strftime('%H:%M on %A, %B %d, %Y')}"
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cognition.config import CognitionConfig

logger = logging.getLogger(__name__)

_COLLECT_TIMEOUT: float = 0.5  # seconds allowed per plugin


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ContextSnapshot:
    """The assembled context for a single turn.

    Attributes:
        volatile: Dynamic system state (time, window, media, etc.) built fresh
            every turn from the hot-loaded context plugins.
        stable: Long-lived data (memory, expiring notes) that is loaded once
            and refreshed only when the underlying files change.
    """

    volatile: str
    stable: str


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------


class ContextAssembler:
    """Gathers context from hot-reloadable plugins and stable memory files.

    Args:
        config: Resolved :class:`~cognition.config.CognitionConfig` instance.
    """

    def __init__(self, config: CognitionConfig) -> None:
        self._config = config
        self._plugin_mtimes: dict[Path, float] = {}
        self._collectors: dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Hot-reload
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Scan the ``path_context_plugins`` directory for ``.py`` changes.

        New or modified files are re-imported; deleted files are unloaded.
        Called automatically by :meth:`assemble`.
        """
        plugin_dir = Path(self._config.path_context_plugins)
        if not plugin_dir.is_dir():
            return

        found: set[Path] = set()

        for py_file in sorted(plugin_dir.glob("*.py")):
            found.add(py_file)
            mtime = py_file.stat().st_mtime

            if self._plugin_mtimes.get(py_file) == mtime:
                continue

            self._load_plugin(py_file)
            self._plugin_mtimes[py_file] = mtime

        for stale in set(self._plugin_mtimes) - found:
            stem = stale.stem
            self._collectors.pop(stem, None)
            del self._plugin_mtimes[stale]
            logger.info("Unloaded context plugin: %s", stale.name)

    def _load_plugin(self, py_file: Path) -> None:
        """Import or re-import a single context plugin file.

        Args:
            py_file: Path to the ``.py`` file to load.
        """
        module_name = f"cognition.context_plugins._plugin_{py_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            logger.warning("Could not create spec for context plugin %s", py_file)
            return

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("Error loading context plugin %s: %s", py_file.name, exc)
            return

        collect_fn = getattr(module, "collect", None)
        if collect_fn is None or not asyncio.iscoroutinefunction(collect_fn):
            logger.warning(
                "Context plugin %s has no async collect() — skipping.", py_file.name
            )
            return

        self._collectors[py_file.stem] = collect_fn
        logger.info("Loaded context plugin: %s", py_file.name)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    async def assemble(self) -> ContextSnapshot:
        """Build a :class:`ContextSnapshot` for the current turn.

        Hot-reloads plugins, then runs all collectors concurrently with a
        per-plugin timeout. Results are joined with newlines.

        Returns:
            ContextSnapshot: Populated volatile and stable context strings.
        """
        self.reload()

        volatile_parts: list[str] = []

        if self._collectors:
            tasks = [
                asyncio.wait_for(fn(), timeout=_COLLECT_TIMEOUT)
                for fn in self._collectors.values()
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

            for stem, result in zip(self._collectors.keys(), raw_results):
                if isinstance(result, Exception):
                    logger.warning("Context plugin '%s' raised: %s", stem, result)
                elif isinstance(result, str) and result.strip():
                    volatile_parts.append(result.strip())

        stable = self._load_stable()

        return ContextSnapshot(
            volatile="\n".join(volatile_parts),
            stable=stable,
        )

    def _load_stable(self) -> str:
        """Load long-lived memory and expiring notes from disk.

        Returns:
            str: Formatted stable context block, or empty string if nothing found.
        """
        import json
        import time

        parts: list[str] = []
        memory_path = Path(self._config.path_memory)

        if memory_path.exists():
            try:
                content = memory_path.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(content)
            except OSError as exc:
                logger.warning("Could not read memory file: %s", exc)

        notes_path = Path(self._config.path_expiring_notes)
        if notes_path.exists():
            try:
                notes: dict = json.loads(notes_path.read_text(encoding="utf-8"))
                now = time.time()
                active: list[str] = []
                changed = False

                for nid in list(notes.keys()):
                    data = notes[nid]
                    if data["expiry"] > now:
                        mins_left = int((data["expiry"] - now) / 60)
                        active.append(
                            f"- {data['content']} (active for {mins_left} more min)"
                        )
                    else:
                        del notes[nid]
                        changed = True

                if changed:
                    notes_path.write_text(
                        json.dumps(notes, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

                if active:
                    parts.append("# Active Constraints\n" + "\n".join(active))
            except Exception as exc:
                logger.warning("Could not read expiring notes: %s", exc)

        return "\n\n".join(parts)
