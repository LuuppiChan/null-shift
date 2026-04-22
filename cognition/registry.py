"""
Async, hot-reloadable tool registry.

Tools can be registered statically with the ``@registry.register`` decorator
(for built-ins), or dynamically by dropping a ``.py`` file into the
``path_tools`` directory. ``reload()`` is called at the start of every turn
so new plugins are picked up without a restart.

Each plugin file must expose one or more ``async def`` functions decorated
with ``@tool`` from this module. Example ``tools/hello.py``::

    from cognition.registry import tool

    @tool
    async def hello(name: str) -> str:
        \"\"\"Say hello to someone.\"\"\"
        return f"Hello, {name}!"

The registry converts each decorated function into an OpenAI-compatible JSON
schema entry using its signature and docstring, making it LLM-agnostic.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import typing
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


from langchain_core.tools import BaseTool
from langchain_core.tools import tool as lc_tool
from langchain_core.utils.function_calling import convert_to_openai_tool


def tool(func: Callable) -> BaseTool:
    """Mark an async function as a registerable tool.

    Wraps the function using LangChain's @tool decorator.

    Args:
        func: An async callable.

    Returns:
        A LangChain BaseTool instance.
    """
    if not inspect.iscoroutinefunction(func):
        raise TypeError(f"@tool requires an async function, got: {func!r}")

    # Langchain's tool decorator will create a BaseTool
    t = lc_tool(func)
    t._is_tool = True  # type: ignore[attr-defined]
    return t


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Registry of async tool coroutines with hot-reload support.

    Attributes:
        _tools: Mapping of tool name to LangChain BaseTool.
        _plugin_mtimes: Last-modified timestamps for each loaded plugin file.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._plugin_mtimes: dict[Path, float] = {}

    # ------------------------------------------------------------------
    # Static registration
    # ------------------------------------------------------------------

    def register(self, func: BaseTool) -> BaseTool:
        """Register a function as a tool. Use as a decorator.

        Args:
            func: An async function decorated with :func:`tool`.

        Returns:
            The original function, allowing decorator chaining.
        """
        if not isinstance(func, BaseTool):
            logger.error(
                "'%s' is not a LangChain tool — skipping registration.",
                getattr(func, "__name__", str(func)),
            )
            return func
        self._tools[func.name] = func
        logger.debug("Registered built-in tool: %s", func.name)
        return func

    # ------------------------------------------------------------------
    # Hot-reload
    # ------------------------------------------------------------------

    def reload(self, path: str | Path) -> None:
        """Scan a directory for ``.py`` plugin files and import changed ones.

        Files that have not changed since the last scan (by mtime) are skipped.
        Newly added files are loaded on first scan; removed files are unloaded.

        Args:
            path: Directory to scan. Missing directories are silently ignored.
        """
        plugin_dir = Path(path)
        if not plugin_dir.is_dir():
            return

        found_paths: set[Path] = set()

        for py_file in sorted(plugin_dir.rglob("*.py")):
            found_paths.add(py_file)
            mtime = py_file.stat().st_mtime

            if self._plugin_mtimes.get(py_file) == mtime:
                continue  # unchanged

            self._load_plugin(py_file)
            self._plugin_mtimes[py_file] = mtime

        # Unload tools from files that no longer exist.
        removed = set(self._plugin_mtimes) - found_paths
        for stale in removed:
            self._unload_plugin(stale)
            del self._plugin_mtimes[stale]

    def _load_plugin(self, py_file: Path) -> None:
        """Import or re-import a single plugin file and register its tools."""
        module_name = f"cognition.tools._plugin_{py_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            logger.warning("Could not create module spec for %s", py_file)
            return

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("Error loading tool plugin %s: %s", py_file.name, exc)
            return

        for attr_name in dir(module):
            func = getattr(module, attr_name)
            if isinstance(func, BaseTool):
                self._tools[func.name] = func
                logger.info("Loaded tool '%s' from %s", func.name, py_file.name)

    def _unload_plugin(self, py_file: Path) -> None:
        """Remove tools that came from a now-deleted plugin file."""
        # Re-derive tool names by checking which tools share the same module.
        prefix = f"cognition.tools._plugin_{py_file.stem}"

        def _get_module(t: BaseTool) -> str:
            if hasattr(t, "func") and hasattr(t.func, "__module__"):
                return t.func.__module__
            return getattr(t, "__module__", "")

        to_remove = [
            name for name, func in self._tools.items() if _get_module(func) == prefix
        ]
        for name in to_remove:
            del self._tools[name]
            logger.info("Unloaded tool '%s' (plugin removed: %s)", name, py_file.name)

    # ------------------------------------------------------------------
    # Query & dispatch
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[BaseTool]:
        """Return all registered tools.

        Returns:
            list[BaseTool]: A list of LangChain BaseTool objects.
        """
        return list(self._tools.values())

    async def call(self, name: str, args: dict[str, Any]) -> str:
        """Dispatch an async tool call by name.

        Args:
            name: Tool name as it appears in the schema.
            args: Keyword arguments forwarded to the coroutine.

        Returns:
            str: The string result of the tool. Non-string return values are
            JSON-serialised.

        Raises:
            KeyError: If the tool name is not registered.
        """
        if name not in self._tools:
            available = ", ".join(self._tools) or "(none)"
            raise KeyError(f"Unknown tool '{name}'. Available tools: {available}")

        tool_instance = self._tools[name]
        result = await tool_instance.ainvoke(args)

        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Module-level registry instance
# ---------------------------------------------------------------------------

registry = ToolRegistry()
