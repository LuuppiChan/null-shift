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


def tool(func: Callable) -> Callable:
    """Mark an async function as a registerable tool.

    The function's name, docstring, and type-annotated parameters are used to
    auto-generate an OpenAI-style JSON schema.

    Args:
        func: An async callable.

    Returns:
        The original function, unchanged, with a ``_is_tool = True`` marker.
    """
    if not inspect.iscoroutinefunction(func):
        raise TypeError(f"@tool requires an async function, got: {func!r}")
    func._is_tool = True  # type: ignore[attr-defined]
    return func


# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------

_PY_TO_JSON_TYPE: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def _annotation_to_schema(annotation: Any) -> dict:
    """Convert a Python type annotation to a JSON Schema fragment.

    Handles :data:`typing.Literal` annotations by emitting an ``enum``
    constraint in addition to the base type.

    Args:
        annotation: A Python type annotation object.

    Returns:
        dict: A JSON Schema fragment (e.g. ``{"type": "string"}`` or
        ``{"type": "string", "enum": ["fast", "slow"]}``)
    """
    origin = getattr(annotation, "__origin__", None)

    # typing.Literal["a", "b"] → {"type": "string", "enum": ["a", "b"]}
    if origin is typing.Literal:
        args = annotation.__args__
        first = args[0] if args else ""
        if isinstance(first, bool):
            base_type = "boolean"
        elif isinstance(first, int):
            base_type = "integer"
        elif isinstance(first, float):
            base_type = "number"
        else:
            base_type = "string"
        return {"type": base_type, "enum": list(args)}

    # Fall back to simple name-based lookup.
    type_name = getattr(annotation, "__name__", "") or str(annotation)
    return {"type": _PY_TO_JSON_TYPE.get(type_name, "string")}


def _build_schema(func: Callable) -> dict:
    """Build an OpenAI function-calling JSON schema from a function's signature.

    Args:
        func: A decorated async function.

    Returns:
        An OpenAI-compatible ``{"type": "function", "function": {...}}`` dict.
    """
    sig = inspect.signature(func)
    props: dict[str, dict] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            annotation = param.annotation
        else:
            annotation = param.annotation

        schema_fragment = _annotation_to_schema(annotation)
        props[name] = schema_fragment

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (inspect.getdoc(func) or "").strip(),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Registry of async tool coroutines with hot-reload support.

    Attributes:
        _tools: Mapping of tool name to coroutine function.
        _schemas: Mapping of tool name to OpenAI JSON schema dict.
        _plugin_mtimes: Last-modified timestamps for each loaded plugin file.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict] = {}
        self._plugin_mtimes: dict[Path, float] = {}

    # ------------------------------------------------------------------
    # Static registration
    # ------------------------------------------------------------------

    def register(self, func: Callable) -> Callable:
        """Register a function as a tool. Use as a decorator.

        Args:
            func: An async function decorated with :func:`tool`.

        Returns:
            The original function, allowing decorator chaining.
        """
        if not getattr(func, "_is_tool", False):
            logger.error(
                "'%s' is not decorated with @tool — skipping registration.",
                func.__name__,
            )
            return func
        self._tools[func.__name__] = func
        self._schemas[func.__name__] = _build_schema(func)
        logger.debug("Registered built-in tool: %s", func.__name__)
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

        for py_file in sorted(plugin_dir.glob("*.py")):
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
            if callable(func) and getattr(func, "_is_tool", False):
                self._tools[func.__name__] = func
                self._schemas[func.__name__] = _build_schema(func)
                logger.info("Loaded tool '%s' from %s", func.__name__, py_file.name)

    def _unload_plugin(self, py_file: Path) -> None:
        """Remove tools that came from a now-deleted plugin file."""
        # Re-derive tool names by checking which tools share the same module.
        prefix = f"cognition.tools._plugin_{py_file.stem}"
        to_remove = [
            name
            for name, func in self._tools.items()
            if getattr(func, "__module__", "") == prefix
        ]
        for name in to_remove:
            del self._tools[name]
            del self._schemas[name]
            logger.info("Unloaded tool '%s' (plugin removed: %s)", name, py_file.name)

    # ------------------------------------------------------------------
    # Query & dispatch
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[dict]:
        """Return all registered tool schemas in OpenAI function-calling format.

        Returns:
            list[dict]: A list of ``{"type": "function", "function": {...}}`` dicts.
        """
        return list(self._schemas.values())

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
            raise KeyError(
                f"Unknown tool '{name}'. Available tools: {available}"
            )
        result = await self._tools[name](**args)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Module-level registry instance
# ---------------------------------------------------------------------------

registry = ToolRegistry()
