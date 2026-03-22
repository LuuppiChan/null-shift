"""
System prompt assembly.

Concatenates the personality file, prompt modules, and the current
:class:`~cognition.context.ContextSnapshot` into a single system prompt
string that is rebuilt every turn.

Prompt modules are ``.md`` files in ``path_prompts``. Each file is
responsible for its own headers — the filename is not used as a section
header. Files are concatenated in **alphabetical order** with a blank line
separator, so the naming convention ``00_tool_usage.md``, ``10_memory.md``
etc. gives deterministic ordering.
"""

from __future__ import annotations

import logging
from pathlib import Path

from cognition.config import CognitionConfig
from cognition.context import ContextSnapshot

logger = logging.getLogger(__name__)


class PromptAssembler:
    """Builds the system prompt from personality, modules, and live context.

    Args:
        config: Resolved :class:`~cognition.config.CognitionConfig` instance.
    """

    def __init__(self, config: CognitionConfig) -> None:
        self._config = config
        self._personality: str = ""
        self._modules: list[str] = []
        self.refresh()

    # ------------------------------------------------------------------
    # Reload
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read the personality file and all ``.md`` modules from disk.

        Safe to call frequently — changes on disk are picked up immediately.
        """
        self._personality = self._load_personality()
        self._modules = self._load_modules()

    def _load_personality(self) -> str:
        """Load the personality file.

        Returns:
            str: File content, or a minimal default if the file is missing.
        """
        path = Path(self._config.path_personality)
        if not path.exists():
            logger.warning(
                "Personality file not found at %s — using fallback.", path
            )
            return "You are a helpful AI assistant."
        return path.read_text(encoding="utf-8").strip()

    def _load_modules(self) -> list[str]:
        """Load all ``.md`` prompt modules from the prompts directory.

        Returns:
            list[str]: Module contents in alphabetical filename order.
        """
        prompts_dir = Path(self._config.path_prompts)
        if not prompts_dir.is_dir():
            return []

        modules: list[str] = []
        for md_file in sorted(prompts_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8").strip()
                if content:
                    modules.append(content)
            except OSError as exc:
                logger.warning("Could not read prompt module %s: %s", md_file.name, exc)

        return modules

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, snapshot: ContextSnapshot) -> str:
        """Assemble the full system prompt for the current turn.

        The structure is::

            {personality}

            {module_1}

            {module_2}
            ...

            # Memory
            {snapshot.stable}

            # Current Context
            {snapshot.volatile}

        Args:
            snapshot: The :class:`~cognition.context.ContextSnapshot` for
                this turn, produced by :class:`~cognition.context.ContextAssembler`.

        Returns:
            str: The complete system prompt string.
        """
        self.refresh()

        parts: list[str] = [self._personality]
        parts.extend(self._modules)

        if snapshot.stable:
            parts.append(f"# Memory\n{snapshot.stable}")

        if snapshot.volatile:
            parts.append(f"# Current Context\n{snapshot.volatile}")

        return "\n\n".join(part for part in parts if part)
