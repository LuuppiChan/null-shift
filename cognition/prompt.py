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

from config import CognitionConfig
from context import ContextSnapshot

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
            logger.warning("Personality file not found at %s — using fallback.", path)
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
        """Assemble the full system prompt for the current turn based on config layout.

        Args:
            snapshot: The :class:`~cognition.context.ContextSnapshot` for this turn.
        Returns:
            str: The complete system prompt string.
        """
        self.refresh()

        parts: list[str] = []

        for item in self._config.prompts_layout:
            if item == "personality":
                if self._personality:
                    parts.append(self._personality)
            elif item == "prompts":
                if self._modules:
                    parts.extend(self._modules)
            elif item == "long_term":
                if snapshot.long_term:
                    parts.append(f"# Memory\n{snapshot.long_term}")
            elif item == "short_term":
                if snapshot.short_term:
                    parts.append(f"# Active Constraints\n{snapshot.short_term}")
            elif item == "context":
                if snapshot.volatile:
                    parts.append(f"# Current Context\n{snapshot.volatile}")
            else:
                logger.warning(
                    "Unrecognized prompt layout category: %s (ignoring)", item
                )

        return "\n\n".join(part for part in parts if part)
