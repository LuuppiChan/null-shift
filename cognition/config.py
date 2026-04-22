"""
Cognition node configuration.

Defaults are defined in ``CognitionConfig``. ``cognition.toml`` provides
user overrides on top — any key absent from the TOML falls back to the
dataclass default.
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass — all defaults live here
# ---------------------------------------------------------------------------


@dataclass
class CognitionConfig:
    # -- LLM -----------------------------------------------------------------
    llm_provider: str = "openai"           # "openai" | "vertexai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_temperature: float = 0.7
    llm_max_history: int = 40              # maximum message count in history
    llm_max_iterations: int = 15           # maximum LLM→tool→LLM loops per turn

    # VertexAI only
    llm_vertex_project: str = ""
    llm_vertex_location: str = "us-central1"

    # -- ZMQ -----------------------------------------------------------------
    zmq_input_bind: str = "tcp://*:5555"   # Cognition binds PULL here
    zmq_output_bind: str = "tcp://*:5556"  # Cognition binds PUB here

    # -- Paths ---------------------------------------------------------------
    path_personality: str = "cognition/personality.md"
    path_memory: str = "cognition/memory/memory.md"
    path_expiring_notes: str = "cognition/memory/expiring_notes.json"
    path_history: str = "cognition/memory/history.json"
    path_tools: str = "cognition/tools"
    path_context_plugins: str = "cognition/context_plugins"
    path_prompts: str = "cognition/prompts"

    # -- Prompts -------------------------------------------------------------
    prompts_layout: list[str] = field(default_factory=lambda: ["personality", "prompts", "long_term", "short_term", "context"])

    # -- Behaviour -----------------------------------------------------------
    log_level: str = "INFO"
    agentic_intent_max_words: int = 60

    # Extra fields that arrive via TOML but aren't declared above are
    # collected here so callers can still inspect them.
    _extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_KNOWN_FIELDS: frozenset[str] = frozenset(f.name for f in fields(CognitionConfig))


def load_config(
    path: str | Path = "cognition/cognition.toml",
    into: CognitionConfig | None = None,
) -> CognitionConfig:
    """Load a ``CognitionConfig`` by merging defaults with a TOML override file.

    The TOML file is expected to use the same flat key names as the dataclass
    fields, optionally grouped under ``[llm]``, ``[zmq]``, or ``[paths]``
    sections. Keys under those sections are automatically prefixed (e.g.
    ``[paths] memory`` → ``path_memory``).

    Args:
        path: Path to the TOML configuration file. Missing file is allowed —
              the defaults are returned as-is.
        into: Optional existing :class:`CognitionConfig` instance to update
              in-place. If omitted, a new instance is created.

    Returns:
        A fully resolved :class:`CognitionConfig` instance (either the one
        passed as ``into`` or a new one).
    """
    cfg = into if into is not None else CognitionConfig()
    toml_path = Path(path)

    if not toml_path.exists():
        # Try finding it relative to this file
        alt_path = Path(__file__).parent / "cognition.toml"
        if alt_path.exists():
            toml_path = alt_path
        else:
            logger.warning("Configuration file %s not found (tried alt: %s) — using defaults.", path, alt_path)
            return cfg

    if sys.version_info < (3, 11):
        raise RuntimeError("Cognition requires Python 3.11+ for stdlib tomllib support.")

    with toml_path.open("rb") as fh:
        raw: dict[str, Any] = tomllib.load(fh)

    # Flatten section-based keys into their field names.
    flat: dict[str, Any] = {}

    _SECTION_PREFIXES: dict[str, str] = {
        "llm": "llm_",
        "zmq": "zmq_",
        "paths": "path_",
        "prompts": "prompts_",
        "behaviour": "",
    }

    for section, prefix in _SECTION_PREFIXES.items():
        if section in raw:
            for k, v in raw.pop(section).items():
                flat[f"{prefix}{k}"] = v

    # Remaining top-level keys go in directly.
    flat.update(raw)

    extra: dict[str, Any] = {}
    for k, v in flat.items():
        if k in _KNOWN_FIELDS and k != "_extra":
            setattr(cfg, k, v)
        else:
            extra[k] = v

    cfg._extra = extra
    return cfg
