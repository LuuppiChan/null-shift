"""
Media content conversion helpers.

Translates the internal mime_type storage format::

    {"mime_type": "image/png", "data": "<base64>"}

into provider-specific multipart content formats for OpenAI and VertexAI.
Backends import from here; the history layer stores the canonical format.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def convert_part_openai(part: dict) -> dict | None:
    """Convert a single mime_type part to an OpenAI multipart content dict.

    Args:
        part: Dict with ``mime_type`` and ``data`` (base64) fields.

    Returns:
        An OpenAI-compatible content part dict, or ``None`` if unsupported.
    """
    mime = part.get("mime_type", "")
    data = part.get("data", "")

    if mime.startswith("image/"):
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data}", "detail": "auto"},
        }
    if mime == "application/pdf":
        return {
            "type": "file",
            "file": {
                "filename": "document.pdf",
                "file_data": f"data:{mime};base64,{data}",
            },
        }

    logger.warning("OpenAI backend: unsupported mime_type '%s' — skipping.", mime)
    return None


def convert_part_vertexai(part: dict) -> dict:
    """Convert a single mime_type part to a VertexAI media content dict.

    Args:
        part: Dict with ``mime_type`` and ``data`` (base64) fields.

    Returns:
        A VertexAI-compatible ``{"type": "media", ...}`` content dict.
    """
    return {
        "type": "media",
        "mime_type": part["mime_type"],
        "data": part["data"],
    }


def convert_content_list_openai(parts: list[dict]) -> list[dict]:
    """Convert a full content list to OpenAI multipart format.

    Passthrough for parts that are already in standard format (e.g.
    ``{"type": "text", "text": "..."}``) and converts mime_type parts.

    Args:
        parts: Mix of mime_type media parts and standard content parts.

    Returns:
        list[dict]: OpenAI-compatible content part list.
    """
    converted: list[dict] = []
    for part in parts:
        if "mime_type" in part:
            result = convert_part_openai(part)
            if result:
                converted.append(result)
        else:
            converted.append(part)
    return converted


def convert_content_list_vertexai(parts: list[dict]) -> list[dict]:
    """Convert a full content list to VertexAI media format.

    Args:
        parts: Mix of mime_type media parts and standard content parts.

    Returns:
        list[dict]: VertexAI-compatible content part list.
    """
    converted: list[dict] = []
    for part in parts:
        if "mime_type" in part:
            converted.append(convert_part_vertexai(part))
        else:
            converted.append(part)
    return converted
