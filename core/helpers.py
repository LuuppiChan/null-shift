import asyncio
import base64
from datetime import datetime, timedelta
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from time import sleep
from typing import Any, Literal, Optional, Self, cast, overload

import cv2
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
)
import numpy as np

from core.backends import get_backend
from core.config import manager, tool_manager, state
from core.registry import LLMTool


@overload
async def ask_ai(
    message: HumanMessage,
    system: str = "",
    model_level: Literal["high", "medium", "low"] = "medium",
    history: list[BaseMessage] | None = None,
    tools: list[LLMTool] | None = None,
) -> AIMessage: ...


@overload
async def ask_ai(
    message: str,
    system: str = "",
    model_level: Literal["high", "medium", "low"] = "medium",
    history: list[BaseMessage] | None = None,
    tools: list[LLMTool] | None = None,
) -> str: ...


async def ask_ai(
    message: str | HumanMessage,
    system: str = "",
    model_level: Literal["high", "medium", "low"] = "medium",
    history: list[BaseMessage] | None = None,
    tools: list[LLMTool] | None = None,
) -> str | AIMessage:
    """
    Temporary query to an AI.
    Args:
        message: User message to the AI
        system: System prompt for the conversation
        model_level: Model level to use
        history: If you want to have a multi message history use this. It assumes correct structure. The system message is appended to the first position and the human message is appended to the end.

    Raises:
        Whatever langchain raises when it fails.

    History state:
    History state is unchanged
    """
    logger = logging.getLogger("ask_ai")

    cfg = manager.get_config()
    level = cfg.llm.models.main
    match model_level:
        case "high":
            level = cfg.llm.models.high
        case "medium":
            level = cfg.llm.models.medium
        case "low":
            level = cfg.llm.models.low

    model = cfg.get_model(level)
    system_message = []
    if system:
        system_message = [SystemMessage(system)]

    if isinstance(message, HumanMessage):
        human = message
    else:
        human = HumanMessage(message)

    messages = system_message + (history or []) + [human]
    llm = get_backend(model)
    response = llm.stream(messages, tools)
    full: AIMessage | None = None
    error: Exception | None = None
    for _ in range(3):
        try:
            async for chunk in response:
                full = chunk if full is None else cast(AIMessage, full + chunk)
        except Exception as e:
            error = e
            full = None
            logger.error("Error while getting answer: %s", e, exc_info=True)

    if full is None:
        logger.error("AI failed to respond.")
        return "AI failed to respond: %s" % error

    res = ""
    if isinstance(full.content, str):
        res += full.content
    elif isinstance(full.content, list):
        for item in full.content:
            if isinstance(item, str):
                res += item
            elif isinstance(item, dict):
                res += str(item.get("text", ""))

    if not res:
        logger.error("AI returned an empty response.")

    if isinstance(message, str):
        return res
    else:
        return full


class PromptHelper:
    """Class to help with prompts."""

    def __init__(self, tag: str, description: Optional[str] = None) -> None:
        self.parts: list[str | PromptHelper] = []
        self.tag: str = tag
        self.description: Optional[str] = description
        self.logger = logging.getLogger(repr(self))

    def add_part(
        self,
        content: str | Any | Self,
        tag: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Add a part to the prompt."""
        if isinstance(content, PromptHelper):
            if tag is not None:
                content.tag = tag
            if description is not None:
                content.description = description
            self.parts.append(content)
        else:
            if tag is None:
                if description is not None:
                    self.logger.error(
                        "Got a description even though tag is None. Ignoring."
                    )
                self.parts.append(str(content))
            else:
                desc = (
                    f' description="{description}"' if description is not None else ""
                )
                self.parts.append(f"<{tag}{desc}>\n{content}\n</{tag}>")

    def compile(self, separator: str = "\n\n") -> str:
        """Compile full prompt from parts."""
        parts: list[str] = []
        for part in self.parts:
            if isinstance(part, PromptHelper):
                parts.append(part.compile(separator))
            else:
                parts.append(part)
        return f"<{self.tag}>\n{separator.join(parts)}\n</{self.tag}>"

    def __bool__(self) -> bool:
        return bool(self.parts)


def xml_tag(content: str | Any, tag: str, description: str = "") -> str:
    """xml tag macro"""
    return f"<{tag}{f' description="{description}"' if description else ''}>\n{content}\n</{tag}>"


def enforce_character_limit(text: str, limit: int | None = None) -> str:
    """Enforce character limit on text outputs."""
    if limit is None:
        cfg = tool_manager.get_config()
        limit = cfg.file_absurd_size_limit

    if limit <= 0:
        return text

    if limit and len(text) > limit:
        return (
            text[:limit] + f"\n... (output truncated due to character limit of {limit})"
        )
    return text


def fmt_dict(to_format: dict[str, Any]) -> str:
    """Return a dict as a string as json with indent=2"""
    return f"{json.dumps(to_format, indent=2)}"


def completed(*args: str) -> subprocess.CompletedProcess[str]:
    """
    Returns executed process of the command as text and output captured.
    May rise a subprocess.CalledProcessError
    """
    output = subprocess.run(args, text=True, check=True, capture_output=True)
    return output


def compress_image_buffer(base64_str: str, quality: int = 50) -> str:
    """base64 image buffer."""
    img_bytes = base64.b64decode(base64_str)

    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    assert img is not None, "Failed to encode base64 image"
    success, compressed_img = cv2.imencode(
        ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    assert success, "Failed to compress base64 image"
    compressed_b64 = base64.b64encode(compressed_img).decode("utf-8")
    return compressed_b64


def compress_image(image_path: str | Path) -> str:
    """
    Returns a compressed base64 string of an image.
    Mime type is 'image/jpeg'
    Uses assertions to check for fails.
    """
    image = cv2.imread(image_path)
    assert image is not None, f"Failed to load image {image_path}"
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 50])
    assert success, f"Failed to compress image {image_path}"
    base64_str = base64.b64encode(buffer).decode()
    return base64_str


def compress_video(
    video_path: str | Path,
    start: Optional[float] = None,
    end: Optional[float] = None,
    target_width: Optional[int] = None,
) -> Optional[str]:
    # parameters should probably be customizable via config
    """
    Returns a compressed base64 string of a video.
    Mime type is 'video/mp4'
    Returns None if ffmpeg didn't write anything or returned a non-zero code.
    """
    if target_width is None:
        target_width = 540

    trim_flags_pre = []
    trim_flags_post = []

    if start is not None:
        trim_flags_pre.extend(["-ss", str(start)])

    if start is not None and end is not None:
        duration = max(0.0, end - start)
        trim_flags_post.extend(["-t", str(duration)])
    elif end is not None:
        trim_flags_post.extend(["-to", str(end)])

    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
        cmd_vaapi = [
            "ffmpeg",
            "-vaapi_device",
            "/dev/dri/renderD128",
            "-y",
            *trim_flags_pre,
            "-i",
            str(video_path),
            *trim_flags_post,
            "-vf",
            f"format=nv12,hwupload,scale_vaapi=w={target_width}:h=-1",
            "-c:v",
            "h264_vaapi",
            "-q:v",
            "26",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            temp_file.name,
        ]
        process = subprocess.run(cmd_vaapi, capture_output=True)

        if process.returncode != 0:
            # fallback
            cmd_fallback = [
                "ffmpeg",
                "-y",
                *trim_flags_pre,
                "-i",
                str(video_path),
                *trim_flags_post,
                "-vf",
                f"scale={target_width}:-1",
                "-c:v",
                "libx264",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                temp_file.name,
            ]
            code = subprocess.run(cmd_fallback, capture_output=True).returncode

            if code != 0:
                return None

        # Read the temporary file into bytes
        temp_file.seek(0)
        buffer = temp_file.read()
        if not buffer:
            return None

        base64_str = base64.b64encode(buffer).decode()
        return base64_str


def compress_audio(
    audio_path: str | Path,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> Optional[str]:
    """
    Returns a compressed base64 string of an audio.
    Mime type is 'audio/aac'
    Returns None if ffmpeg didn't write anything or returned a non-zero code.

    start and end are timestamps in seconds.  When both are provided
    the clip runs from start to end; when only one is provided the clip
    runs from start to the end of the file or from the beginning to end.
    """
    trim_flags_pre: list[str] = []
    trim_flags_post: list[str] = []

    if start is not None:
        trim_flags_pre.extend(["-ss", str(start)])

    if start is not None and end is not None:
        duration = max(0.0, end - start)
        trim_flags_post.extend(["-t", str(duration)])
    elif end is not None:
        trim_flags_post.extend(["-to", str(end)])

    with tempfile.NamedTemporaryFile(suffix=".aac") as temp_file:
        cmd = [
            "ffmpeg",
            "-y",
            *trim_flags_pre,
            "-i",
            str(audio_path),
            *trim_flags_post,
            "-b:a",
            "128k",
            temp_file.name,
        ]

        process = subprocess.run(cmd, capture_output=True)

        if process.returncode != 0:
            return None

        temp_file.seek(0)
        buffer = temp_file.read()
        if not buffer:
            return None

        base64_str = base64.b64encode(buffer).decode()
        return base64_str


def get_permission(prompt: str) -> tuple[bool, str]:
    """
    Sends a permission request.
    Gives the result and decline_reason.
    """
    from core.socket_system import socket_out
    from global_types import BusMessage, MessageTopic, PendingPermissionRequest

    logger = logging.getLogger("get_permission")

    assert state.main_loop is not None

    timeout = manager.get_config().permissions.timeout
    req = PendingPermissionRequest(title=prompt)
    state.pending_permission_requests.append(req)
    try:
        asyncio.run_coroutine_threadsafe(
            socket_out.send(
                BusMessage(
                    topic=MessageTopic.PERMISSION_REQUEST,
                    payload=req.model_dump(),
                )
            ),
            state.main_loop,
        ).result(5)
    except TimeoutError as e:
        logger.error(
            "Failed to send permission request to socket_out: %s",
            e,
            exc_info=True,
            stack_info=True,
        )
        return False, "Failed to send permission request. Inform user."

    end = datetime.now() + timedelta(seconds=timeout)
    while datetime.now() < end:
        for res in state.pending_permission_responses:
            if res.id == req.id:
                state.pending_permission_responses.remove(res)
                state.pending_permission_requests.remove(req)
                break
        sleep(0.01)
    else:
        logger.warning("Permission %s timed out.", req.id)
        state.pending_permission_requests.remove(req)
        res = req
        res.accepted = False
        res.decline_reason = "Permission request timed out."

    if res.accepted is not None:
        return res.accepted, res.decline_reason

    if res.response_text is None:
        logger.error(
            "Permission request didn't have res.accepted or res.response_text, automatically declining."
        )
        return False, "Permission response was malformed. Inform the user."

    # Prioritise decline
    for word in manager.get_config().permissions.no_words:
        if word.lower() in res.response_text.lower():
            return False, res.decline_reason

    for word in manager.get_config().permissions.yes_words:
        if word.lower() in res.response_text.lower():
            return True, res.decline_reason

    return False, res.decline_reason
