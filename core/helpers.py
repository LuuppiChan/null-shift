import json
import logging
from pathlib import Path
import subprocess
from typing import Any, Optional, Self
import tempfile

import cv2
import base64

from core.config import manager, tool_manager


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


def enforce_character_limit(text: str) -> str:
    """Enforce character limit on text outputs."""
    cfg = tool_manager.get_config()
    limit = cfg.file_absurd_size_limit
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


def compress_audio(audio_path: str | Path) -> Optional[str]:
    """
    Returns a compressed base64 string of an audio.
    Mime type is 'audio/aac'
    Returns None if ffmpeg didn't write anything or returned a non-zero code.
    """
    with tempfile.NamedTemporaryFile(suffix=".aac") as temp_file:
        cmd = ["ffmpeg", "-y", "-i", str(audio_path), "-b:a", "128k", temp_file.name]

        process = subprocess.run(cmd, capture_output=True)

        if process.returncode != 0:
            return None

        temp_file.seek(0)
        buffer = temp_file.read()
        if not buffer:
            return None

        base64_str = base64.b64encode(buffer).decode()
        return base64_str


def get_permission(prompt: str) -> bool:
    """
    Placeholder for permission to use this tool.
    Currently asks console input.
    """
    timeout = manager.get_config().core_permission_timeout

    # Socket system thing here I think.
    print(prompt)
    output = subprocess.run(
        [
            "zenity",
            "--question",
            "--title",
            "Vector Permission Request",
            "--text",
            prompt,
        ],
        timeout=timeout,
        #                                   ^^^^ will be customizable
    )
    if output.returncode == 0:
        text = "yes"
    else:
        text = "no"

    if text is None:
        return False

    # Prioritise decline
    for word in manager.get_config().core_permission_no_words:
        if word.lower() in text.lower():
            return False

    for word in manager.get_config().core_permission_yes_words:
        if word.lower() in text.lower():
            return True

    return False
