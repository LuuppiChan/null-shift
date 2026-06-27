import base64
from pathlib import Path
import subprocess
import tempfile
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
import magic
import numpy as np
import cv2

from gui.config import manager


def _enforce_character_limit(text: str, limit: int = 10_000) -> str:
    """Enforce character limit on text outputs."""
    if limit <= 0:
        return text

    if limit and len(text) > limit:
        return (
            text[:limit] + f"\n... (output truncated due to character limit of {limit})"
        )
    return text


def _build_media_range_desc(start: Optional[float], end: Optional[float]) -> str:
    """Build a human-readable time range description for media segments."""
    if start is not None and end is not None:
        return f"{start}s to {end}s"
    if start is not None:
        return f"{start}s onwards"
    if end is not None:
        return f"up to {end}s"
    return ""


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


def file_read(
    file_path: str,
    start_line: Optional[int | float] = None,
    end_line: Optional[int | float] = None,
    character_limit: int | None = None,
    show_line_numbers: bool = False,
) -> str | list[str | dict[str, str]]:
    """
    Primary file reading tool.
    Reads files with line-range support.
    Handles both text and binary files, with automatic encoding detection.
    Respects permission boundaries.
    Start and end lines are seconds when reading videos or audio.

    Supported file types includes, but is not limited to
    - text
    - image
    - video
    - pdf
    - audio

    start and end lines are supported on
    - text
    - video
    - pdf (if pdf returns only text)
    - audio

    Args:
        file_path: Absolute path to the file or an artifact name.
        start_line: Optional line to start the read at. Starts at the first line if not specified. (1-indexed)
        end_line: Optional line to end the read at. Reads to the end if not specified. (exclusive)
        character_limit: Max character limit. None means default max character limit.
        show_line_numbers: Show line numbers when reading text.
    """
    file_path = str(Path(file_path).expanduser().resolve())

    # Preserve raw time values for media before converting to 0-indexed for text.
    media_start = start_line
    media_end = end_line

    if start_line is None:
        start_line = 0
    else:
        start_line -= 1

    content_list = []
    path = Path(file_path)
    mime = magic.from_file(path, mime=True)

    if character_limit is None:
        character_limit = 10_000

    if mime.startswith("image/"):
        content_list.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{compress_image(path)}"},
            }
        )
    elif mime.startswith("video/"):
        video = compress_video(path, start=media_start, end=media_end)
        if video is None:
            content_list.append({"text": "Failed to read video."})
        else:
            content_list.append(
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": video,
                    "mime_type": "video/mp4",
                }
            )
            if media_start is not None or media_end is not None:
                desc = _build_media_range_desc(media_start, media_end)
                content_list.append(
                    {
                        "text": f"The video segment ({desc}) has been attached to this message."
                    }
                )
            else:
                content_list.append(
                    {"text": "The video has been attached to this message."}
                )
    elif mime.startswith("audio/"):
        audio = compress_audio(path, start=media_start, end=media_end)
        if audio is None:
            content_list.append({"text": "Failed to read audio."})
        else:
            content_list.append(
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": audio,
                    "mime_type": "audio/aac",
                }
            )
            if media_start is not None or media_end is not None:
                desc = _build_media_range_desc(media_start, media_end)
                content_list.append(
                    {
                        "text": f"The audio segment ({desc}) has been attached to this message."
                    }
                )
            else:
                content_list.append(
                    {"text": "The audio has been attached to this message."}
                )
    elif mime == "application/pdf":
        cfg = manager.get_config()
        if cfg.send_pdf_bin:
            data = base64.b64encode(path.read_bytes()).decode()
            content_list.append(
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": data,
                    "mime_type": "mime",
                }
            )
        else:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            full_text = []
            for doc in docs:
                text = doc.page_content
                full_text.append(text)

            lines = "\n\n".join(full_text).splitlines()
            start_line = round(start_line)
            if end_line is None:
                text = lines[start_line:]
            else:
                end_line = round(end_line)
                text = lines[start_line:end_line]

            content_list.append(
                {"text": _enforce_character_limit("\n".join(text), character_limit)}
            )
    else:
        # if mime.startswith("text/") or any([mime.startswith(t) for t in SAFE_APP_TYPES]):
        if show_line_numbers:
            lines = [
                f"{i + 1}" + line
                for i, line in enumerate(path.read_text(errors="replace").splitlines())
            ]
        else:
            lines = path.read_text(errors="replace").splitlines()
        start_line = round(start_line)
        if end_line is None:
            text = lines[start_line:]
        else:
            end_line = round(end_line)
            text = lines[start_line:end_line]
        content_list.append(
            {"text": _enforce_character_limit("\n".join(text), character_limit)}
        )

    if len(content_list) == 1 and "text" in content_list[0]:
        return content_list[0]["text"]
    return content_list
