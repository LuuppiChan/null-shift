import asyncio
import base64
import logging
import os
import re
from pathlib import Path
from typing import Literal, Optional

import magic
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool

from core.config import manager, tool_manager
from core.helpers import compress_audio, compress_image, compress_video, get_permission
from core.helpers import enforce_character_limit as _enforce_character_limit
from global_types import run_killable

logger = logging.getLogger(__name__)


artifacts = ["MEMORY.md", "plan.md", "task.md"]
type Artifacts = Literal[*artifacts]  # pyright: ignore[reportInvalidTypeForm]


def _to_path(artifact: Artifacts) -> str:
    cfg = tool_manager.get_config()
    match artifact:
        case "MEMORY.md":
            return cfg.dynamic_memory_path
        case "plan.md":
            return cfg.dynamic_plan_path
        case "task.md":
            return cfg.dynamic_task_path
        case _:
            return ""


def _is_allowed(file_path: str | Path) -> bool:
    cfg = tool_manager.get_config()

    # resolved path
    path = Path(file_path).expanduser().resolve()

    if str(path) in cfg.file_path_whitelist:
        return True

    if str(path) in cfg.file_path_blacklist:
        return False

    return True


def _bypass_prompt(file_path: str | Path) -> bool:
    cfg = tool_manager.get_config()

    # resolved path
    path = Path(file_path).expanduser().resolve()
    for allowed in cfg.file_path_whitelist:
        allowed = Path(allowed).expanduser().resolve()
        if str(path).startswith(str(allowed)):
            return True
    return False


def _validate_and_resolve_path(
    file_path: str | Artifacts, action: Literal["read", "edit"]
) -> tuple[str, Optional[str]]:
    cfg = tool_manager.get_config()
    if file_path in artifacts:
        resolved_path = _to_path(file_path)
        if not resolved_path:
            return (
                "",
                "Tool error, path was empty when it was expected to be a path to an artifact.",
            )
        return resolved_path, None

    if not _is_allowed(file_path):
        action_str = "read" if action == "read" else "edited or written to"
        return "", f"Error: File {file_path} is blacklisted and cannot be {action_str}."

    if action == "edit":
        if cfg.file_prompt_edit and not _bypass_prompt(file_path):
            if not get_permission(cfg.file_prompt_edit_prompt.format(file_path)):
                return "", f"User declined the request to edit {file_path}"

    else:
        if cfg.file_prompt_read and not _bypass_prompt(file_path):
            if not get_permission(cfg.file_prompt_read_prompt.format(file_path)):
                return "", f"User declined the request to read {file_path}"

    return str(file_path), None


@tool
def file_read(
    file_path: str | Artifacts,
    start_line: Optional[int | float] = None,
    end_line: Optional[int | float] = None,
) -> str | list[str | dict[str, str]]:
    """
    Primary file reading tool.
    Reads files with line-range support.
    Handles both text and binary files, with automatic encoding detection.
    Respects permission boundaries.
    Start and end lines are seconds when reading videos.

    Supported file types includes, but is not limited to
    - text
    - image
    - video
    - pdf
    - audio

    Args:
        file_path: Absolute path to the file or an artifact name.
        start_line: Optional line to start the read at. Starts at the first line if not specified. (1-indexed)
        end_line: Optional line to end the read at. Reads to the end if not specified. (exclusive)
    """
    resolved_path, err = _validate_and_resolve_path(file_path, "read")
    if err:
        return err
    file_path = resolved_path

    if start_line is None:
        start_line = 0
    else:
        start_line -= 1

    content_list = []
    path = Path(file_path)
    mime = magic.from_file(path, mime=True)

    if mime.startswith("text/") or mime == "application/xml":
        lines = path.read_text().splitlines()
        start_line = round(start_line)
        if end_line is None:
            text = lines[start_line:]
        else:
            end_line = round(end_line)
            text = lines[start_line:end_line]
        content_list.append({"text": _enforce_character_limit("\n".join(text))})
    elif mime.startswith("image/"):
        content_list.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{compress_image(path)}"},
            }
        )
    elif mime.startswith("video/"):
        video = compress_video(path)
        if video is None:
            content_list.append(
                {"text": _enforce_character_limit("Failed to read video.")}
            )
        else:
            content_list.append(
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": video,
                    "mime_type": "video/mp4",
                }
            )
            content_list.append(
                {"text": "The video has been attached to this message."}
            )
    elif mime.startswith("audio/"):
        audio = compress_audio(path)
        if audio is None:
            content_list.append(
                {"text": _enforce_character_limit("Failed to read audio.")}
            )
        else:
            content_list.append(
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": audio,
                    "mime_type": "audio/aac",
                }
            )
            content_list.append(
                {"text": "The audio has been attached to this message."}
            )
    elif mime == "application/pdf":
        # Just send to the content is as base64 as-is
        # Vertex AI will probably handle it.
        from core.backends import get_backend
        from core.backends.vertexai import VertexAIBackend

        cfg = manager.get_config()
        model = cfg.get_model(cfg.llm.models.main)
        if isinstance(get_backend(model), VertexAIBackend):
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
            for doc in docs:
                doc.page_content = _enforce_character_limit(doc.page_content)
                content_list.append(doc.to_json())
    else:
        return _enforce_character_limit(f"Error: Unsupported file type: {mime}")

    if len(content_list) == 1 and "text" in content_list[0]:
        return content_list[0]["text"]
    return content_list


@tool
def file_edit(
    file_path: str | Artifacts,
    target_content: str,
    replacement_content: str,
    start_line: int,
    end_line: int,
    allow_multiple: bool = False,
) -> str:
    """
    Make targeted edits to existing files using search and replace.
    Uses exact string matching to locate edit targets.
    Args:
        file_path: Absolute path to the file or an artifact name.
        target_content: The exact string to be replaced (including whitespace).
        replacement_content: The new text to insert.
        start_line: Start of the search range (1-indexed).
        end_line: End of the search range (inclusive).
        allow_multiple: If True, replaces all occurrences in range. If False, errors if multiple found.
    Returns:
        str: Confirmation or error details.
    """
    resolved_path, err = _validate_and_resolve_path(file_path, "edit")
    if err:
        return err
    file_path = resolved_path

    if not os.path.exists(file_path):
        return _enforce_character_limit(f"Error: File '{file_path}' does not exist.")

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            return _enforce_character_limit(
                f"Error: Invalid line range {start_line}-{end_line}. File has {len(lines)} lines."
            )

        content_to_search = "".join(lines[start_line - 1 : end_line])
        count = content_to_search.count(target_content)

        if count == 0:
            return _enforce_character_limit(
                f"Error: target_content not found in lines {start_line}-{end_line}."
            )
        if count > 1 and not allow_multiple:
            return _enforce_character_limit(
                f"Error: target_content found {count} times in lines {start_line}-{end_line}. Set allow_multiple=True to replace all."
            )

        new_content = content_to_search.replace(target_content, replacement_content)

        # Reconstruct the file
        final_lines = lines[: start_line - 1] + [new_content] + lines[end_line:]

        with open(file_path, "w") as f:
            f.writelines(final_lines)

        return _enforce_character_limit(
            f"Successfully updated '{file_path}' (replaced {count} occurrence(s))."
        )
    except Exception as e:
        return _enforce_character_limit(f"Error editing file: {e}")


@tool
def file_write(
    file_path: str | Artifacts, content: str, overwrite: bool = False
) -> str:
    """
    Writes or appends content to a file.
    Also can be used to delete files and directories by giving a path to them and writing an empty string.
    Args:
        file_path: Absolute path to the file or an artifact name.
        content: The text to write. An empty string means to delete the file.
        overwrite: If True, replaces content. If False, appends.
    """
    resolved_path, err = _validate_and_resolve_path(file_path, "edit")
    if err:
        return err
    file_path = resolved_path

    try:
        # Create directory if it doesn't exist (useful for scratchpad)
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if content == "":
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    return "Error: Directory not empty, cannot remove."
            else:
                path.unlink(missing_ok=True)
            return "File deleted or directory removed."

        mode = "w" if overwrite else "a"
        with open(file_path, mode) as f:
            f.write(content)
        return _enforce_character_limit(
            f"Successfully {'written to' if overwrite else 'appended to'} '{file_path}'."
        )
    except Exception as e:
        return _enforce_character_limit(f"Error writing to file: {e}")


@tool
def file_glob(pattern: str, file_path: str | Artifacts) -> str:
    """
    Find files matching a glob pattern across the system.
    Searches the workspace using glob patterns (e.g., **/*.md).

    Args:
        pattern: Glob pattern
        file_path: Absolute path to the directory to start the glob search from, or an artifact name.
    """
    resolved_path, err = _validate_and_resolve_path(file_path, "read")
    if err:
        return err
    file_path = resolved_path

    cfg = tool_manager.get_config()
    timeout = cfg.file_query_timeout

    try:
        path_obj = Path(file_path)
        if not path_obj.is_dir():
            return _enforce_character_limit(
                f"Error: The provided path '{file_path}' is not a directory or does not exist."
            )

        found_files = []
        files = asyncio.run(
            run_killable(lambda: list(path_obj.glob(pattern)), timeout=timeout)
        )
        if files is None:
            raise TimeoutError(
                "Globbing took too long and timed out or there was an internal issue with the tool"
            )

        for p in files:
            if not p.is_file():  # Only search files
                continue
            if _is_allowed(p):
                found_files.append(str(p.resolve()))

        if not found_files:
            return _enforce_character_limit(
                f"No files found matching pattern '{pattern}' in '{file_path}'."
            )
        return _enforce_character_limit("\n".join(found_files))
    except Exception as e:
        return _enforce_character_limit(f"Error globbing files: {e}")


@tool
def file_grep(
    pattern: str,
    file_path: str | Artifacts,
    include: Optional[str] = None,
    exclude: Optional[str] = None,
    ignore_case: bool = False,
) -> str:
    """
    Search file contents using regular expressions.
    Performs regex search across files in the workspace.
    Returns matching lines with file paths and line numbers.
    Supports case-insensitive and include/exclude patterns.

    Args:
        pattern: Regex pattern to search for within file contents.
        file_path: Absolute path to the directory to start the grep search from, or an artifact name.
        include: Optional glob pattern to include files (e.g., "*.py", "**/*.js"). If not specified, all files are considered.
        exclude: Optional glob pattern to exclude files. Exclude takes precedence over include.
        ignore_case: If true, the search will be case-insensitive.
    """
    resolved_path, err = _validate_and_resolve_path(file_path, "read")
    if err:
        return err
    file_path = resolved_path

    cfg = tool_manager.get_config()
    timeout = cfg.file_query_timeout

    path_obj = Path(file_path)
    if not path_obj.is_dir():
        return _enforce_character_limit(
            f"Error: The provided path '{file_path}' is not a directory or does not exist."
        )

    # Determine glob pattern for file selection
    glob_pattern = (
        include if include else "**/*"
    )  # Search all files recursively if no include pattern is given

    found_matches = []
    try:
        # Get all files matching the include pattern
        all_files = asyncio.run(
            run_killable(lambda: list(path_obj.glob(glob_pattern)), timeout=timeout)
        )
        if all_files is None:
            raise TimeoutError("File globbing for grep took too long and timed out")

        # Filter out excluded files and non-allowed files
        files_to_search = []
        for p in all_files:
            if not p.is_file():  # Only search files
                continue
            if not _is_allowed(p):  # Check if file is allowed to be read
                continue

            # Apply exclude pattern if present
            if exclude and p.match(exclude):
                continue
            files_to_search.append(p)

        if not files_to_search:
            return _enforce_character_limit(
                f"No files found matching pattern '{glob_pattern}' (and not matching exclude '{exclude}' if provided) in '{file_path}'."
            )

        regex_flags = re.IGNORECASE if ignore_case else 0
        compiled_pattern = re.compile(pattern, regex_flags)

        for current_file_path in files_to_search:
            try:
                # Read content and search
                lines = current_file_path.read_text().splitlines()
                for i, line in enumerate(lines):
                    if compiled_pattern.search(line):
                        found_matches.append(f"{current_file_path}:{i + 1}: {line}")
            except Exception as e:
                # Handle individual file read errors gracefully
                logger.warning(f"Could not read file {current_file_path}: {e}")
                continue  # Skip to the next file

    except TimeoutError:
        return _enforce_character_limit("Error: Grep operation timed out.")
    except Exception as e:
        return _enforce_character_limit(f"Error during grep operation: {e}")

    if not found_matches:
        return _enforce_character_limit(
            f"No matches found for pattern '{pattern}' in '{file_path}'."
        )
    return _enforce_character_limit("\n".join(found_matches))


@tool
def file_directory_contents(
    directory_path: str | Artifacts, depth: int = 1, pattern: Optional[str] = None
) -> str:
    """
    Lists the files and subdirectories within a given path, optionally with depth control and pattern filtering.

    Args:
        directory_path: The absolute path to the directory or an artifact name.
        depth: The maximum depth of subdirectories to traverse. Defaults to 1 (only immediate children).
               Set to 0 or a negative number to disable depth limit.
        pattern: Optional glob pattern to filter files and directories (e.g., "*.py", "docs/*").

    Returns:
        str: A formatted list of directory contents.
    """

    resolved_path, err = _validate_and_resolve_path(directory_path, "read")
    if err:
        return err
    directory_path = resolved_path

    path_obj = Path(directory_path)
    if not path_obj.is_dir():
        return _enforce_character_limit(
            f"Error: The provided path '{directory_path}' is not a directory or does not exist."
        )

    cfg = tool_manager.get_config()
    all_found_items = []
    timeout = cfg.file_query_timeout

    try:
        # Define the function to collect items for killable execution
        def collect_items():
            result_list = []
            # Use a list as a queue for BFS: (path_object, current_level)
            queue = [(path_obj, 0)]

            while queue:
                current_dir, current_level = queue.pop(0)  # BFS

                # Check if we should explore contents of this directory
                # For depth = 1, we list children of root (level 1) but don't add their children to queue.
                # For depth = 0 or negative, we explore infinitely.
                # For depth > 0, we explore if current_level is less than depth.
                should_explore_children = (depth <= 0) or (current_level < depth)

                for item in current_dir.iterdir():
                    if not _is_allowed(item):
                        continue

                    if pattern and not item.match(pattern):
                        continue

                    # Format path relative to the initial directory_path
                    relative_path = item.relative_to(path_obj)
                    if item.is_dir():
                        result_list.append(f"[DIR] {relative_path}")
                        if should_explore_children:
                            queue.append((item, current_level + 1))
                    else:
                        result_list.append(f"[FILE] {relative_path}")
            return result_list

        all_found_items = asyncio.run(run_killable(collect_items, timeout=timeout))

        if all_found_items is None:  # Timeout occurred
            raise TimeoutError("Directory listing took too long and timed out")

        if not all_found_items:
            return _enforce_character_limit(
                f"No contents found in '{directory_path}' matching the criteria."
            )
        return _enforce_character_limit("\n".join(all_found_items))

    except TimeoutError:
        return _enforce_character_limit(
            f"Error: Listing directory '{directory_path}' timed out."
        )
    except Exception as e:
        return _enforce_character_limit(
            f"Error listing directory '{directory_path}': {e}"
        )
