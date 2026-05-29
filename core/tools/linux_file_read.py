"""
Linux specific tools for the AI to read files.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.tools import tool

from core.config import tool_manager
from core.helpers import enforce_character_limit

# Is set every time this module is reloaded, I think that's frequent enough.
allowed_commands = tool_manager.get_config().linux_read_allowed_commands

type PossibleProgram = Literal[
    *allowed_commands  # pyright: ignore[reportInvalidTypeForm]
]


class Command(TypedDict):
    program: PossibleProgram | str
    args: list[str] | None


def _run_command_safe(
    programs: list[Command],
    timeout: float = tool_manager.get_config().linux_read_command_timeout,
) -> str:
    # Bytes or string?
    # For now let's use strings.
    # Maybe if we encounter some decoding issue we will use bytes internally.
    piped = ""
    for i, program in enumerate(programs):
        p = program.get("program")
        args = program.get("args")
        if p is None:
            return f"Error: Program at index {i} has no program field."
        if args is None:
            args = []

        if p not in allowed_commands:
            return f"Error: {p} at the index of {i} is not in the allowed programs."

        out = subprocess.run(
            [p, *args], text=True, capture_output=True, input=piped, timeout=timeout
        )
        if out.returncode != 0:
            return f"Error running '{p}': {enforce_character_limit(out.stdout or out.stderr or '(No output)')}"
        piped = out.stdout or out.stderr

    cat = (
        "\n\n[SYSTEM]: Using cat to read files is highly discouraged, use the dedicated file_read tool."
        if len(programs) == 1 and programs[0].get("program") == "cat"
        else ""
    )

    # At the end piped is just the output of the last program.
    return enforce_character_limit(piped) + cat


@tool
def run_command_safe(
    programs: list[Command],
    timeout: float = tool_manager.get_config().linux_read_command_timeout,
) -> str:
    """
    This tool allows running the given white listed CLI programs.
    This tool is NOT a common program execution tool, but a tool that gives you limited access to the user's system.
    This tool is annotated with the allowed programs and the programs are most commonly read-only.

    To pipe input you must put list the programs in order.
    This does not use bash to execute the tools, but a secure system to check for the executed commands.
    Always use absolute paths.
    Piping works by piping stdout or if stdout is empty, stderr is piped.

    Args:
        programs: list of program objects
            program object: One single program to run, under is a breakdown of the keys
                program: the name of the program you want to run
                args: list of arguments to the given program
        timeout: Per-command timeout to mitigate hanging.

    Examples:
    Refer to the provided JSON schema for the actual format.

    # simple
    run_command_safe(
        {
            "program": "file",
            "args": ["/some/file"]
        }
    )

    # piping
    run_command_safe(
        {
            "program": "cat",
            "args": ["/some/path/to/*.txt"]
        },
        {
            "program": "grep",
            "args": ["-i", "hello"]
        }
    )

    # no args
    run_command_safe({"program": "ls"})
    """
    return _run_command_safe(programs, timeout)


# Make this to support scratchpad instead of hard-coded desktop
def _is_in_desktop(target_path: Path) -> bool:
    """
    A separate check to verify if the given path is located
    somewhere inside the current Linux user's Desktop directory.
    """
    desktop_dir = (Path.home() / "Desktop").resolve()

    # .is_relative_to() checks if a path is a child of another path (Requires Python 3.9+)
    return target_path.is_relative_to(desktop_dir)


@tool
def decompress_file(input_path: str, target_path: str) -> str:
    """
    Uncompresses a file to a target path.
    Currently only Desktop file paths are allowed.
    Fails if target path exists.

    Args:
        input_path: Absolute path to the compressed archive (e.g., .zip, .tar.gz).
        target_path: Destination path where contents will be extracted.
    """
    # 1. Expand user (e.g., '~') and resolve to an absolute path for the input
    resolved_input = Path(input_path).expanduser().resolve()

    if not resolved_input.is_file():
        raise FileNotFoundError(
            f"The input compressed file was not found: {resolved_input}"
        )

    # Prepare the target path
    resolved_target = Path(target_path).expanduser().resolve()

    # 2. Check if the target path already exists; error out if it does
    if resolved_target.exists():
        raise FileExistsError(
            f"Target path '{resolved_target}' already exists. Aborting operation."
        )

    # 3. Perform the separate check to ensure the target is inside the Desktop
    if not _is_in_desktop(resolved_target):
        raise PermissionError(
            f"Target path '{resolved_target}' is not within the current user's Desktop directory."
        )

    # 4. Uncompress the file
    # shutil.unpack_archive creates the target directory and extracts contents automatically
    shutil.unpack_archive(str(resolved_input), str(resolved_target))

    # 5. Return a string stating what was done
    return f"Success: Uncompressed '{resolved_input.name}' into the directory '{resolved_target}'."
