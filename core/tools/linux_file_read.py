"""
Linux specific tools for the AI to read files.
"""

import subprocess
from typing import Literal, TypedDict

from langchain_core.tools import tool

from core.config import tool_manager

# Is set every time this module is reloaded, I think that's frequent enough.
allowed_commands = tool_manager.get_config().linux_read_allowed_commands

type PossibleProgram = Literal[*allowed_commands]  # pyright: ignore[reportInvalidTypeForm]


class Command(TypedDict):
    program: PossibleProgram
    args: list[str] | None


@tool
def run_command(programs: list[Command]) -> str:
    """
    This tool allows running the given read-only CLI programs.
    To pipe input you must put list the programs in order.
    This does not use bash to execute the tools, but a secure system to check for the executed commands.
    Always use absolute paths.
    Piping works by piping stdout or if stdout is empty, stderr is piped.

    Args:
        programs: list of program objects
        program: One single program to run, under is a breakdown of the keys
            program: the name of the program you want to run
            args: list of arguments to the given program

    Examples:
    Refer to the provided JSON schema for the actual format.

    # simple
    run_command(
        {
            "program": "file",
            "args": ["/some/file"]
        }
    )

    # piping
    run_command(
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
    run_command({"program": "ls"})
    """
    # create per-command timeout to config
    # create configurable ALLOWED_PROGRAMS to config
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

        out = subprocess.run([p, *args], text=True, capture_output=True, input=piped)
        if out.returncode != 0:
            return f"Error running {p}: {out.stdout or out.stderr}"
        piped = out.stdout or out.stderr

    # At the end piped is just the output of the last program.
    return piped
