"""
Linux specific tools for the AI to read files.
"""

import subprocess
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, create_model

from core.config import tool_manager
from core.helpers import enforce_character_limit


def get_run_command_schema() -> type[BaseModel]:
    config = tool_manager.get_config()
    allowed_commands: list[str] = config.linux_read_allowed_commands
    default_timeout: float = config.linux_read_command_timeout

    # Dynamically build the inner Command model with the fresh allowed list
    CommandModel = create_model(
        "Command",
        program=(
            str,
            Field(
                description="The absolute path of the program to run.",
                json_schema_extra={"enum": allowed_commands},  # pyright: ignore[reportArgumentType]
            ),
        ),
        args=(
            list[str] | None,
            Field(None, description="List of arguments for the program."),
        ),
    )

    # Dynamically build the outer schema including the timeout
    DynamicSchema = create_model(
        "RunCommandSchema",
        programs=(
            list[CommandModel],
            Field(
                description="List of commands to execute safely. Piping works by sequential order."
            ),
        ),
        timeout=(
            float,
            Field(
                default=default_timeout,
                description="Per-command timeout to mitigate hanging.",
            ),
        ),
    )
    return DynamicSchema


def _run_command_safe(
    programs: Any,  # it's a dynamic pydantic model
    timeout: float = tool_manager.get_config().linux_read_command_timeout,
) -> str:
    # Bytes or string?
    # For now let's use strings.
    # Maybe if we encounter some decoding issue we will use bytes internally.
    allowed_commands = tool_manager.get_config().linux_read_allowed_commands
    piped = ""
    for i, program in enumerate(programs):
        p = program.program
        args = program.args
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
        if len(programs) == 1 and programs[0].program == "cat"
        else ""
    )

    # At the end piped is just the output of the last program.
    return enforce_character_limit(piped) + cat


@tool(
    args_schema=get_run_command_schema(),
    description="""This tool allows running the given white listed CLI programs.
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
run_command_safe({"program": "ls"})""",
)
def run_command_safe(
    programs: list[dict[str, str | list[str]]],
    timeout: float = tool_manager.get_config().linux_read_command_timeout,
) -> str:
    return _run_command_safe(programs, timeout)
