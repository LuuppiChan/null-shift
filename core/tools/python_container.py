import logging
from pathlib import Path
import subprocess
import json

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool(description="""Run arbitrary python code inside an isolated container.
It's recommended you write a file and then run if the script is long or needs to be used multiple times with changes made in between.
Tip: Print the result to see it.
**You cannot interact with the operating system with this tool meaning you cannot read external files.**""")
def python_container(code_or_path: str, timeout_seconds: float = 20.0) -> str:
    try:
        if Path(code_or_path).is_file():
            code_or_path = Path(code_or_path).read_text()
    except Exception as e:
        logger.warning("Error while making a path: %s", e)

    podman_cmd: list[str] = [
        "podman",
        "run",
        "--rm",
        # "--network",
        # "none",
        "--memory",
        "512m",
        "--cpus",
        "1.0",
        "--cap-drop=all",
        "agent-sandbox",
        code_or_path,
    ]

    try:
        # Run the container and capture outputs
        result = subprocess.run(
            podman_cmd, capture_output=True, text=True, timeout=timeout_seconds
        )

        return "Stdout:\n" + result.stdout + "Stderr:\n" + result.stderr

    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout_seconds} seconds.",
                "exit_code": 124,
            },
            indent=2,
        )
