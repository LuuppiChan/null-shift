"""
Enabling this tool allows the assistant full access to your computer.
"""

import subprocess
from langchain_core.tools import tool


@tool
def run_command(cmd: list[str], timeout: float | None = 10.0) -> str:
    """
    Runs an arbitrary command using python subprocess.

    The this tool is really simple:
    ```python
    p = subprocess.run(cmd, timeout=timeout, check=True, capture_output=True, text=True)
    return p.stdout or p.stderr
    ```
    """
    p = subprocess.run(cmd, timeout=timeout, check=True, capture_output=True, text=True)
    return p.stdout or p.stderr
