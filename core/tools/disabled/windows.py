import asyncio
import base64
import json
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Optional

import zmq.asyncio
import libvirt
import libvirt_qemu
from langchain_core.tools import tool
from openai import BaseModel
from pydantic import ConfigDict

from core.config import tool_manager

logger = logging.getLogger(__name__)
PASSWORD = "super secret password that nobody will guess"


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    password: str
    background: bool = False
    command: list[str]
    timeout: Optional[float] = 120.0


class Response(BaseModel):
    error: Optional[str] = None
    llm_message: str


async def send(
    command: list[str],
    password: str,
    timeout: Optional[float] = 120.0,
    background: bool = False,
) -> str:
    """
    Send a command to the message server on Windows guest.
    Runs on Session 1.
    """
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect("tcp://192.168.122.71:5550")
    # This breaks
    # sock.setsockopt(zmq.RCVTIMEO, int(timeout or 270 + 30))
    msg = Message(
        command=command, password=password, timeout=timeout, background=background
    )
    await sock.send(msg.model_dump_json().encode())
    try:
        frame = await sock.recv()
    except zmq.error.Again:
        logger.error("Message timed out")
        return (
            "[SYSTEM INTERVENTION]: "
            "Timed out while waiting response from Windows. "
            "This is not your fault. "
            "Inform the user immediately of the error."
        )

    data = json.loads(frame.decode(errors="replace"))
    res = Response(**data)
    if res.error:
        logger.error(res.error)
    sock.close()
    ctx.destroy()
    return res.llm_message.strip()


def _enforce_character_limit(text: str) -> str:
    """Enforce character limit on text outputs."""
    cfg = tool_manager.get_config()
    limit = cfg.file_absurd_size_limit
    if limit and len(text) > limit:
        return (
            text[:limit] + f"\n... (output truncated due to character limit of {limit})"
        )
    return text


def run_command(
    path: str,
    args: list[str],
    timeout: Optional[float] = 120.0,
    background: bool = False,
) -> str:
    """
    Runs a command in the virtual machine.
    Lol same Session 0 issue as SSH.
    """
    try:
        conn = libvirt.open("qemu:///system")
        dom = conn.lookupByName("agent-playground")
    except libvirt.libvirtError as e:
        logger.critical(f"Libvirt failed to open: {e}")
        return "Fatal system error while opening command. This is not your fault. Report immediately to the user."

    cmd = {
        "execute": "guest-exec",
        "arguments": {
            "path": path,
            "arg": args,
            "capture-output": True,
        },
    }

    try:
        response = libvirt_qemu.qemuAgentCommand(dom, json.dumps(cmd), 10, 0)
        res_dict = json.loads(response)
        pid = res_dict["return"]["pid"]
        logger.info("Started process with PID: %s", pid)

        if timeout:
            end = datetime.now() + timedelta(seconds=timeout)
        else:
            end = None

        while not background:
            status_cmd = {
                "execute": "guest-exec-status",
                "arguments": {"pid": int(pid)},
            }
            status_response = libvirt_qemu.qemuAgentCommand(
                dom, json.dumps(status_cmd), 10, 0
            )
            status_res: dict = json.loads(status_response)

            if status_res.get("return", {}).get("exited"):
                out_b64 = status_res["return"].get("out-data", "")
                err_b64 = status_res["return"].get(
                    "err-data", ""
                )  # Always check stderr too

                out = base64.b64decode(out_b64).decode()
                err = base64.b64decode(err_b64).decode()
                res = out or err
                break  # Exit the loop once the process finishes
            # True if it has passed
            elif end and end < datetime.now():
                logger.warning("Process timed out.")
                res = "Error: Process timed out."
                break

            sleep(0.01)  # 0.01 is a bit aggressive for the Guest Agent
        else:
            res = f"Started process with PID: {pid}"

    except libvirt.libvirtError as e:
        logger.critical("Communication Error: %s", e)
        res = "Fatal system error while opening command. This is not your fault. Report immediately to the user."

    conn.close()
    return res


@tool
def run_powershell(
    code: str, timeout: Optional[float] = 120.0, background: bool = False
) -> str:
    """
    Run a PowerShell script and get the output.
    This tool is considered legacy due to its limitations. Prefer python.

    Args:
        code: The code to be written to PowerShell.
        timeout: An optional timeout which will make the tool return if the command takes too long. (Default, 120.0)
    """
    res = asyncio.run(
        send(["powershell.exe", "-Command", code], PASSWORD, timeout, background)
    )
    return _enforce_character_limit(res)


@tool
def run_python(
    code: str, timeout: Optional[float] = 120.0, background: bool = False
) -> str:
    """
    Run a Python script and get the output.
    This code is run in the global interpreter.
    This tool was made to address flaws in the run_powershell tool.
    - This tool has no maximum character limit.
    - This tool can do anything that run_powershell can do.

    Args:
        code: Code to run.
        timeout: An optional timeout which will make the tool return if the command takes too long. (Default, 120.0)
        background: Whether to detach the process and leave it running on the background.
    """
    background_code = """
import subprocess

subprocess.Popen(
    ["python", "C:\\tmp.py"],
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
"""
    path = Path("/home/luuppi/vm_drive/tmp.py")
    path.write_text(code)

    if background:
        bg = Path("/home/luuppi/vm_drive/bg.py")
        bg.write_text(background_code)
        command = ["python", "C:\\bg.py"]
    else:
        command = ["python", "C:\\tmp.py"]

    res = asyncio.run(send(command, PASSWORD, timeout, background))

    if background:
        bg.unlink(True)  # pyright: ignore[reportPossiblyUnboundVariable]
        if not res:
            res = "Background process initiated. Output is not captured. Use a log file if you need to track progress."

    path.unlink(True)
    return _enforce_character_limit(res)


# Mount agent wm
# sshfs "AI Agent"@192.168.122.71:C:/ ~/vm_drive -o IdentityFile=~/.ssh/id_ed25519


def run_powershell_old(code: str, timeout: Optional[float] = 120.0) -> str:
    """
    Run a PowerShell script and get the output.
    This tool is considered legacy due to its limitations. Prefer python.

    Args:
        code: The code to be written to PowerShell.
        timeout: An optional timeout which will make the tool return if the command takes too long. (Default, 120.0)
    """
    command = "$ProgressPreference = 'SilentlyContinue';" + code
    encoded = base64.b64encode(command.encode("utf-16-le")).decode()
    command = f"powershell.exe -NonInteractive -EncodedCommand {encoded}"
    try:
        out = subprocess.run(
            [
                "ssh",
                "-i",
                "~/.ssh/id_ed25519",
                "AI Agent@192.168.122.71",
                command,
            ],
            text=True,
            check=True,
            capture_output=True,
            timeout=timeout,
        )
        res = out.stdout or out.stderr
    except subprocess.TimeoutExpired as e:
        res = f"Timeout: {e.stdout or e.stderr}"
    except subprocess.CalledProcessError as e:
        res = f"Error running command: {e.stdout or e.stderr}"

    return _enforce_character_limit(res)


def run_python_old(
    code: str, timeout: Optional[float] = 120.0, background: bool = False
) -> str:
    """
    Run a Python script and get the output.
    This code is run in the global interpreter.
    This tool was made to address flaws in the run_powershell tool.
    - This tool has no maximum character limit.
    - This tool can do anything that run_powershell can do.
    - This tool can spawn background processes.

    Args:
        code: Code to run.
        timeout: An optional timeout which will make the tool return if the command takes too long. (Default, 120.0)
        background: Whether to detach the process and leave it running on the background.
    """
    background_code = """
import subprocess

subprocess.Popen(
    ["python", "C:\\tmp.py"],
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
"""
    path = Path("/home/luuppi/vm_drive/tmp.py")
    path.write_text(code)

    if background:
        bg = Path("/home/luuppi/vm_drive/bg.py")
        bg.write_text(background_code)
        command = "python C:\\bg.py"
    else:
        command = "python C:\\tmp.py"

    try:
        out = subprocess.run(
            [
                "ssh",
                "-i",
                "~/.ssh/id_ed25519",
                "AI Agent@192.168.122.71",
                command,
            ],
            text=True,
            check=True,
            capture_output=True,
            timeout=timeout,
        )
        res = out.stdout or out.stderr
    except subprocess.TimeoutExpired as e:
        res = f"Timeout: {e.stdout or e.stderr}"
    except subprocess.CalledProcessError as e:
        res = f"Error running command: {e.stdout or e.stderr}"

    if background:
        bg.unlink(True)  # pyright: ignore[reportPossiblyUnboundVariable]
        if not res:
            res = "Background process initiated. Output is not captured. Use a log file if you need to track progress."

    path.unlink(True)
    return _enforce_character_limit(res)
