import json
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Any, Literal
from langchain_core.tools import tool

from core.config import tool_manager
from core.helpers import compress_image, get_permission


@tool(
    description="""Gets currently open windows and data related to them in a list of objects.
Data includes:
- id: Identifier for that specific window
- title
- app_id
- pid
- workspace_id: Tells what workspace id is the windows located at
- is_focused
- is_floating: Whether the window is floating or tiled
- is_urgent: Whether the window wants attention
- layout: Physical size properties"""
)
def get_windows() -> str:
    output = subprocess.run(
        ["niri", "msg", "-j", "windows"],
        text=True,
        timeout=0.5,
        check=True,
        capture_output=True,
    ).stdout
    windows = json.loads(output)
    # re-format the data.
    return json.dumps(windows, indent=2)


@tool(
    description="""Focuses a given window in the user's environment based on the window's id.
Use only when the user requests the usage.

Args:
    id: Number identifier for the window."""
)
def focus_window(id: int) -> str:
    try:
        config = tool_manager.get_config()
        if config.niri_focus_window_permission:
            if not get_permission(config.niri_focus_window_prompt.format(id=id)):
                return "User declined the request to switch window focus."

        _ = subprocess.run(
            ["niri", "msg", "action", "focus-window", "--id", str(id)],
            check=True,
            text=True,
            capture_output=True,
        )
        return f"Focused window {id}."
    except subprocess.CalledProcessError as e:
        return f"Error focusing window: {e.stdout}\n{e.stderr}"


@tool(description="""Sets monitors either on or off.""")
def set_monitor(state: Literal["on", "off"]) -> str:
    try:
        config = tool_manager.get_config()
        if config.niri_set_monitor_permission:
            if not get_permission(config.niri_set_monitor_prompt.format(state=state)):
                return f"User declined the request to turn {state} the monitors."

        subprocess.run(["niri", "msg", "action", f"power-{state}-monitors"], check=True)
        return f"Monitors are now {state}."
    except subprocess.CalledProcessError as e:
        return f"Error setting monitors {state}: {e}"


# disabled because causes confusion with browser screenshotting
# @tool(
#     description="""Returns a screenshot of the currently focused monitor or window.
# This tool is for when the user asks you to "look" or "see" something.
#
# # Examples
# - "Look at this. Isn't it beautiful?" -> yes
# - "Can you see what I'm looking at?" -> yes
# - "What do I do in this task?" -> yes, if system prompt doesn't provide any other clue, if using read_screen tool doesn't provide with anything useful try to use other tools to gather context
# - "Look what you've done." -> no, this is clearly a rhetoric saying
# - "Can you see me?" -> maybe, this reads only the screen and cannot read a webcam, user might be referring to some other tool, but fall back to this if no webcam tool is found. Also note the system context such as focused window for a clue.
# - "Let's see..." -> no, this doesn't actually have anything to do with seeing
# - "Does this work?" -> likely, choose your answer based on the conversation history. As a rule of thumb expect that the user wants you to take a new screenshot."""
# )
def read_screen(area: Literal["monitor", "window"] = "monitor") -> list[dict[str, Any]]:
    target = "screenshot-screen" if area == "monitor" else "screenshot-window"

    with NamedTemporaryFile(suffix=".png") as file:
        path = Path(file.name)
        subprocess.run(
            [
                "niri",
                "msg",
                "action",
                target,
                "--path",
                path,
                "--write-to-disk",
                "true",
            ],
            check=True,
            capture_output=True,
        )
        # Important!
        # The niri tool returns before the screenshot is written.
        # Half a second isn't enough. 1 second or this poller.
        while path.stat().st_size == 0:
            sleep(0.1)
        data = compress_image(path)
    msg = [
        {"type": "text", "text": "The screenshot has been attached to this message."},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{data}"},
        },
    ]
    return msg


# Some other actions might be added. But what though?
# These are already pretty useless.
# THe set monitor off being the most useful.


@tool(description="""Set sound volume for the user.""")
def set_volume(percentage: float, relative: bool = False) -> str:
    percentage = percentage / 100
    if isinstance(percentage, float):
        percentage = min(max(percentage, 1), 0)

    rel = ""
    if relative:
        if percentage < 0:
            rel = f"{abs(percentage)}-"
        else:
            rel = f"{percentage}+"

    return (
        subprocess.run(
            ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", str(percentage) + rel],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        or f"Volume set to {percentage}."
    )
