"""
Definitions and Specifications: Provide clear, explicit, specific, and complete descriptions of any definitions and/or specifications unique to the problem.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import re
from subprocess import CalledProcessError
from typing import Any, Optional

from core.helpers import PromptHelper, completed, fmt_dict


class SystemContext:
    def __init__(self) -> None:
        window: dict[str, Any] = json.loads(
            completed(
                "bash", "-c", "niri msg -j windows | jq '.[] | select(.is_focused)'"
            ).stdout
        )
        workspace: dict[str, Any] = json.loads(
            completed(
                "bash", "-c", "niri msg -j workspaces | jq '.[] | select(.is_focused)'"
            ).stdout
        )
        try:
            playing_media: dict[str, Any] = json.loads(
                completed(
                    "playerctl",
                    "metadata",
                    "--format",
                    '{"artist":"{{artist}}","title":"{{title}}"}',
                ).stdout
            )
        except Exception:
            playing_media = {}
        media_status: str = completed("playerctl", "status").stdout

        if media_status == "Playing":
            self.playing_media: Optional[dict[str, Any]] = playing_media
        else:
            self.playing_media = None
        self.load1, self.load5, _ = os.getloadavg()
        self.focused_window: dict[str, Any] = window
        self.focused_workspace: dict[str, Any] = workspace
        self.datetime = datetime.now().strftime("%H:%M on %A, %B %d, %Y")
        self.home_path = Path.home()

    def get_bat(self) -> str:
        try:
            for bat_id in ["BAT1", "BAT0"]:
                try:
                    result = completed(
                        "upower",
                        "-i",
                        f"/org/freedesktop/UPower/devices/battery_{bat_id}",
                    )
                    percentage = re.search(r"percentage:\s+(\d+%)", result.stdout)
                    state = re.search(r"state:\s+([\w-]+)", result.stdout)
                    if percentage and state:
                        return f"{percentage.group(1)} ({state.group(1)})"
                except CalledProcessError:
                    pass
        except Exception:
            pass
        return "No battery detected"


def collect() -> str | None:
    ctx = SystemContext()
    prompt = PromptHelper("dynamic_context")
    prompt.add_part(
        "This section contains dynamic data about the surrounding environment meaning it's always up-to-date.\nThis information may or may not be relevant to your task, it is up for you to decide.",
        "description",
    )
    # prompt.add_part(ctx.home_path, "home_path", "Absolute home path")
    prompt.add_part(ctx.datetime, "current_time", "Current date and time")
    # prompt.add_part(
    #     fmt_dict(ctx.focused_window),
    #     "currently_focused_window",
    #     "Currently focused window in JSON",
    # )
    # prompt.add_part(
    #     fmt_dict(ctx.focused_workspace),
    #     "currently_focused_workspace",
    #     "Currently focused workspace in JSON",
    # )
    # if ctx.playing_media:
    #     prompt.add_part(
    #         fmt_dict(ctx.playing_media),
    #         "currently_playing_media",
    #         "Currently playing media in JSON",
    #     )
    #
    # prompt.add_part(f"{ctx.load1} (1m)\n{ctx.load5} (5)", "current_system_load")
    # prompt.add_part(ctx.get_bat(), "current_battery_percentage")
    # prompt.add_part(
    #     completed("wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@").stdout,
    #     "current_media_volume",
    # )
    return prompt.compile()
