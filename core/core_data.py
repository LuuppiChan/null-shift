import collections
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, ConfigDict, Field


from core.agent import AgentData
from core.config import tool_manager


class Data:
    """
    Contains data from the system useful for prompts and tools.
    """

    @staticmethod
    def datetime() -> str:
        return datetime.now().strftime("%H:%M:%S on %A, %d %B, %Y")

    @staticmethod
    def home_path() -> Path:
        return Path.home()

    @staticmethod
    def focused_window() -> str:
        return subprocess.run(
            ["bash", "-c", "niri msg -j windows | jq '.[] | select(.is_focused)'"],
            capture_output=True,
            text=True,
        ).stdout

    @staticmethod
    def focused_monitor() -> str:
        return subprocess.run(
            ["bash", "-c", "niri msg -j workspaces | jq '.[] | select(.is_focused)'"],
            capture_output=True,
            text=True,
        ).stdout

    @staticmethod
    def battery() -> str:
        try:
            for bat_id in ["BAT1", "BAT0"]:
                try:
                    result = subprocess.run(
                        [
                            "upower",
                            "-i",
                            f"/org/freedesktop/UPower/devices/battery_{bat_id}",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    percentage = re.search(r"percentage:\s+(\d+%)", result.stdout)
                    state = re.search(r"state:\s+([\w-]+)", result.stdout)
                    if percentage and state:
                        return f"{percentage.group(1)} ({state.group(1)})"
                except subprocess.CalledProcessError:
                    pass
        except Exception:
            pass
        return "No battery detected"

    @staticmethod
    def load1() -> str:
        return str(os.getloadavg()[0])

    @staticmethod
    def load5() -> str:
        return str(os.getloadavg()[1])

    @staticmethod
    def media_status() -> str:
        return subprocess.run(
            ["playerctl", "status"], capture_output=True, text=True
        ).stdout

    @staticmethod
    def playing_media() -> str:
        return subprocess.run(
            [
                "bash",
                "-c",
                'playerctl metadata --format \'{"artist":"{{artist}}","title":"{{title}}"}\' | jq .',
            ],
            capture_output=True,
            text=True,
        ).stdout

    @staticmethod
    def volume() -> str:
        return subprocess.run(
            ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"],
            capture_output=True,
            text=True,
        ).stdout

    @staticmethod
    def scratchpad() -> Path:
        return (
            Path(tool_manager.get_config().dynamic_scratchpad_path)
            .expanduser()
            .resolve()
        )


data = Data()


class LocalData(BaseModel):
    """
    Contains data about the current stream and a reference to the global data.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    history: collections.deque[BaseMessage] = Field(
        default_factory=lambda: collections.deque(maxlen=1000)
    )

    # just in case global data is arbitrary
    model_config = ConfigDict(arbitrary_types_allowed=True)

    global_data: Data = data
    agent: AgentData = Field(default_factory=AgentData)
    last_compression: AIMessage = Field(default_factory=lambda: AIMessage(""))

    def _add_history(self, message: BaseMessage):
        self.history.append(message)
