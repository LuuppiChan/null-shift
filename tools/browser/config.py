from pathlib import Path
from pydantic import BaseModel, ConfigDict

from global_tools import ConfigManager


class BrowserConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    socket_path: str = "tcp://*:5557"


manager = ConfigManager(Path("./browser_config.toml"), BrowserConfig())
