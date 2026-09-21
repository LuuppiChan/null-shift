from pathlib import Path

from pydantic import BaseModel, ConfigDict

from global_tools import ConfigManager


class BrowserConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    socket_path: str = "tcp://127.0.0.1:5557"
    typing_delay_ms: float = 5
    typing_timeout_ms: float = 30_000


manager = ConfigManager(Path("./browser_config.toml"), BrowserConfig())
