from pathlib import Path
from pydantic import BaseModel

from global_tools import ConfigManager


class GuiConfig(BaseModel):
    default_collapse_state: bool = False
    end_space: bool = True
    tooltip_len: int = 500


manager = ConfigManager(
    Path("/home/luuppi/Documents/coding/projects/null_shift/gui/config.toml"),
    GuiConfig(),
)
