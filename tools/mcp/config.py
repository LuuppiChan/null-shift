from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from global_tools import ConfigManager


class ServerConfig(BaseModel):
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    url: str | None = None


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    socket_path: str = "tcp://127.0.0.1:5558"
    init_timeout_s: float = 30
    call_timeout_s: float = 120
    # Where decoded image content (screenshots) is written. Empty = tempdir.
    image_dir: str = ""
    servers: list[ServerConfig] = Field(default_factory=list)


manager = ConfigManager(Path("./tools/mcp/mcp_config.toml"), MCPConfig())
