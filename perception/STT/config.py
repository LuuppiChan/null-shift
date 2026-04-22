"""
STT Configuration

Contains configuration management for the STT node.
"""

import logging
from pathlib import Path
from typing import Any
import tomllib

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

class STTConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    zmq_target_address: str = "tcp://127.0.0.1:5555"

    stt_model_path: str = "/home/luuppi/Documents/coding/python_projects/assistant/models/whisper_cpp/ggml-large-v3-turbo-q5_0.bin"
    stt_threads: int = 12

    voice_pause_threshold: float = 1.2
    voice_phrase_time_limit: float = 60.0
    voice_non_speaking_duration: float = 1.2

    tts_vad_sensitivity: float = 5.0
    tts_vad_consecutive_chunks: int = 3

    voice_ignore_fuzz_ratio: int = 95
    voice_ignore_list: list[str] = Field(default_factory=list)

    log_level: str = "INFO"


def with_overrides(cls: type[BaseModel], config_path: Path) -> BaseModel:
    """Get the config with overrides."""
    overrides: dict[str, Any] = {}

    path = Path(config_path)
    if path.exists():
        try:
            overrides = tomllib.loads(path.read_text())
        except Exception as e:
            logger.error(f"Failed to parse config file: {e}")

    return cls(**overrides)


class ConfigManager:
    def __init__(self, config_path: Path, default_config: STTConfig) -> None:
        self.path = config_path
        self._config: STTConfig = default_config
        self._last_mtime: float = 0.0

    def get_config(self) -> STTConfig:
        """Get the current configuration, reloading if the file has changed."""
        if self.path.exists():
            current_mtime = self.path.stat().st_mtime
            if current_mtime > self._last_mtime:
                self._config = with_overrides(type(self._config), self.path) # type: ignore
                logger.info(
                    "Config was changed %s -> %s. Reloaded.",
                    self._last_mtime,
                    current_mtime,
                )
                self._last_mtime = current_mtime
        else:
            # Try to write default if it doesn't exist to make it easier for user
            pass

        return self._config


manager = ConfigManager(Path("stt_config.toml"), STTConfig())
