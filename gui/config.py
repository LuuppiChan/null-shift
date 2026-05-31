from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

from global_tools import ConfigManager


class STTConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    stt_model_path: str = "/home/luuppi/Documents/coding/python_projects/assistant/models/whisper_cpp/ggml-large-v3-turbo-q5_0.bin"
    stt_threads: int = 12

    pause_threshold: float = 1.2
    phrase_time_limit: float = 60.0
    non_speaking_duration: float = 1.2

    tts_vad_sensitivity: float = 5.0
    tts_vad_consecutive_chunks: int = 3

    ignore_fuzz_ratio: int = 95
    ignore_list: list[str] = Field(default_factory=list)

    wake_word: str | list[str] | None = None


class GuiConfig(BaseModel):
    default_collapse_state: bool = False
    end_space: bool = True
    tooltip_len: int = 500
    auto_send_voice: bool = False

    voice: STTConfig = Field(default_factory=STTConfig)


manager = ConfigManager(
    Path("/home/luuppi/Documents/coding/projects/null_shift/gui/config.toml"),
    GuiConfig(),
)
