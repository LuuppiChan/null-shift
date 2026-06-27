from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

from global_tools import ConfigManager


class Whisper(BaseModel):
    model_path: str = "/home/luuppi/Documents/coding/python_projects/assistant/models/whisper_cpp/ggml-large-v3-turbo-q5_0.bin"
    threads: int = 12


class STTConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    whisper: Whisper = Field(default_factory=Whisper)

    context_injection: str = ""

    pause_threshold: float = 1.2
    phrase_time_limit: float = 60.0
    non_speaking_duration: float = 1.2

    tts_vad_sensitivity: float = 5.0
    tts_vad_consecutive_chunks: int = 3

    ignore_fuzz_ratio: int = 95
    ignore_list: list[str] = Field(default_factory=list)

    replace_filter: dict[str, str] = Field(default_factory=dict)

    wake_word: str | list[str] | None = None


class Piper(BaseModel):
    model: str = ""
    config: str = ""
    cuda: bool = False
    pool_size: int = 3


class TTSConfig(BaseModel):
    backend: Literal["piper"] = "piper"
    always: bool = False
    with_stt: bool = True
    stop_on_voice: bool = True
    audio_feedback: bool = True
    context_injection: str = ""
    replace_filter: dict[str, str] = Field(default_factory=dict)

    piper: Piper = Field(default_factory=Piper)


class GuiConfig(BaseModel):
    default_collapse_state: bool = False
    end_space: bool = True
    end_space_height: int = 100
    tooltip_len: int = 500
    auto_send_voice: bool = False
    send_pdf_bin: bool = False

    voice: STTConfig = Field(default_factory=STTConfig)
    speak: TTSConfig = Field(default_factory=TTSConfig)


manager = ConfigManager(
    Path("/home/luuppi/Documents/coding/projects/null_shift/gui/config.toml"),
    GuiConfig(),
)
