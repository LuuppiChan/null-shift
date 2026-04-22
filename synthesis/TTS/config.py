import tomllib
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class PiperModelConfig(BaseModel):
    model: str
    config: str

class BlockRule(BaseModel):
    name: str
    start_tag: str
    end_tag: str
    replacement: str = ""

class TTSConfig(BaseModel):
    zmq_output_bind: str = "tcp://localhost:5556"
    default_tts_lang: str = "en"
    voice_rate: float = 1.0
    voice_volume: float = 1.0
    tts_language_detection_min_chars: int = 5
    piper_models: Dict[str, PiperModelConfig]
    cleaning_blocks: List[BlockRule] = []
    cleaning_replacements: Dict[str, str] = {}

def load_config() -> TTSConfig:
    config_path = Path(__file__).parent / "tts_config.toml"
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    return TTSConfig(**data)

config = load_config()
