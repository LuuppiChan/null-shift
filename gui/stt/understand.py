import logging
import sys

import numpy as np
from pywhispercpp.model import Model as WhisperModel
import speech_recognition as sr

from gui.config import manager


logger = logging.getLogger("Understand")

class WhisperSTT:
    """Wrapper around pywhispercpp Model."""

    def __init__(self):
        config = manager.get_config()
        logger.info(
            f"Initializing GGML model {config.voice.stt_model_path} via pywhispercpp..."
        )
        self.model = WhisperModel(
            model=config.voice.stt_model_path,
            n_threads=config.voice.stt_threads,
            redirect_whispercpp_logs_to=sys.stdout,
        )
        logger.info("Whisper model initialized.")

    def transcribe(self, audio_data: sr.AudioData) -> str:
        try:
            audio_bytes = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
            audio_np = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )

            segments = self.model.transcribe(audio_np)
            text = " ".join([seg.text for seg in segments]).strip()
            return text
        except Exception as e:
            logger.error(f"Whisper STT error: {e}")
            return ""
