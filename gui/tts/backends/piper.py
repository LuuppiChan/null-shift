import gc
import io
import logging
import queue
from typing import TYPE_CHECKING
import wave


from gui.config import GuiConfig, manager
from gui.tts.backends import AudioFile, BaseTTSBackend

logger = logging.getLogger(__name__)


class PiperTTSBackend(BaseTTSBackend):
    def __init__(self) -> None:
        if TYPE_CHECKING:
            import piper

        cfg = manager.get_config()
        self.config = cfg.speak.piper
        self.model_pool: queue.Queue[piper.PiperVoice] = queue.Queue()
        self.pool_size = cfg.speak.piper.pool_size
        self.first_load = True

    def load_models(self):
        import piper
        logger.info("Loading models")
        if self.first_load:
            logger.info("First load detected")
            manager.config_updated.connect(self.config_updated)
        self.first_load = False

        cfg = manager.get_config()
        for _ in range(self.pool_size):
            model = piper.PiperVoice.load(
                model_path=cfg.speak.piper.model,
                config_path=cfg.speak.piper.config,
                use_cuda=cfg.speak.piper.cuda,
            )
            self.model_pool.put(model)

    def config_updated(self, cfg: GuiConfig):
        if self.config == cfg.speak.piper:
            return
        self.config = cfg.speak.piper

        collected = 0
        while collected != self.pool_size:
            model = self.model_pool.get()
            del model
            collected += 1

        gc.collect()

        self.pool_size = cfg.speak.piper.pool_size
        self.load_models()

    def generate(self, text: str) -> AudioFile:
        if self.first_load:
            self.load_models()

        logger.info("Waiting for a model to be free")
        model = self.model_pool.get()
        logger.info("Generating text")

        try:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(model.config.sample_rate)
                model.synthesize_wav(text, wav_file)
            wav_buffer.seek(0)
            if len(wav_buffer.getvalue()) <= 44:
                logger.error(f"Model failed to generate voice audio data for text: '{text}'")
        finally:
            self.model_pool.put(model)

        logger.info("Returning generated audio file")

        return AudioFile(wav_buffer)
