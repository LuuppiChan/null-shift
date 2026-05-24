from typing import Any, Iterator
from config import config

class TTSEngine:
    """
    Facade for the Text-to-Speech system.
    """
    def __init__(self) -> None:
        if config.provider == "chattts":
            from tts.chattts_provider import ChatTTSProvider
            self.provider = ChatTTSProvider()
        else:
            from tts.piper_provider import PiperTTSProvider
            self.provider = PiperTTSProvider()


    def speak(self, text: str, wait: bool = True, stop_existing: bool = True) -> None:
        """Synthesizes and plays a single string of text."""
        self.provider.speak(text, wait=wait, stop_existing=stop_existing)

    def speak_stream(self, sentence_generator: Any, stop_existing: bool = True) -> None:
        """Synthesizes and plays a stream of sentences."""
        # Wrap the generator if it's not a proper Iterator
        def gen_wrapper():
            for item in sentence_generator:
                yield item

        self.provider.speak_stream(gen_wrapper(), stop_existing=stop_existing)

    def stop(self) -> None:
        """Stops all active TTS processes."""
        self.provider.stop()

    def is_speaking(self) -> bool:
        """Returns True if the TTS engine is currently playing audio."""
        return self.provider.is_speaking()
