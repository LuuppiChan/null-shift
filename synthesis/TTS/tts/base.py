from abc import ABC, abstractmethod
from typing import Any, Iterator


class BaseTTSProvider(ABC):
    """
    Abstract interface for Text-to-Speech providers.
    """

    @abstractmethod
    def speak(self, text: str, wait: bool = True, stop_existing: bool = True) -> None:
        """Synthesize and play a single string of text."""
        pass

    @abstractmethod
    def speak_stream(
        self, sentence_generator: Iterator[str], stop_existing: bool = True
    ) -> None:
        """Synthesize and play a stream of sentences."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop all active speaking processes."""
        pass

    @abstractmethod
    def is_speaking(self) -> bool:
        """Return True if the engine is currently playing audio."""
        pass
