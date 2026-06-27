from abc import ABC, abstractmethod
from dataclasses import dataclass
import io


@dataclass
class AudioFile:
    data: io.BytesIO


class BaseTTSBackend(ABC):
    @abstractmethod
    def generate(self, text: str) -> AudioFile:
        """Synthesize the entire text string into a complete audio file."""
