import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import logging
import tempfile
import threading
from typing import AsyncIterable

from gui.config import GuiConfig, manager
from gui.tts.backends import AudioFile, BaseTTSBackend
from gui.tts.backends.piper import PiperTTSBackend
from gui.tts.sentence_stream import SentenceStream

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Text to speech"""

    def __init__(self, speech_event: threading.Event) -> None:
        self.backend: BaseTTSBackend
        self.backend_id: str = ""
        self._set_backend()
        self.speaking = False
        self.audio_playback = False
        self.speech_event = speech_event

        self.speak_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.speak_task: asyncio.Task | None = None

        manager.config_updated.connect(self._config_updated)

    async def _read_queue_chunks(self) -> AsyncIterable[str]:
        try:
            while True:
                chunk = await self.speak_queue.get()
                if chunk is None:
                    break
                yield chunk
        except asyncio.CancelledError:
            pass

    async def _speak_canceller_task(self):
        while self.speaking:
            cfg = manager.get_config()
            if cfg.speak.stop_on_voice:
                if self.speech_event.is_set() and self.audio_playback:
                    await self.stop_stream()
                    await self.abort()
                    break
            await asyncio.sleep(0.1)

    async def start_stream(self):
        """Start internal speaking stream."""
        logger.info("Starting tts stream")
        if self.speak_task is not None:
            self.speak_task.cancel()
            await self.speak_task
        while not self.speak_queue.empty():
            self.speak_queue.get_nowait()

        self.speak_task = asyncio.create_task(self.speak(self._read_queue_chunks()))
        asyncio.create_task(self._speak_canceller_task())

    def put_stream(self, text: str):
        """Put text to stream"""
        logger.info("Adding to tts stream (put_stream)")
        if self.speak_task is None:
            # temporary
            # possibly will raise an error or do something
            return
        self.speak_queue.put_nowait(text)

    async def stop_stream(self):
        """Stop internal speaking stream."""
        logger.info("Stopping tts stream gracefully")
        if self.speak_task is not None and not self.speak_task.done():
            await self.speak_queue.put(None)

    async def abort(self):
        """Stop speaking."""
        logger.info("Aborting speech")
        self.speaking = False
        if self.speak_task is not None:
            self.speak_task.cancel()
            await self.speak_task

    def _set_backend(self):
        """
        Sets current backend.

        Args:
            exists: Whether a backend exists or not when called.
        """
        cfg = manager.get_config()
        backend_id = cfg.speak.backend
        # avoid reloading
        if self.backend_id == backend_id:
            return

        match backend_id:
            case "piper":
                self.backend = PiperTTSBackend()

        self.backend_id = backend_id

    def _config_updated(self, _: GuiConfig):
        self._set_backend()

    async def _audio_generator(
        self, chunks: AsyncIterable[str]
    ) -> AsyncIterable[AudioFile]:
        sentences = SentenceStream(chunks)
        with ThreadPoolExecutor() as pool:

            async def submitter():
                async for sentence in sentences.sentences():
                    logger.info("Got sentence. Creating audio genertion task.")
                    if pool._shutdown:
                        return

                    if sentence.strip():
                        future = pool.submit(self.backend.generate, sentence)
                        await futures.put(future)

            futures: asyncio.Queue[Future[AudioFile]] = asyncio.Queue()
            submitter_task = asyncio.create_task(submitter())

            try:
                while not submitter_task.done() or not futures.empty():
                    future = await futures.get()

                    while not future.done():
                        if not self.speaking:
                            raise asyncio.CancelledError
                        await asyncio.sleep(0.05)
                    logger.info("Sentence audio generated")
                    yield future.result()
            except asyncio.CancelledError:
                submitter_task.cancel()
                pool.shutdown(cancel_futures=True)

    async def speak(self, chunks: AsyncIterable[str]):
        """Speak this chunk stream out loud."""
        self.speaking = True
        import pygame

        try:
            pygame.mixer.init()
            async for file in self._audio_generator(chunks):
                if not self.speaking:
                    break

                logger.info("Speaking sentence")
                self.audio_playback = True
                await self._speak(file)
                self.audio_playback = False

                if not self.speaking:
                    break
        finally:
            pygame.mixer.quit()
            self.speaking = False

    async def speak_single(self, text: str):
        """Speaks only this text. Then returns."""

        async def iter():
            yield text

        self.speaking = True
        import pygame

        try:
            pygame.mixer.init()
            async for file in self._audio_generator(iter()):
                if not self.speaking:
                    break

                logger.info("Speaking sentence")
                self.audio_playback = True
                await self._speak(file)
                self.audio_playback = False

                if not self.speaking:
                    break
        finally:
            pygame.mixer.quit()
            self.speaking = False

    async def _speak(self, file: AudioFile):
        import pygame

        with tempfile.NamedTemporaryFile(suffix=".wav") as temp:
            temp.write(file.data.read())
            temp.flush()
            temp.seek(0)

            pygame.mixer.music.load(temp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if not self.speaking:
                    pygame.mixer.music.unload()
                    pygame.mixer.music.stop()
                    return
                await asyncio.sleep(0.05)
            pygame.mixer.music.unload()
