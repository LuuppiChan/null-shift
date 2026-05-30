import asyncio
import logging
import queue
import threading
import time

from thefuzz import fuzz, process

from global_tools import Signal
from gui.config import manager
from gui.stt.listen import MicrophoneListener
from gui.stt.understand import WhisperSTT

logger = logging.getLogger("Transcriber")


class Transcriber:
    def __init__(self):
        self.on_input = Signal(str)
        self.whisper = WhisperSTT()
        self.mic_listener = MicrophoneListener()
        self.running = False
        self.transcriber_thread: threading.Thread
        self._result_queue: queue.Queue[str] = queue.Queue()
        self.async_emitter_task: asyncio.Task | None = None

    async def async_emitter(self):
        """
        Starts an async signal emitter.
        This is used because signals don't go through threads.
        """

        async def listen():
            logger.info("Emitter started")
            try:
                while self.running:
                    try:
                        text = self._result_queue.get_nowait()
                        logger.info("Got text from queue: %s", text)
                        self.on_input.emit(text)
                        logger.info("Emitter: signal emitted")
                    except queue.Empty:
                        await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                logger.info("Emitter task cancelled")
                return
            except Exception as e:
                logger.error("Emitter task crashed: %s", e, exc_info=True)
                raise

        self.async_emitter_task = asyncio.create_task(listen())

    async def start(self):
        self.running = True
        self.transcriber_thread = threading.Thread(
            target=self._transcription_worker, daemon=True
        )
        await self.async_emitter()
        self.mic_listener.start()
        self.transcriber_thread.start()

    async def stop(self):
        self.running = False
        self.mic_listener.stop()
        if self.async_emitter_task is not None:
            self.async_emitter_task.cancel()
        self.transcriber_thread.join()

    def _transcription_worker(self):
        # Wait for calibration
        while not self.mic_listener.calibrated and self.running:
            time.sleep(0.1)

        while self.running:
            try:
                # We use a short timeout so we can periodically check `self.running` and clear events
                try:
                    msg_type, data = self.mic_listener.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    # Periodically reset speech onset event just in case it triggers without a phrase completion
                    if self.mic_listener.speech_event.is_set():
                        self.mic_listener.speech_event.clear()
                    continue

                if msg_type == "audio" and data:
                    text = self.whisper.transcribe(data).strip()
                    logger.info(f"Raw Transcribed: {text}")
                    if text:
                        # Clear speech event if there was one
                        self.mic_listener.speech_event.clear()

                        cfg = manager.get_config()
                        result = process.extractOne(
                            text.lower(), cfg.voice.ignore_list, scorer=fuzz.ratio
                        )
                        if result:
                            match, score, *_ = result
                            ignored = cfg.voice.ignore_fuzz_ratio <= score
                            if ignored:
                                logger.info(
                                    "Input matches '%s' from the ignore list with a score of %s",
                                    match,
                                    score,
                                )
                            continue

                        self._result_queue.put(text)

            except Exception as e:
                logger.error(f"Transcription worker error: {e}", exc_info=True)
