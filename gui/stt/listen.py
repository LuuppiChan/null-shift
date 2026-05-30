import logging
import math
import queue
import struct
import subprocess
import threading
import time
from typing import Any

import speech_recognition as sr

from gui.config import manager

logger = logging.getLogger("Listen")
vad_logger = logging.getLogger("VAD")

class VADMonitor:
    """Raw VAD for speech onset using low-latency parec/pacat."""

    def __init__(self, recognizer: sr.Recognizer, speech_event: threading.Event):
        self.recognizer = recognizer
        self.speech_event = speech_event
        self.running = True
        self.thread: threading.Thread

    def start(self):
        self.thread = threading.Thread(target=self._vad_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _vad_loop(self):
        VAD_CHUNK = 512
        proc = None
        for cmd, name in [
            (
                [
                    "parec",
                    "--format=s16le",
                    "--rate=16000",
                    "--channels=1",
                    "--latency-msec=20",
                ],
                "parec",
            ),
            (
                [
                    "pacat",
                    "-r",
                    "--format=s16le",
                    "--rate=16000",
                    "--channels=1",
                    "--latency-msec=20",
                ],
                "pacat",
            ),
        ]:
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                vad_logger.info(f"Speech onset detection started ({name}).")
                break
            except FileNotFoundError:
                continue

        if proc is None:
            vad_logger.warning(
                "parec/pacat not found. Falling back to phrase-level detection only."
            )
            return

        consecutive_above = 0
        try:
            while self.running:
                if proc.stdout is not None:
                    data = proc.stdout.read(VAD_CHUNK * 2)
                else:
                    break

                if not data:
                    time.sleep(0.01)
                    continue

                if self.speech_event.is_set():
                    consecutive_above = 0
                    continue

                shorts = struct.unpack("%dh" % (len(data) // 2), data)
                rms = (
                    math.sqrt(sum(s * s for s in shorts) / len(shorts)) if shorts else 0
                )

                config = manager.get_config()
                # Multiply threshold by sensitivity
                threshold = (
                    self.recognizer.energy_threshold * config.voice.tts_vad_sensitivity
                )

                if rms > threshold:
                    consecutive_above += 1
                    required = config.voice.tts_vad_consecutive_chunks
                    vad_logger.debug(
                        f"rms={rms:.0f} th={threshold:.0f} chunk {consecutive_above}/{required}"
                    )
                    if consecutive_above >= required:
                        vad_logger.info("Onset detected.")
                        self.speech_event.set()
                        consecutive_above = 0
                else:
                    consecutive_above = 0
        finally:
            proc.terminate()
            proc.wait()


class MicrophoneListener:
    """Handles SR microphone capturing."""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        config = manager.get_config()
        self.recognizer.pause_threshold = config.voice.pause_threshold
        self.recognizer.non_speaking_duration = config.voice.non_speaking_duration
        self.microphone = sr.Microphone()

        self.audio_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.speech_event = threading.Event()
        self.calibrated = False

        self.vad = VADMonitor(self.recognizer, self.speech_event)
        self.running = True
        self.listen_thread: threading.Thread

    def start(self):
        self.calibrated = False
        self.running = True
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        self.vad.start()

    def stop(self):
        self.running = False
        self.vad.stop()
        self.listen_thread.join()

    def _listen_loop(self):
        logger.info("Calibrating for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.calibrated = True
            logger.info("Calibrated. Ready to listen.")

            while self.running:
                try:
                    config = manager.get_config()
                    # Re-apply config if changed
                    self.recognizer.pause_threshold = config.voice.pause_threshold
                    self.recognizer.non_speaking_duration = (
                        config.voice.non_speaking_duration
                    )

                    audio = self.recognizer.listen(
                        source,
                        timeout=1.0,  # Brief timeout so we can exit cleanly
                        phrase_time_limit=config.voice.phrase_time_limit,
                    )
                    self.audio_queue.put(("audio", audio))
                except sr.WaitTimeoutError:
                    pass
                except Exception as e:
                    logger.error(f"Microphone error: {e}")
                    time.sleep(1.0)
