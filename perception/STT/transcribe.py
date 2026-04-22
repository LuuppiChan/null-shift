"""
STT Listener Node.

This script acts as an independent listener that connects to the Null-Shift core via ZMQ.
It captures microphone audio, runs transcription via local Whisper, and pushes inputs.
"""

import asyncio
import ctypes
import json
import logging
import math
import queue
import struct
import subprocess
import sys
import threading
import time
from ctypes.util import find_library
from typing import Any

from thefuzz import fuzz, process
import numpy as np
import speech_recognition as sr
import zmq.asyncio
from pywhispercpp.model import Model as WhisperModel

from config import manager

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("STTNode")
vad_logger = logging.getLogger("VAD")

# Suppress ALSA error messages
try:
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
        None,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
    )

    def py_error_handler(filename, line, function, err, fmt):
        pass

    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

    lib_path = find_library("asound")
    if lib_path:
        asound = ctypes.cdll.LoadLibrary(lib_path)
        asound.snd_lib_error_set_handler(c_error_handler)
except Exception:
    pass


class ZMQClient:
    """Handles async ZMQ pushing."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.PUSH)
        # We start disconnected, we'll connect on the first send/check
        self.current_address = ""
        self._lock = asyncio.Lock()

    async def _ensure_connection(self):
        new_address = manager.get_config().zmq_target_address
        if new_address != self.current_address:
            async with self._lock:
                new_address = manager.get_config().zmq_target_address  # Double check
                if new_address != self.current_address:
                    if self.current_address:
                        try:
                            self.socket.disconnect(self.current_address)
                            logger.info(f"Disconnected from {self.current_address}")
                        except zmq.ZMQError as e:
                            logger.error(f"Failed to disconnect: {e}")

                    try:
                        self.socket.connect(new_address)
                        self.current_address = new_address
                        logger.info(f"Connected ZMQ PUSH to {new_address}")
                    except zmq.ZMQError as e:
                        logger.error(f"Failed to connect to {new_address}: {e}")

    async def send_text(self, text: str):
        """Sends transcribed text to the core as an instant input message."""
        await self._ensure_connection()
        topic = "input"
        payload = {"type": "instant", "body": text}

        logger.debug(f"Sending message over ZMQ: {payload}")
        try:
            await self.socket.send_multipart(
                [topic.encode(), json.dumps(payload).encode()]
            )
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    def send_text_sync(self, text: str):
        """Thread-safe way to schedule a send from a background thread."""
        asyncio.run_coroutine_threadsafe(self.send_text(text), self.loop)


class VADMonitor:
    """Raw VAD for speech onset using low-latency parec/pacat."""

    def __init__(self, recognizer: sr.Recognizer, speech_event: threading.Event):
        self.recognizer = recognizer
        self.speech_event = speech_event
        self.running = True
        self.thread = threading.Thread(target=self._vad_loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False

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
                    self.recognizer.energy_threshold * config.tts_vad_sensitivity
                )

                if rms > threshold:
                    consecutive_above += 1
                    required = config.tts_vad_consecutive_chunks
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


class WhisperSTT:
    """Wrapper around pywhispercpp Model."""

    def __init__(self):
        config = manager.get_config()
        logger.info(
            f"Initializing GGML model {config.stt_model_path} via pywhispercpp..."
        )
        self.model = WhisperModel(
            model=config.stt_model_path,
            n_threads=config.stt_threads,
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


class MicrophoneListener:
    """Handles SR microphone capturing."""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        config = manager.get_config()
        self.recognizer.pause_threshold = config.voice_pause_threshold
        self.recognizer.non_speaking_duration = config.voice_non_speaking_duration
        self.microphone = sr.Microphone()

        self.audio_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.speech_event = threading.Event()
        self.calibrated = False

        self.vad = VADMonitor(self.recognizer, self.speech_event)
        self.running = True

    def start(self):
        t = threading.Thread(target=self._listen_loop, daemon=True)
        t.start()
        self.vad.start()

    def stop(self):
        self.running = False
        self.vad.stop()

    def _listen_loop(self):
        logger.info("Calibrating for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            self.calibrated = True
            logger.info("Calibrated. Ready to listen.")

            while self.running:
                try:
                    config = manager.get_config()
                    # Re-apply config if changed
                    self.recognizer.pause_threshold = config.voice_pause_threshold
                    self.recognizer.non_speaking_duration = (
                        config.voice_non_speaking_duration
                    )

                    audio = self.recognizer.listen(
                        source,
                        timeout=1.0,  # Brief timeout so we can exit cleanly
                        phrase_time_limit=config.voice_phrase_time_limit,
                    )
                    self.audio_queue.put(("audio", audio))
                except sr.WaitTimeoutError:
                    pass
                except Exception as e:
                    logger.error(f"Microphone error: {e}")
                    time.sleep(1.0)


class STTNode:
    """Main orchestrator for STT."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.zmq_client = ZMQClient(loop)
        self.whisper = WhisperSTT()
        self.mic_listener = MicrophoneListener()
        self.running = True
        self.transcriber_thread = threading.Thread(
            target=self._transcription_worker, daemon=True
        )

    def start(self):
        self.mic_listener.start()
        self.transcriber_thread.start()

    def stop(self):
        self.running = False
        self.mic_listener.stop()

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
                        match, score = process.extractOne(
                            text.lower(), cfg.voice_ignore_list, scorer=fuzz.ratio
                        )
                        ignored = cfg.voice_ignore_fuzz_ratio <= score
                        if ignored:
                            logger.info(
                                "Input matches '%s' from the ignore list with a score of %s",
                                match,
                                score,
                            )
                        else:
                            # Send to core
                            self.zmq_client.send_text_sync(text)

            except Exception as e:
                logger.error(f"Transcription worker error: {e}", exc_info=True)


async def main():
    logger.info("Starting up Null-Shift STT Listener Node.")
    loop = asyncio.get_running_loop()

    node = STTNode(loop)
    node.start()

    try:
        # Keep the async loop alive indefinitely while background threads do work
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Shutting down...")
    finally:
        node.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Ctrl+C received. Exiting.")
