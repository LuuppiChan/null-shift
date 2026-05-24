import os
import subprocess
import threading
import logging
from typing import Iterator
import langdetect

from config import config
from tts.base import BaseTTSProvider

logger = logging.getLogger("PiperTTS")


class PiperTTSProvider(BaseTTSProvider):
    def __init__(self):
        """
        Initializes the PiperTTSProvider.
        Validates Piper executable and model paths for supported languages.
        """
        # We check for the executable in the venv OR system path.
        venv_bin = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv/bin/piper")
        self.piper_path = venv_bin if os.path.exists(venv_bin) else "piper"

        self.ready = True

        # Validate that models exist
        for lang, model_cfg in config.piper_models.items():
            if not os.path.exists(model_cfg.model):
                logger.warning(f"Piper model not found for '{lang}' at {model_cfg.model}")
                self.ready = False

        if self.ready:
            logger.info(f"Initialized with models: {list(config.piper_models.keys())}")

        self.active_processes = {}
        self.current_lang = None
        self._stop_event = threading.Event()
        self.speak_thread = None
        self._lock = threading.Lock()

    def speak(self, text: str, wait: bool = True, stop_existing: bool = True) -> None:
        if not self.ready:
            logger.info(f"[Assistant would say]: {text}")
            return

        if stop_existing:
            self.stop()
        
        self._stop_event.clear()

        def single_sentence_gen():
            yield text

        self.speak_thread = threading.Thread(
            target=self._speak_stream_internal, args=(single_sentence_gen(),)
        )
        self.speak_thread.daemon = True
        self.speak_thread.start()

        if wait:
            self.speak_thread.join()

    def speak_stream(
        self, sentence_generator: Iterator[str], stop_existing: bool = True
    ) -> None:
        if not self.ready:
            for text in sentence_generator:
                logger.info(f"[Assistant would say]: {text}")
            return

        if stop_existing:
            self.stop()
        
        self._stop_event.clear()

        self.speak_thread = threading.Thread(
            target=self._speak_stream_internal, args=(sentence_generator,)
        )
        self.speak_thread.daemon = True
        self.speak_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            for lang, processes in self.active_processes.items():
                if processes:
                    p_piper, p_aplay = processes
                    try:
                        if p_piper.stdin:
                            p_piper.stdin.close()
                        p_piper.terminate()
                    except:
                        pass
                    try:
                        p_aplay.terminate()
                    except:
                        pass
            self.active_processes.clear()
            self.current_lang = None

    def is_speaking(self) -> bool:
        return self.speak_thread is not None and self.speak_thread.is_alive()

    def _get_or_create_process(
        self, lang: str
    ) -> tuple[subprocess.Popen, subprocess.Popen]:
        with self._lock:
            if lang in self.active_processes and self.active_processes[lang]:
                p_piper, p_aplay = self.active_processes[lang]
                if p_piper.poll() is None and p_aplay.poll() is None:
                    return p_piper, p_aplay

            if lang not in config.piper_models:
                lang = config.default_tts_lang

            model_cfg = config.piper_models[lang]

            piper_cmd = [
                self.piper_path,
                "--model",
                model_cfg.model,
                "--config",
                model_cfg.config,
                "--output_raw",
                "--length-scale",
                str(config.voice_rate),
                "--volume",
                str(config.voice_volume),
            ]

            p_piper = subprocess.Popen(
                piper_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )

            import json

            sample_rate = "22050"
            try:
                with open(model_cfg.config, "r") as f:
                    config_data = json.load(f)
                    sample_rate = str(
                        config_data.get("audio", {}).get("sample_rate", 22050)
                    )
            except Exception:
                pass

            aplay_cmd = [
                "aplay",
                "-r",
                sample_rate,
                "-c",
                "1",
                "-f",
                "S16_LE",
                "-t",
                "raw",
                "-",
            ]
            p_aplay = subprocess.Popen(
                aplay_cmd, stdin=p_piper.stdout
            )
            if p_piper.stdout is not None:
                p_piper.stdout.close()

            self.active_processes[lang] = (p_piper, p_aplay)
            return p_piper, p_aplay

    def _speak_stream_internal(self, sentence_generator: Iterator[str]) -> None:
        import re
        cumulative_text = ""
        current_lang = config.default_tts_lang

        for sentence in sentence_generator:
            if self._stop_event.is_set():
                break

            clean_sentence = re.sub(r"[^a-zA-ZäöåÄÖÅ0-9\s]", "", sentence).strip()
            if clean_sentence:
                cumulative_text += " " + clean_sentence

                if len(cumulative_text) > config.tts_language_detection_min_chars:
                    try:
                        detected = langdetect.detect(cumulative_text)
                        if detected in config.piper_models:
                            current_lang = detected
                    except langdetect.lang_detect_exception.LangDetectException:
                        pass

            try:
                p_piper, p_aplay = self._get_or_create_process(current_lang)
                logger.info(f"Speaking sentence ({current_lang}): {repr(sentence)}")
                with self._lock:
                    if p_piper.stdin is not None and not p_piper.stdin.closed:
                        p_piper.stdin.write((sentence + "\n").encode("utf-8"))
                        p_piper.stdin.flush()
            except (BrokenPipeError, ValueError):
                with self._lock:
                    if current_lang in self.active_processes:
                        self.active_processes[current_lang] = None
                pass
            except Exception as e:
                logger.error(f"Error: {e}")

        if not self._stop_event.is_set():
            for lang, processes in list(self.active_processes.items()):
                if processes:
                    p_piper, p_aplay = processes
                    try:
                        p_piper.stdin.close()
                    except:
                        pass
                    p_aplay.wait()

            self.active_processes.clear()
