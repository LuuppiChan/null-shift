import os
import subprocess
import threading
import logging
from typing import Iterator
import numpy as np
import torch
import ChatTTS

from config import config, ChatTTSConfig
from tts.base import BaseTTSProvider

logger = logging.getLogger("ChatTTSProvider")


class ChatTTSProvider(BaseTTSProvider):
    def __init__(self) -> None:
        """
        Initializes the ChatTTSProvider.
        Loads ChatTTS models and sets up the speaker configuration.
        """
        super().__init__()
        self.ready = False
        self.speak_thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()
        self._lock: threading.Lock = threading.Lock()
        self.active_processes: list[subprocess.Popen] = []

        try:
            chattts_cfg = config.chattts if config.chattts else ChatTTSConfig()
            self.compile: bool = chattts_cfg.compile
            self.temperature: float = chattts_cfg.temperature
            self.top_K: int = chattts_cfg.top_K
            self.top_P: float = chattts_cfg.top_P
            self.refine_text: bool = chattts_cfg.refine_text

            self.chat = ChatTTS.Chat()

            device_str = chattts_cfg.device
            if not device_str:
                device_str = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading ChatTTS on device: {device_str} (compile={self.compile})...")
            device_obj = torch.device(device_str)
            self.chat.load(compile=self.compile, device=device_obj)

            if not self.chat.has_loaded():
                logger.error("ChatTTS failed to load models.")
                return

            # Speaker configuration
            self.spk_emb: torch.Tensor | None = None
            if chattts_cfg.speaker_string:
                logger.info("Loading configured speaker string...")
                spk_emb_np = self.chat.speaker._decode(chattts_cfg.speaker_string)
                self.spk_emb = torch.from_numpy(spk_emb_np).to(device=self.chat.device)
            else:
                seed = chattts_cfg.speaker_seed if chattts_cfg.speaker_seed is not None else 42
                logger.info(f"Generating random speaker with seed {seed}...")
                torch.manual_seed(seed)
                spk_str = self.chat.sample_random_speaker()
                spk_emb_np = self.chat.speaker._decode(spk_str)
                self.spk_emb = torch.from_numpy(spk_emb_np).to(device=self.chat.device)
                logger.info(f"Speaker generated successfully. Seed: {seed}. Representing string: {spk_str}")

            self.ready = True
            logger.info("ChatTTSProvider initialized and ready.")

        except Exception as e:
            logger.exception(f"Failed to initialize ChatTTSProvider: {e}")

    def speak(self, text: str, wait: bool = True, stop_existing: bool = True) -> None:
        if not self.ready:
            logger.info(f"[ChatTTS Provider not ready - Assistant would say]: {text}")
            return

        if stop_existing:
            self.stop()

        self._stop_event.clear()

        def single_sentence_gen() -> Iterator[str]:
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
            logger.info("[ChatTTS Provider not ready - consuming stream without speaking]")
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
        
        try:
            if hasattr(self, 'chat') and self.chat is not None:
                self.chat.interrupt()
        except Exception as e:
            logger.error(f"Error interrupting ChatTTS: {e}")

        with self._lock:
            for p in self.active_processes:
                try:
                    p.terminate()
                except Exception:
                    pass
            self.active_processes.clear()

    def is_speaking(self) -> bool:
        return self.speak_thread is not None and self.speak_thread.is_alive()

    def _speak_stream_internal(self, sentence_generator: Iterator[str]) -> None:
        for sentence in sentence_generator:
            if self._stop_event.is_set():
                break

            sentence = sentence.strip()
            if not sentence:
                continue

            try:
                params_infer_code = self.chat.InferCodeParams()
                params_infer_code.spk_emb = self.spk_emb
                params_infer_code.temperature = self.temperature
                params_infer_code.top_K = self.top_K
                params_infer_code.top_P = self.top_P
                params_infer_code.ensure_non_empty = False

                logger.info(f"Synthesizing sentence: {repr(sentence)}")
                res = self.chat.infer(
                    sentence,
                    skip_refine_text=not self.refine_text,
                    split_text=False,
                    params_infer_code=params_infer_code,
                )

                if not res or len(res) == 0:
                    continue

                wav = res[0]
                # Ensure 1D numpy array
                if hasattr(wav, "ndim") and wav.ndim > 1:
                    wav = wav.flatten()

                # Convert float32 array to 16-bit PCM bytes
                pcm = np.clip(wav, -1.0, 1.0)
                pcm = (pcm * 32767.0).astype(np.int16)

                # Add leading (0.1s) and trailing (0.5s) digital silence padding
                # to prevent hardware wake-up clipping and early audio cut-offs.
                leading_silence = np.zeros(2400, dtype=np.int16)
                trailing_silence = np.zeros(12000, dtype=np.int16)
                pcm = np.concatenate([leading_silence, pcm, trailing_silence])

                wav_bytes = pcm.tobytes()

                if self._stop_event.is_set():
                    break

                aplay_cmd = [
                    "aplay",
                    "-r",
                    "24000",  # ChatTTS default sample rate
                    "-c",
                    "1",
                    "-f",
                    "S16_LE",
                    "-t",
                    "raw",
                    "-",
                ]

                p_aplay = subprocess.Popen(
                    aplay_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

                with self._lock:
                    self.active_processes.append(p_aplay)

                logger.info("Playing audio...")
                _, stderr = p_aplay.communicate(input=wav_bytes)

                if p_aplay.returncode != 0 and stderr:
                    logger.error(
                        f"aplay error (exit code {p_aplay.returncode}): {stderr.decode()}"
                    )

                with self._lock:
                    if p_aplay in self.active_processes:
                        self.active_processes.remove(p_aplay)

            except Exception as e:
                logger.error(f"Error in ChatTTS speak generator loop: {e}", exc_info=True)
