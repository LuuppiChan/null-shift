import asyncio
import sys
import zmq
import zmq.asyncio
import queue
import threading
import json
import logging
import re
from typing import Iterator


from tts.engine import TTSEngine
from tts.cleaner import StreamCleaner
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TTS_Listener")

class ZmqTtsListener:
    def __init__(self):
        self.engine = TTSEngine()
        self.ctx = zmq.asyncio.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(config.zmq_output_bind)
        
        # Subscribe to all relevant topics
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "out.stream")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "event.abort")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "event.started")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "event.finished")
        
        # Punctuation to split sentences (period, question mark, exclamation mark followed by space or newline)
        self.sentence_endings = re.compile(r'([.?!])(?=\s|$)')
        self.buffer = ""
        self.cleaner = StreamCleaner(config.cleaning_blocks, config.cleaning_replacements)
        
        self.msg_queue = queue.Queue()
        self._generator_sentinel = object()
        self._current_generator_active = False

    def _sentence_generator(self) -> Iterator[str]:
        while True:
            item = self.msg_queue.get()
            if item is self._generator_sentinel:
                break
            yield item
    
    def _start_new_stream(self):
        if not self._current_generator_active:
            self._current_generator_active = True
            logger.info("Starting new TTS stream")
            self.engine.speak_stream(self._sentence_generator(), stop_existing=True)

    def _flush_buffer(self):
        sentence = self.buffer.strip()
        if sentence:
            self.msg_queue.put(sentence)
        self.buffer = ""

    def _stop_stream(self):
        if self._current_generator_active:
            logger.info("Stopping TTS stream immediately")
        self.buffer = ""
        self.cleaner.stop()
        self.engine.stop()
        if self._current_generator_active:
            self.msg_queue.put(self._generator_sentinel)
            self._current_generator_active = False
            # Clear queue
            while not self.msg_queue.empty():
                try: 
                    self.msg_queue.get_nowait()
                except queue.Empty:
                    break

    async def run(self):
        logger.info(f"Connecting to ZMQ output stream at {config.zmq_output_bind}")
        while True:
            try:
                parts = await self.sub.recv_multipart()
                logger.debug(f"Received ZMQ parts: {len(parts)}")
                if len(parts) < 2:
                    continue
                
                topic = parts[0].decode("utf-8")
                logger.info(f"Topic: {topic}")
                
                try:
                    payload = json.loads(parts[1].decode("utf-8"))
                except Exception as e:
                    logger.error(f"Failed to parse payload: {e}")
                    continue

                if topic == "event.started":
                    self._stop_stream()
                    self._start_new_stream()
                    
                elif topic == "event.abort":
                    self._stop_stream()
                    
                elif topic == "event.finished":
                    self.buffer += self.cleaner.flush()
                    self._flush_buffer()
                    if self._current_generator_active:
                        self.msg_queue.put(self._generator_sentinel)
                        self._current_generator_active = False
                        
                elif topic == "out.stream":
                    if not self._current_generator_active:
                        self._start_new_stream()
                    
                    text_part = payload.get("text", "")
                    if text_part:
                        cleaned_part = self.cleaner.process_chunk(text_part)
                        if cleaned_part:
                            self.buffer += cleaned_part
                            
                            while True:
                                match = self.sentence_endings.search(self.buffer)
                                if match:
                                    end_idx = match.end()
                                    sentence = self.buffer[:end_idx].strip()
                                    if sentence:
                                        self.msg_queue.put(sentence)
                                    self.buffer = self.buffer[end_idx:]
                                else:
                                    if '\n' in self.buffer:
                                        parts_split = self.buffer.split('\n', 1)
                                        if parts_split[0].strip():
                                            self.msg_queue.put(parts_split[0].strip())
                                        self.buffer = parts_split[1]
                                    else:
                                        break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in ZMQ loop: {e}")

        # flush on shutdown
        self._stop_stream()

if __name__ == "__main__":
    listener = ZmqTtsListener()
    try:
        asyncio.run(listener.run())
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)
