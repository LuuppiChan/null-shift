import logging
import re
from typing import AsyncGenerator, AsyncIterable

from gui.tts.characer_stream import CharacterStream
from gui.config import manager

logger = logging.getLogger(__name__)


class SentenceStream:
    def __init__(self, chunks: AsyncIterable[str]) -> None:
        """
        chunks is the stream of speakable text chunks from back-end.
        It can be a token, sentence or even a full response.
        """
        self.iterable = chunks

    async def sentences(self) -> AsyncGenerator[str, None]:
        """Generator that generates speakable sentences."""
        buf = ""
        chars = CharacterStream(self.characters())
        try:
            while True:
                c = await chars.peek()
                match c:
                    case "`":
                        if await chars.peek(3) == "```":
                            await chars.skip(3)
                            while await chars.peek(3) != "```":
                                await chars.skip()
                            await chars.skip(3)
                            buf += "Code block."
                        else:
                            await chars.skip()
                            while await chars.peek() != "`":
                                buf += await chars.next()
                            await chars.skip()
                    case "$":
                        if await chars.peek(2) == "$$":
                            await chars.skip(2)
                            while await chars.peek(2) != "$$":
                                await chars.skip()
                            await chars.skip(2)
                        else:
                            await chars.skip()
                            while await chars.peek() != "$":
                                await chars.skip()
                            await chars.skip()
                        buf += "LaTex block."
                    case "." if await chars.peek(2) not in [". ", ".\n", "."]:
                        # append the sentence end character to buffer since it's not actually sentence end.
                        buf += await chars.skip(2)
                    case "." | "?" | "!":
                        # append the sentence end.
                        sentence = buf.strip() + await chars.next()
                        logger.info("Sending sentence: %s", sentence)

                        yield self.apply_filters(sentence).strip()
                        buf = ""
                    case _:
                        buf += await chars.next()
        except StopAsyncIteration as e:
            last = (buf + e.args[0]).strip()
            logger.info("Empty iterator, last buffer: %s", last)
            yield self.apply_filters(buf + e.args[0]).strip()

    def apply_filters(self, text: str) -> str:
        cfg = manager.get_config()
        for pat, repl in cfg.speak.replace_filter.items():
            text = re.sub(pat, repl, text)
        return text

    async def characters(self) -> AsyncGenerator[str, None]:
        """Generator of characters from the stream."""
        async for chunk in self.iterable:
            for char in chunk:
                yield char
