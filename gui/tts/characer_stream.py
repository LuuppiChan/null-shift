import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)

class CharacterStream:
    def __init__(self, iterator: AsyncIterator[str]) -> None:
        self.iterator = iterator
        self.buffer: list[str] = []

    async def next(self) -> str:
        """
        Consumes character from iterator.
        
        Raises:
            StopAsyncIteration
        """
        if self.buffer:
            return self.buffer.pop(0)
        try:
            return await anext(self.iterator)
        except StopAsyncIteration:
            logger.info("Character stream ended")
            raise StopAsyncIteration("".join(self.buffer))

    async def peek(self, n: int = 1) -> str:
        """Look ahead by n characters."""
        while len(self.buffer) < n:
            try:
                char = await anext(self.iterator)
                self.buffer.append(char)
            except StopAsyncIteration:
                break
        return "".join(self.buffer[:n])

    async def skip(self, n: int = 1) -> str:
        """Skip n characters"""
        skipped = ""
        try:
            for _ in range(n):
                skipped += await self.next() or ""
        except StopAsyncIteration:
            pass
        return skipped
