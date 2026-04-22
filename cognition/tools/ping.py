from time import sleep

from registry import tool


@tool
async def sleep_for(seconds: int) -> str:
    """Sleep for the specified amount of seconds."""
    sleep(seconds)
    return "Done sleeping."
