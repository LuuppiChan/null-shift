from typing import NoReturn
import asyncio
import os
from pathlib import Path


def run() -> NoReturn:
    """Run main module."""
    # Fix directory issues by normalizing it.
    os.chdir(Path(__file__).parent)

    import core.main
    try:
        asyncio.run(core.main.main())
    except KeyboardInterrupt:
        pass
    finally:
        core.main.ctx.term()
        exit()
