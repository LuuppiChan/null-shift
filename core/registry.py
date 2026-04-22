import asyncio
from datetime import datetime, timedelta
import logging
import hashlib
import importlib.util
from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool

from core.config import manager
from global_types import CachedFile, run_batch_ordered

logger = logging.getLogger(__name__)

type LLMTool = BaseTool  # Callable[..., Any]  # Callable[..., Awaitable[Any]]


async def get_tools() -> dict[str, LLMTool]:
    """Returns a tool map."""
    return await tool_registry.get_tools()


class ToolRegistry:
    def __init__(self) -> None:
        self.cache: dict[Path, ToolFile] = {}
        self.tool_cache: dict[str, LLMTool] = {}
        self.refreshed = datetime.now()

    async def refresh_cache(self):
        """
        Refreshes the cache by removing removed files and adding new files to the cache.
        DOES NOT RELOAD PER-FILE CACHE!
        """
        current_files = set(await asyncio.to_thread(self.get_files))

        keys_to_remove = self.cache.keys() - current_files
        for k in keys_to_remove:
            del self.cache[k]

        keys_to_add = current_files - self.cache.keys()
        for k in keys_to_add:
            self.cache[k] = ToolFile(k)

    def get_files(self) -> list[Path]:
        """Returns all the python files (.py) in the current tool directory based on the config."""
        config = manager.get_config()
        path = Path(config.core_tools_path)
        pattern = "**/*.py" if config.core_tools_recursive else "*.py"
        return list(path.glob(pattern))

    async def get_tools(self) -> dict[str, LLMTool]:
        """Gets all the tools from tools folder."""

        needs_refresh = self.refreshed + timedelta(
            manager.get_config().core_tools_min_refresh_delay
        )
        if datetime.now() > needs_refresh:
            logger.info(
                "Tools are fresh enough, returning cache without checking file changes."
            )
            return self.tool_cache
        else:
            logger.info(
                "Tools have exeeded minimum refresh time, they will be refreshed."
            )

        await self.refresh_cache()

        timeout = manager.get_config().core_tools_module_timeout
        logger.info("Loading tools with timeout: %s", timeout)
        files = sorted(self.cache.values(), key=lambda tf: tf.file.name)
        results: list[list[LLMTool] | None] = await run_batch_ordered(
            list(map(lambda f: f.get_functions, files)), timeout=timeout
        )
        tools: dict[str, LLMTool] = {}

        for i, result in enumerate(results):
            if result is None:
                logger.warning(
                    "Tool file %s wasn't loaded in time.", files[i].file.name
                )
                continue

            for tool in result:
                tools[tool.name] = tool

        self.tool_cache = tools
        self.refreshed = datetime.now()
        return tools


class ToolFile(CachedFile):
    def __init__(self, file: Path) -> None:
        super().__init__(file)
        self.tools_cache: Optional[list[LLMTool]] = None

    def get_functions(self) -> list[LLMTool]:
        """Gets all tools from the file or cached file."""
        if not self.file.exists():
            return []

        if not self.needs_refresh() and self.tools_cache is not None:
            return self.tools_cache

        # trick to make program imports work.
        path_hash = hashlib.md5(str(self.file.resolve()).encode()).hexdigest()[:12]
        module_name = f"core.tools._plugin_{self.file.stem}_{path_hash}"

        spec = importlib.util.spec_from_file_location(module_name, self.file)
        if spec is None or spec.loader is None:
            logger.warning("Error loading file '%s'", self.file.name)
            return []

        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.warning("Error executing file '%s': %s", self.file.name, e)
            return []

        new_mtime = self.file.stat().st_mtime
        self.mtime = new_mtime

        tools: list[LLMTool] = []
        for attr_name in dir(module):
            func = getattr(module, attr_name)
            if isinstance(func, BaseTool):
                logger.info(
                    "Loaded tool: %s from module %s",
                    func.name,
                    module.__name__,
                )
                tools.append(func)
            elif not attr_name.startswith("_"):
                logger.debug("%r from %s is not a tool.", func, module.__name__)

        self.tools_cache = tools
        return self.tools_cache


tool_registry = ToolRegistry()
