import asyncio
import hashlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Optional, cast

from langchain_core.messages import SystemMessage

from core.config import CoreConfig, manager
from core.core_data import LocalData
from global_types import CachedFile, run_batch_ordered

logger = logging.getLogger(__name__)


async def get_context(data: LocalData | None = None) -> SystemMessage:
    """Gets current system context."""
    return await prompt_assembler.get_prompt(data)


class PromptAssembler:
    """Class handling system prompt."""

    def __init__(self) -> None:
        config = manager.get_config()
        self.path = Path(config.prompt_config.path)
        self.prompt_cache: dict[Path, CachedFile] = {}
        manager.config_updated.connect(self.config_updated)

    async def get_prompt(self, data: LocalData | None = None) -> SystemMessage:
        """Returns full system prompt."""
        await self.refresh_cache()

        timeout = manager.get_config().prompt_config.function_timeout
        logger.info("Loading prompt fragments with timeout: %s", timeout)
        files = sorted(self.prompt_cache.values(), key=lambda pf: pf.file.name)

        def mapping(f: CachedFile):
            if isinstance(f, PythonPromptFile):
                # update data
                f.data = data
            return cast(Callable[[], Any], f.get_contents)

        results: list[Optional[str]] = await run_batch_ordered(
            # Fucking type checker I have to cast Any or str to get you working.
            # This function actually might return None, but the type checker thinks
            # it has to return only a string.
            list(map(mapping, files)),
            timeout=timeout,
        )

        parts: list[str] = []
        for result in results:
            if result:
                parts.append(result)

        return SystemMessage(content="\n\n".join(parts))

    async def refresh_cache(self):
        """
        Refreshes the cache by removing removed files and adding new files to the cache.
        DOES NOT RELOAD PER-FILE CACHE!
        """
        current_files = set(await asyncio.to_thread(self.get_files))

        keys_to_remove = self.prompt_cache.keys() - current_files
        for k in keys_to_remove:
            del self.prompt_cache[k]

        keys_to_add = current_files - self.prompt_cache.keys()
        for k in keys_to_add:
            if k.suffix == ".py":
                self.prompt_cache[k] = PythonPromptFile(k)
            else:
                self.prompt_cache[k] = CachedFile(k)

    def config_updated(self, config: CoreConfig):
        """Reloads the path if the config was updated."""
        self.path = Path(config.prompt_config.path)

    def get_files(self) -> list[Path]:
        """
        Returns all file paths based on the config directory.
        Returns an empty list if the path doesn't exist.
        It's recommended to wrap this inside an async thread.
        """
        config = manager.get_config()
        if not self.path.exists():
            logger.error("Prompt path '%s' doesn't exist.", self.path)
            return []

        extensions = config.prompt_config.file_names
        pattern = "**/*" if config.prompt_config.recursive else "*"
        files: list[Path] = [
            p for p in self.path.glob(pattern) if p.suffix in extensions and p.is_file()
        ]

        return files


class PythonPromptFile(CachedFile):
    def __init__(self, file: Path) -> None:
        super().__init__(file)
        self.fn_cache: Callable[[], Optional[str]] | Callable[[Optional[LocalData]]] = (
            lambda: None
        )
        self.data: LocalData | None = None

    def get_contents(self) -> Optional[str]:
        if not self.file.exists():
            logger.warning(
                "File %s was registered, but doesn't exist anymore.", self.file.name
            )
            return None

        if not self.needs_refresh():
            try:
                result = self.call_cache()
            except Exception as e:
                logger.warning(
                    "Error while executing prompt file '%s': %s", self.file.name, e
                )
                return None
            return result

        # trick to make program imports work.
        path_hash = hashlib.md5(str(self.file.resolve()).encode()).hexdigest()[:12]
        module_name = f"core.tools._plugin_{self.file.stem}_{path_hash}"

        spec = importlib.util.spec_from_file_location(module_name, self.file)
        if spec is None or spec.loader is None:
            logger.warning("Error loading prompt module '%s'", self.file.name)
            return None

        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.warning("Error executing prompt module '%s': %s", self.file.name, e)
            return None

        config = manager.get_config()
        function: Optional[
            Callable[[], Optional[str]] | Callable[[Optional[LocalData]], Optional[str]]
        ] = getattr(module, config.prompt_config.function_name, None)
        if function is None:
            logger.warning(
                "Prompt module %s has no function named %s",
                self.file.name,
                config.prompt_config.function_name,
            )
            return None

        new_mtime = self.file.stat().st_mtime
        self.mtime = new_mtime
        self.fn_cache = function
        try:
            result = self.call_cache()
        except Exception as e:
            logger.warning("Error executing prompt module '%s': %s", self.file.name, e)
            return None

        return result

    def call_cache(self, data: LocalData | None = None) -> str | None:
        """Call the function cache correctly."""
        if data is None:
            data = self.data

        sig = inspect.signature(self.fn_cache)
        # simple check for now.
        if len(sig.parameters):
            fn = cast(Callable[[LocalData | None], str | None], self.fn_cache)
            return fn(data)
        else:
            fn = cast(Callable[[], str | None], self.fn_cache)
            return fn()


prompt_assembler = PromptAssembler()
