from typing import AsyncIterator, Optional, cast
from langchain_core.messages import AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from core.backends import LLMBackend
from core.config import CoreConfig
from core.registry import LLMTool
from core.config import manager


class VertexAIBackend(LLMBackend):
    def __init__(self) -> None:
        self.llm: ChatGoogleGenerativeAI
        self._update_backend(manager.get_config())
        manager.config_updated.connect(self._update_backend)

    def _update_backend(self, cfg: CoreConfig):
        self.config = {
            "model": cfg.llm_model_name,
            "api_key": cfg.llm_api_key if cfg.llm_api_key else None,
            "project": cfg.vertexai_project_id,
            "location": cfg.vertexai_location,
            "temperature": cfg.llm_temperature,
            "thinking_level": (
                cfg.llm_reasoning_effort if cfg.llm_reasoning_effort else None
            ),
            # "include_thoughts": True if cfg.llm_reasoning_effort else None,
            # I'm going to gamble and hope this is an allowed key always.
            "include_thoughts": True,
            "max_tokens": cfg.llm_max_tokens,
            "top_p": cfg.llm_top_p,
            "vertexai": True,
        }
        self.config = {k: v for k, v in self.config.items() if v is not None}
        self._update_llm()

    def _update_llm(self):
        self.llm = ChatGoogleGenerativeAI(**self.config)

    def stream(
        self, messages: list[BaseMessage], tools: Optional[list[LLMTool]]
    ) -> AsyncIterator[AIMessage]:
        if tools:
            return self.llm.bind_tools(tools).astream(messages)
        else:
            return self.llm.astream(messages)

    def structured_output[T: BaseModel](
        self, messages: list[BaseMessage], structure: type[T]
    ) -> T:
        return cast(T, self.llm.with_structured_output(structure).invoke(messages))

    def reset_customs(self):
        self._update_backend(manager.get_config())

    def set_temperature(self, temperature: float):
        self.config["temperature"] = temperature
        self._update_llm()

    def set_model(self, model: str):
        self.config["model"] = model
        self._update_llm()

    def set_thinking(self, level: str | None):
        self.config["thinking_level"] = level
        self._update_llm()
