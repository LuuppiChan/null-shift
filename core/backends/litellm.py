from typing import AsyncIterator, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel

from core.backends import LLMBackend
from core.config import CoreConfig, manager
from core.registry import LLMTool


class LiteLLMBackend(LLMBackend):
    """LLM backend base class."""

    def __init__(self) -> None:
        self.llm: ChatLiteLLM
        self._update_backend(manager.get_config())
        manager.config_updated.connect(self._update_backend)

    def _update_backend(self, config: CoreConfig):
        self.llm = ChatLiteLLM(
            api_base=config.llm_base_url,
            model=config.llm_model_name,
            api_key=config.llm_api_key,
            temperature=config.llm_temperature,
            top_p=config.llm_top_p,
            streaming=True,
            custom_llm_provider=config.litellm_provider,
            model_kwargs={
                "presence_penalty": config.llm_presence_penalty,
                "frequency_penalty": config.llm_frequency_penalty,
                "reasoning_effort": (
                    config.llm_reasoning_effort if config.llm_reasoning_effort else None
                ),
            },
        )

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
        """Returns structured output as defined by the structure parameter."""
        return cast(T, self.llm.with_structured_output(structure).invoke(messages))

    def reset_customs(self):
        """Reset custom configuration for the back-end to config values."""
        self._update_backend(manager.get_config())

    def set_model(self, model: str):
        """Set custom model."""
        self.llm.model_name = model

    def set_temperature(self, temperature: float):
        """Set custom temperature."""
        self.llm.temperature = temperature

    def set_thinking(self, level: str | None):
        self.llm.model_kwargs["reasoning_effort"] = level
