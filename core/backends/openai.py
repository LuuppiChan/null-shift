from typing import Any, AsyncIterator, Mapping, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage, BaseMessageChunk
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from core.backends import LLMBackend
from core.config import CoreConfig, manager
from core.registry import LLMTool

# langchain discards reasoning tokens for some reason for openai backend
# So here's a monke patch
### Start ###
import langchain_openai.chat_models.base as openai_base

_original_convert = openai_base._convert_delta_to_message_chunk


def _patched_convert_delta(
    _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    chunk = _original_convert(_dict, default_class)
    if default_class.__name__ == "AIMessageChunk" and _dict.get("reasoning_content"):
        chunk.additional_kwargs["reasoning_content"] = _dict["reasoning_content"]
    return chunk


# Apply monkey patch globally for this instance
openai_base._convert_delta_to_message_chunk = _patched_convert_delta
### End ###


class OpenAIBackend(LLMBackend):
    def __init__(self) -> None:
        self.llm: ChatOpenAI
        self._update_backend(manager.get_config())
        manager.config_updated.connect(self._update_backend)

    def _update_backend(self, config: CoreConfig):
        """Gets the back-end llm based on the config, ready to be used."""
        self.llm = ChatOpenAI(
            base_url=config.llm_base_url,
            model=config.llm_model_name,
            api_key=lambda: config.llm_api_key,
            temperature=config.llm_temperature,
            top_p=config.llm_top_p,
            presence_penalty=config.llm_presence_penalty,
            frequency_penalty=config.llm_frequency_penalty,
            reasoning_effort=(
                config.llm_reasoning_effort if config.llm_reasoning_effort else None
            ),
        )

    def stream(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[LLMTool]],
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
        self.llm.temperature = temperature

    def set_model(self, model: str):
        self.llm.model_name = model
