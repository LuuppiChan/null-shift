from typing import Any, AsyncIterator, Mapping, Optional, cast

# langchain discards reasoning tokens for some reason for openai backend
# So here's a monke patch
### Start ###
import langchain_openai.chat_models.base as openai_base
from langchain_core.messages import AIMessage, BaseMessage, BaseMessageChunk
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from core.backends import LLMBackend
from core.config import CoreConfig, ModelInfo, manager
from core.registry import LLMTool

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
    def __init__(self, model: ModelInfo) -> None:
        self.llm: ChatOpenAI = ChatOpenAI(
            base_url=model.url,
            model=str(model.name),
            api_key=lambda: str(model.api_key),
            temperature=model.temperature,
            top_p=model.top_p,
            presence_penalty=model.presence_penalty,
            frequency_penalty=model.frequency_penalty,
            reasoning_effort=(
                model.reasoning_effort if model.reasoning_effort else None
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
