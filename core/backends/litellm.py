from typing import AsyncIterator, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel

from core.backends import LLMBackend
from core.config import ModelInfo, manager
from core.registry import LLMTool


class LiteLLMBackend(LLMBackend):
    """LLM backend base class."""

    def __init__(self, model: ModelInfo) -> None:
        self.llm: ChatLiteLLM = ChatLiteLLM(
            api_base=model.url,
            model=str(model.name),
            api_key=model.get_api_key(),
            temperature=model.temperature,
            top_p=model.top_p,
            streaming=True,
            custom_llm_provider=manager.get_config().llm.litellm_provider,
            model_kwargs={
                "presence_penalty": model.presence_penalty,
                "frequency_penalty": model.frequency_penalty,
                "reasoning_effort": (
                    model.reasoning_effort if model.reasoning_effort else None
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
