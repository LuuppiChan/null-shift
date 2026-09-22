from collections.abc import AsyncIterator
from typing import cast

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel, SecretStr

from core.backends import LLMBackend
from core.config import ModelInfo
from core.registry import LLMTool


class OpenRouterBackend(LLMBackend):
    def __init__(self, model: ModelInfo) -> None:
        self.llm = ChatOpenRouter(
            model=model.name,
            base_url=model.url,
            api_key=SecretStr(model.get_api_key() or ""),
            reasoning=(
                model.reasoning
                or (
                    {"effort": model.reasoning_effort}
                    if model.reasoning_effort
                    else None
                )
            ),
            temperature=model.temperature,
            top_p=model.top_p,
            presence_penalty=model.presence_penalty,
            frequency_penalty=model.frequency_penalty,
            max_tokens=model.max_tokens,
            default_headers=model.default_headers,
            model_kwargs=model.extra_body or {},
            openrouter_provider=model.openrouter_provider,
        )

    def stream(
        self, messages: list[BaseMessage], tools: list[LLMTool] | None
    ) -> AsyncIterator[AIMessage]:
        if tools:
            return self.llm.bind_tools(tools).astream(messages)
        else:
            return self.llm.astream(messages)

    def structured_output[T: BaseModel](
        self, messages: list[BaseMessage], structure: type[T]
    ) -> T:
        return cast(T, self.llm.with_structured_output(structure).invoke(messages))
