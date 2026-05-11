from typing import AsyncIterator, Optional, cast
from langchain_core.messages import AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from core.backends import LLMBackend
from core.config import ModelInfo
from core.registry import LLMTool
from core.config import manager


class VertexAIBackend(LLMBackend):
    def __init__(self, model: ModelInfo) -> None:
        cfg = manager.get_config()
        api_key = model.get_api_key()
        self.config = {
            "model": model.name,
            "api_key": api_key if api_key else None,
            "project": cfg.llm.vertexai_project_id,
            "location": cfg.llm.vertexai_location,
            "temperature": model.temperature,
            "thinking_level": (
                model.reasoning_effort if model.reasoning_effort else None
            ),
            # "include_thoughts": True if cfg.llm_reasoning_effort else None,
            # I'm going to gamble and hope this is an allowed key always.
            "include_thoughts": True,
            "max_tokens": model.max_tokens,
            "top_p": model.top_p,
            "vertexai": True,
        }

        self.config = {k: v for k, v in self.config.items() if v is not None}
        self.llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(**self.config)

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
