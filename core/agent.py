"""
Contains tools and helper functions related to agent loop.
"""

import logging
from typing import Literal
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from core.backends import get_backend
from core.config import manager
from global_types import Difficulty

logger = logging.getLogger(__name__)


class DifficultyInferSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    difficulty: Literal[1, 2, 3] = Field(description="Difficulty rating of the task.")
    reason: str = Field(description="Reason for the given difficulty.")


def infer_difficulty(message: str) -> DifficultyInferSchema:
    """Infer difficulty based on the message with the power of black magic... I mean AI. **AI**"""
    config = manager.get_config()
    system = SystemMessage(config.task_infer_prompt)
    human = HumanMessage("User: " + message)
    history: list[BaseMessage] = [system, human]
    llm = get_backend()
    llm.set_model(config.task_infer_model)
    llm.set_temperature(config.task_infer_temperature)
    try:
        logger.info("Inferring task difficulty.")
        difficulty = llm.structured_output(history, DifficultyInferSchema)
    except Exception as e:
        logger.error("Error inferring difficulty: %s", e)
        difficulty = DifficultyInferSchema(
            difficulty=config.task_infer_default_fallback, reason="Fallback"
        )
    llm.reset_customs()
    return difficulty


def convert_difficulty(difficulty: DifficultyInferSchema) -> Difficulty:
    match difficulty.difficulty:
        case 1:
            return Difficulty.SIMPLE
        case 2:
            return Difficulty.TOOL_ASSISTED
        case 3:
            return manager.get_config().task_infer_agent_mode
