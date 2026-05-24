"""
Collects memory, task list and plan.
"""

import logging
from pathlib import Path

from core.config import tool_manager
from core.core_data import LocalData
from core.helpers import PromptHelper

logger = logging.getLogger(__name__)


def collect(data: LocalData | None = None) -> str | None:
    cfg = tool_manager.get_config()
    parts: list[str] = []
    memory = PromptHelper("memory", "Your MEMORY.md file")
    file = Path(cfg.dynamic_memory_path).expanduser().resolve()
    if file.exists():
        memory.add_part(file.read_text())
        logger.info("Loaded MEMORY.md")
    else:
        logger.warning("No MEMORY.md file found.")
    parts.append(memory.compile())

    task_at_hand = PromptHelper("task_at_hand", "Extra details for the current task.")
    file = Path(cfg.dynamic_plan_path).expanduser().resolve()
    if file.exists():
        task_at_hand.add_part(
            file.read_text(), "plan", "Your plan to completing an agentic task."
        )
        logger.info("Loaded plan.md")
    else:
        logger.warning("No plan.md file found.")

    file = Path(cfg.dynamic_task_path).expanduser().resolve()
    if file.exists():
        task_at_hand.add_part(
            file.read_text(),
            "task_list",
            "Your task list to completing an agentic task.",
        )
        logger.info("Loaded task.md")
    else:
        logger.warning("No task.md file found.")

    if data:
        goal = data.agent.goal
        if goal:
            task_at_hand.add_part(goal, "current_goal", "Goal of the current task.")

        context = data.agent.context
        if context:
            task_at_hand.add_part(context, "goal_context", "Additional context for completing the goal.")

    if task_at_hand:
        parts.append(task_at_hand.compile())

    return "\n\n".join(parts)
