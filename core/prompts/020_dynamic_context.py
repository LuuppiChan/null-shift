"""
Definitions and Specifications: Provide clear, explicit, specific, and complete descriptions of any definitions and/or specifications unique to the problem.
"""

from core.helpers import PromptHelper
from core.core_data import data


def collect() -> str | None:
    prompt = PromptHelper("dynamic_context")
    prompt.add_part(
        "This section contains dynamic data about the surrounding environment meaning it's always up-to-date.\nThis information may or may not be relevant to your task, it is up for you to decide.",
        "description",
    )
    time = PromptHelper("time_context", "Current time and data about passed events")
    time.add_part(data.datetime(), "current_date_time")
    # time.add_part(data.recent_events(10), "recent_events")
    prompt.add_part(time)
    prompt.add_part(data.home_path(), "user_home_path")
    prompt.add_part(
        data.scratchpad(),
        "assistant_scratchpad_path",
        "This is your dedicated workspace. You can freely read, write, and modify files here. Use this directory to draft code, store intermediate data, write down step-by-step plans, or keep track of your thoughts during complex tasks. Consider it your working memory.",
    )

    return prompt.compile()
