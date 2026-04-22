"""
Tools related to agent system.
"""

import json
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import tool

from core.config import manager, tool_manager
from global_types import is_autonomous

type Task = dict[Literal["plan", "steps"], str | list[tuple[str, bool]]]
type Data = dict[str, bool | str | Task | None]


class NotAutonomous(Exception): ...


def _data() -> Data:
    """Get agent data."""
    config = manager.get_config()
    path = Path(config.task_agent_data_path).expanduser().resolve()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        path.write_text("{}")
    return json.loads(path.read_text())


def _save(data: dict[str, Any]):
    """Save agent data"""
    config = manager.get_config()
    path = Path(config.task_agent_data_path).expanduser().resolve()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    path.write_text(json.dumps(data))


# Maybe add later
# if not is_autonomous(_data().get("difficulty")):
#     raise NotAutonomous("Current task doesn't need agent tools.")


@tool
def agent_complete_objective() -> str:
    """
    Complete your current objective.

    If the system prompt tells that you are in agent mode (Autonomous Strict or Autonomous Trajectory) you must call this after completing the current objective.
    If the current mode is Simple or Tool Assisted you don't need to call this.
    """
    data = _data()
    data["completed"] = True
    base = Path("/home/luuppi/vm_drive/null-shift")
    (base / "plan.md").unlink(True)
    (base / "task.md").unlink(True)
    data["goal"] = None
    _save(data)
    return "Objective marked as completed. Provide a final answer to the user. Note that the user cannot clearly see the messages you sent while completing objective."


...
