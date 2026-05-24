"""
Tools related to agent system.
"""

from pathlib import Path
from langchain_core.tools import tool

from core.vector import vector
from global_types import is_autonomous


class NotAutonomous(Exception): ...


if not is_autonomous(vector.data.agent.difficulty):
    raise NotAutonomous("Current task doesn't need agent tools.")


@tool
def agent_complete_objective() -> str:
    """
    Complete your current objective.

    If the system prompt tells that you are in agent mode (Autonomous Strict or Autonomous Trajectory) you must call this after completing the current objective.
    If the current mode is Simple or Tool Assisted you don't need to call this.
    """
    from core.vector import vector
    data = vector.data.agent
    data.completed = True
    data.goal = None
    data.context = None

    base = Path("/home/luuppi/vm_drive/null-shift")
    (base / "plan.md").unlink(True)
    (base / "task.md").unlink(True)
    return "Objective marked as completed. Provide a final answer to the user. Note that the user cannot clearly see the messages you sent while completing objective."


...
