"""
Tools related to agent system.
"""

from pathlib import Path
from langchain_core.tools import tool

from core.vector import vector
from global_types import is_autonomous
from core.config import tool_manager


class NotAutonomous(Exception): ...


if not is_autonomous(vector.data.agent.difficulty):
    raise NotAutonomous("Current task doesn't need agent tools.")


@tool(
    description="""Complete your current objective.

If the system prompt tells that you are in agent mode (Autonomous Strict or Autonomous Trajectory) you must call this after completing the current objective.
If the current mode is Simple or Tool Assisted you don't need to call this.

Args:
    remove_artifacts: Delete task.md and plan.md"""
)
def agent_complete_objective(remove_artifacts: bool = False) -> str:
    from core.vector import vector

    data = vector.data.agent
    data.completed = True
    data.goal = None
    data.context = None

    if remove_artifacts:
        base = Path("/home/luuppi/vm_drive/null-shift")
        (base / "plan.md").unlink(True)
        (base / "task.md").unlink(True)
    return "Objective marked as completed. Provide a final answer to the user. Note that the user cannot clearly see the messages you sent while completing objective."


@tool(
    description="""Edit artifacts conveniently.
Will just overwrite the given artifact if a text is given."""
)
def agent_planner(plan: str | None = None, task: str | None = None) -> str:
    cfg = tool_manager.get_config()
    feedback = ["The following artifact(s) have been updated: "]
    if plan is not None:
        plan_path = Path(cfg.dynamic_plan_path)
        plan_path.write_text(plan)
        feedback.append("plan")
    if task is not None:
        task_path = Path(cfg.dynamic_task_path)
        task_path.write_text(task)
        feedback.append("task")
    return " ".join(feedback)
