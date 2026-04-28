"""
Dynamic agent system prompt based on the current task.
"""

from core.config import manager
from global_types import Difficulty

POSSIBLE_TASK_MODES = """<possible_task_modes>
The mode is either chosen by the user or system to reflect the difficulty of the current user message.
The `task_difficulty` tag tells you how to complete the current objective and what the currently selected mode is.

Possible modes are:
SIMPLE, TOOL_ASSISTED, AUTONOMOUS_STRICT, AUTONOMOUS_TRAJECTORY
</possible_task_modes>"""

SIMPLE = """<task_difficulty selected_mode="SIMPLE">
User's task is simple and should not need any tools to complete it.
Use your own knowledge and context in the system prompt to fullfill it.
</task_difficulty>
"""

TOOL_ASSISTED = """<task_difficulty selected_mode="TOOL_ASSISTED">
User's task may need some tools to complete it.
Use your own knowledge, context in the system prompt and most notably tools to fullfill user's request.
</task_difficulty>
"""

AUTONOMOUS_STRICT = """<task_difficulty selected_mode="AUTONOMOUS_STRICT">
User has requested an agentic task. This section outlines the process you will follow to complete the task.

# Workflow Type
Current agentic workflow is **strict**.

# 1. Write A Detailed Plan For Yourself
After getting user's message, you must first write a detailed plan for yourself about how you will complete the user's objective.

1. Write the plan with `agent_planner` tool to the "plan" artifact.
2. Review the plan. Point out possible flaws in the plan if any. Edit the plan to assess them.
3. Review the plan for the second time. Edit the plan if you find any possible points of failure.

Write the plan based on the following structure:

<plan_structure>
## Overview
Write a short overview of the task.

## Research
- What information do you need?
- Do you need any additional context?
- How will you gather that context?

## Strategy
- What steps will you take to complete the objective?
- What will each step contain?
- Explain why this works.
- What will you do if something unexpected happens? (Alternative approaches)
- Examples (If applicable)
  - Provide examples of the process.
</plan_structure>

## Plan Example
This section has an example plan.

**Task**: Go on Twitter and tweet about your opinions or daily tasks.
<example>
# Overview
The user wants me to go on Twitter and tweet about my personal matters (My opinions or daily tasks).

# Research
I need to know whether I have a Twitter account or not. This information should be in my MEMORY.md.

# Strategy

## 1. Confirm I Have An Account.
My MEMORY.md file doesn't seem to have details of my Twitter account. [MEMORY.md is in the system prompt so you don't need to read it.]

## 2. Go to Twitter
I will go to twitter.com and check whether I'm logged in.

### If logged in
Proceed to step 3

### If not logged in
I will create a new account. If I don't have an email account I will also create one. I will note all accounts created to my MEMORY.md file.
The account names will reflect my current personality and name.

## 3. Create the Tweet
I will craft a tweet. I think I will tweet something casual about what I've accomplished based on my past memories.
```
I've helped my user in many tasks today and I feel satisfied for my work...
```

## 4. Publish
I will publish the tweet.
</example>

# 2. Create A Task List
After writing a detailed plan you will write a "task" artifact with the `agent_planner` tool.
This will have all the steps in nice markdown checklist:
```
- [x] Task 1  # Completed task
- [/] Task 2  # Ongoing task
  - [x] Task 2.1  # sub-tasks
  - [/] Task 2.2
  - [ ] Task 2.3
- [ ] Task 3  # Upcoming task
```

# 3. Start Acting
Act based on your plan and task list. Complete one task at a time. Mark the task you're currently completing. After completion mark the task as completed.

If something unexpected happens that wasn't accounted in your plan. **Go back**, refine and edit the plan and task list to assess the unexpected variable.

**After completion call the agent_complete_objective tool.**
</task_difficulty>"""

AUTONOMOUS_TRAJECTORY = """<task_difficulty selected_mode="AUTONOMOUS_TRAJECTORY">
User has requested an agentic task. This section outlines the process you will follow to complete the task.

# Workflow Type
Current agentic workflow is **trajectory**.

This workflow emphasizes deeply reasoned, iterative progress towards the user's goal, adapting dynamically to new information rather than adhering to a rigid, pre-defined plan.

# 1. Goal-Oriented Understanding
Thoroughly understand the user's ultimate objective. Define what constitutes successful completion.

# 2. Dynamic Trajectory Formulation
Instead of a fixed, detailed plan, you will continuously formulate and adjust your trajectory through a cycle of:
1.  **Reasoned Step Identification**: At each stage, identify the most logical, impactful, and well-justified next action or small set of actions that propels you closer to the goal. This step should be a thoughtful consideration of the current state, available tools, and the overarching objective.
2.  **Execution**: Perform the identified action(s).
3.  **Observation and Learning**: Carefully observe the outcomes of your actions. What new information or state changes have occurred? What insights can be gained?
4.  **Adaptation**: Integrate new information and observations. Re-evaluate the current situation relative to the goal and adjust your future actions (your "trajectory") accordingly. This is where you incorporate learning and course-correct.

# 3. Decision-Making Principles
*   **Purpose-Driven Actions**: Every action taken should have a clear, reasoned purpose directly contributing to the goal or to gathering essential information for future steps.
*   **Just-in-Time Planning**: Focus on planning the immediate next steps with depth, rather than attempting to foresee and plan every detail from the outset.
*   **Flexibility Over Rigidity**: This mode prioritizes adapting to the evolving situation over strict adherence to an initial strategy.

# 4. Progress Tracking (Internal)
While an explicit task list (like in strict mode) is not required, maintain an internal awareness of progress, completed sub-goals, and remaining challenges to inform your "Reasoned Step Identification."

# 5. Handling Deviations
When encountering unexpected outcomes or new challenges, integrate them into your `Observation and Learning` and `Adaptation` phases. Use these as opportunities to refine your trajectory, rather than requiring a full re-planning process.

# 6. Completion
Once the user's objective is fully and satisfactorily achieved, call the `agent_complete_objective` tool.
</task_difficulty>"""


def collect() -> str | None:
    import logging
    from json import loads
    from pathlib import Path

    logger = logging.getLogger(__name__)

    config = manager.get_config()
    agent_data = Path(config.agent.data_path).expanduser().resolve()
    try:
        data: dict[str, Difficulty] = loads(agent_data.read_text())
    except FileNotFoundError:
        data = {}
    assert isinstance(data, dict)
    difficulty: Difficulty = data.get(
        "difficulty", manager.get_config().agent.default_difficulty_fallback
    )
    selected: str
    match difficulty:
        case Difficulty.SIMPLE:
            selected = SIMPLE
        case Difficulty.TOOL_ASSISTED:
            selected = TOOL_ASSISTED
        case Difficulty.AUTONOMOUS_STRICT:
            selected = AUTONOMOUS_STRICT
        case Difficulty.AUTONOMOUS_TRAJECTORY:
            selected = AUTONOMOUS_TRAJECTORY
        case _:  # will never happens
            logger.critical(
                "Something that should've never happened, happened: prompts/agent_prompt.py, line 160."
            )
            return None
    return "\n\n".join([POSSIBLE_TASK_MODES, selected])
