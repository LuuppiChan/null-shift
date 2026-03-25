import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass
class AgentConfig:
    """Agent Wrapper configuration loaded dynamically from TOML."""
    # -- ZMQ -----------------------------------------------------------------
    zmq_core_input: str = "tcp://127.0.0.1:5555"
    zmq_core_output: str = "tcp://127.0.0.1:5556"
    zmq_signals_bind: str = "tcp://*:5557"
    zmq_ui_bind: str = "tcp://*:5558"

    # -- Prompts -------------------------------------------------------------
    prompts_step_done: str = "Please proceed to the next step. If you have hallucinated or errors occured, you can edit the plan."
    prompts_plan_updated: str = "Plan update logic acknowledged. Please proceed."
    prompts_phase_done: str = "Phase acknowledged. Please proceed with the execution."
    prompts_strict_template: str = "You are in Strict Plan mode. First create an implementation plan, then execute it step by step.\nTask: {body}"
    prompts_trajectory_template: str = "You are in Target Trajectory mode. Navigate to the target using tools and plan on a small scale mid-task. Do not create a large initial plan.\nTask: {body}"
    prompts_wake_up: str = "You exited the tool loop without explicitly completing the task or taking a step. Please use a tool to proceed or mark the task as done."
    prompts_agentic_loop_title: str = "Agentic Loop"
    
    # -- Behaviour -----------------------------------------------------------
    log_level: str = "INFO"

    _extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

_KNOWN_FIELDS: frozenset[str] = frozenset(f.name for f in fields(AgentConfig))

def load_config(path: str | Path = "wrapper_config.toml") -> AgentConfig:
    cfg = AgentConfig()
    toml_path = Path(path)

    if not toml_path.exists():
        return cfg

    with toml_path.open("rb") as fh:
        raw: dict[str, Any] = tomllib.load(fh)

    flat: dict[str, Any] = {}

    _SECTION_PREFIXES: dict[str, str] = {
        "zmq": "zmq_",
        "prompts": "prompts_",
        "behaviour": "",
    }

    for section, prefix in _SECTION_PREFIXES.items():
        if section in raw:
            for k, v in raw.pop(section).items():
                flat[f"{prefix}{k}"] = v

    flat.update(raw)
    extra: dict[str, Any] = {}
    
    for k, v in flat.items():
        if k in _KNOWN_FIELDS and k != "_extra":
            setattr(cfg, k, v)
        else:
            extra[k] = v

    cfg._extra = extra
    return cfg

