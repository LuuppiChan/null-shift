"""
Contains all configuration and fetches the configuration dynamically based on mtime from disk.

Additionally contains runtime state.
"""

import asyncio
import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from global_tools import ConfigManager
from global_types import Difficulty

logger = logging.getLogger(__name__)


class ToolConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Niri
    niri_focus_window_permission: bool = False
    niri_focus_window_prompt: str = (
        "Are you sure you want to focus window with an id of {id}?"
    )
    niri_set_monitor_permission: bool = False
    niri_set_monitor_prompt: str = (
        "Are you sure you want to turn {state} your monitors?"
    )

    # Browser tool settings
    browser_zmq_address: str = "tcp://127.0.0.1:5560"
    browser_website_blacklist: list[str] = Field(default_factory=list)
    browser_website_whitelist: list[str] = Field(default_factory=list)
    browser_require_confirmation_on_untrusted_navigation: bool = False
    browser_dom_char_limit: int = 4000
    browser_confirm_tab_close: bool = False
    browser_confirm_close_tab_prompt: str = "Are you sure you want to close tab {}?"

    file_path_blacklist: list[str] = Field(default_factory=list)
    file_path_whitelist: list[str] = Field(default_factory=list)
    file_prompt_read: bool = False
    file_prompt_read_prompt: str = "Are you sure you want to read {}"
    file_prompt_edit: bool = True
    file_prompt_edit_prompt: str = "Are you sure you want to write {}"
    file_query_timeout: float = 30.0
    file_absurd_size_limit: int = 10000

    linux_read_command_timeout: float = 30.0
    linux_read_allowed_commands: list[str] = Field(
        default_factory=lambda: [
            "grep",
            "cat",
            "head",
            "tail",
            "tree",
            "ls",
            "wc",
            "file",
            "stat",
            "strings",
            "diff",
            "cmp",
            "cut",
            "paste",
            "md5sum",
            "sha256sum",
            "du",
            "hexdump",
            "od",
            "readelf",
            "nm",
            "ldd",
            "uniq",
            "objdump",
        ]
    )

    # These should be with the brain path
    # I will move these later.
    # No, this actually lives completely outside the core.
    dynamic_memory_path: str = "~/vm_drive/null-shift/MEMORY.md"
    dynamic_plan_path: str = "~/vm_drive/null-shift/plan.md"
    dynamic_task_path: str = "~/vm_drive/null-shift/task.md"


class CoreConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    llm_provider: Literal["openai", "vertexai", "litellm"] | str = "openai"
    llm_model_name: str = "gemini-2.5-flash-lite-preview"
    llm_temperature: Optional[float] = Field(default=0.6, ge=0)
    llm_base_url: str = "http://localhost:4000/v1"
    llm_api_key: str = ""
    llm_top_p: Optional[float] = Field(default=None, ge=0, le=1)
    llm_presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    llm_frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    llm_reasoning_effort: Optional[str] = None
    llm_max_tokens: Optional[int] = None

    vertexai_project_id: str = ""
    vertexai_location: str = "global"

    litellm_provider: Optional[str] = None

    task_infer_model: str = "gemini-2.5-flash-lite"
    task_infer_temperature: Optional[float] = 0.4
    task_infer_prompt: str = ""
    task_infer_agent_mode: Literal[
        Difficulty.AUTONOMOUS_STRICT, Difficulty.AUTONOMOUS_TRAJECTORY
    ] = Difficulty.AUTONOMOUS_TRAJECTORY
    task_infer_default_fallback: Literal[1, 2, 3] = 2
    task_default_difficulty_fallback: Difficulty = Difficulty.AUTONOMOUS_TRAJECTORY

    task_agent_ignore_iterations: bool = False
    task_agent_data_path: str = "~/.null-shift/brain/agent.json"
    task_agent_continue_prompt: str = "[AGENT SYSTEM]: You haven't indicated completion intent. Continue the task or call the agent_complete_objective tool. Remeber to stay in the tool call loop by calling tools every response!"

    core_default_batch_task_name: str = "Info"
    core_max_llm_iterations: int = 1000
    core_max_llm_retries: int = 3
    core_retry_delay: float = 5.0
    core_history_length: int = 25
    core_history_path: str = "~/.null-shift/brain/history.json"
    core_history_compression: bool = True
    core_history_compression_threshold: int = 40
    core_history_compression_target_length: int = 20
    core_history_compression_model: str = "gemini-2.5-flash-lite"
    # just default, put one to the .toml
    core_history_compression_prompt: str = (
        "Summarize key details from the user's message."
    )

    core_prompt_path: str = "prompts/"
    core_prompt_recursive: bool = False
    core_prompt_file_names: list[str] = [".py", ".md", ".xml"]
    core_prompt_function_name: str = "collect"
    core_prompt_function_timeout: float = 0.5

    core_tools_path: str = "tools/"
    core_tools_recursive: bool = False
    core_tools_module_timeout: float = 1.0
    core_tools_min_refresh_delay: float = 5.0

    core_permission_yes_words: list[str] = Field(
        default_factory=lambda: [
            "yes",
            "yeah",
            "yep",
            "sure",
            "ok",
            "okay",
            "do it",
            "go ahead",
            "proceed",
        ]
    )
    core_permission_no_words: list[str] = Field(
        default_factory=lambda: [
            "no",
            "nope",
            "nah",
            "cancel",
            "stop",
            "don't",
            "do not",
            "dont",
            "abort",
        ]
    )
    core_permission_timeout: float = 30.0
    core_response_default_topic: str = "response"

    zmq_input_bind: str = "tcp://*:5555"
    zmq_output_bind: str = "tcp://*:5556"

    log_silenced_libraries: list[str] = Field(default_factory=list)
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file_path: str = "core.log"


class RuntimeState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    shutdown_event: asyncio.Event = Field(default_factory=asyncio.Event)
    is_running: bool = False


manager = ConfigManager(Path("core_config.toml"), CoreConfig())
state = RuntimeState()
tool_manager = ConfigManager(Path("tool_config.toml"), ToolConfig())
