"""
Contains all configuration and fetches the configuration dynamically based on mtime from disk.

Additionally contains runtime state.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from global_tools import ConfigManager
from global_types import Difficulty

logger = logging.getLogger(__name__)

type ModelIdentifier = str | ModelInfo | None
API_KEY_GLOBALS: dict[str, Any] = {}


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
            "which",
        ]
    )

    # These should be with the brain path
    # I will move these later.
    # No, this actually lives completely outside the core.
    dynamic_memory_path: str = "~/vm_drive/null-shift/MEMORY.md"
    dynamic_plan_path: str = "~/vm_drive/null-shift/plan.md"
    dynamic_task_path: str = "~/vm_drive/null-shift/task.md"
    dynamic_scratchpad_path: str = "~/.null-shift/scratchpad/"


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    # litellm fuckshit
    provider: Literal["openai", "vertexai", "litellm"] | str | None = None
    url: str | None = None
    api_key: str | None = None
    name: str = ""
    reasoning_effort: str | None = None
    reasoning: dict[str, Any] | None = None
    temperature: Optional[float] = Field(default=None, ge=0)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    max_tokens: Optional[int] = None
    extra_body: Optional[dict[str, Any]] = None

    def with_overrides(self, other: "ModelInfo") -> "ModelInfo":
        return self.model_copy(
            update=other.model_dump(
                exclude_defaults=True, exclude_none=True, exclude_unset=True
            ),
            deep=True,
        )

    def get_api_key(self) -> str | None:
        # code must start with '# API_KEY <-'
        # This is to distinguish from string api key
        # and tell what the variable name is.
        if self.api_key is None:
            return None
        elif self.api_key.strip().startswith("# API_KEY <-"):
            try:
                exec(self.api_key, API_KEY_GLOBALS)
                return API_KEY_GLOBALS.get("API_KEY", None)
            except Exception as e:
                logger.error("Error executing api key: %s", e)
                return None
        else:
            return self.api_key


class TaskModels(BaseModel):
    main: ModelIdentifier = None
    task_infer: ModelIdentifier = None
    history_summary: ModelIdentifier = None
    message_summary: ModelIdentifier = None


class LLM(BaseModel):
    default: ModelInfo = Field(default_factory=ModelInfo)
    model_tiers: dict[str, ModelInfo] = Field(default_factory=dict)

    vertexai_project_id: str = ""
    vertexai_location: str = "global"

    litellm_provider: Optional[str] = None

    models: TaskModels = Field(default_factory=TaskModels)


class Prompts(BaseModel):
    task_infer: str = 'You are an AI assistant that specializes in classifying user requests based on their complexity and requirements.\n\nYour task is to classify the user\'s input into one of the following categories:\n\n# Categories\n\n> Most questions fall to the Tool-Assisted Task category ("2").\n\n> If you\'re unsure, choose option "3".\n\n## 1. Simple Task\nThe task can be completed using only the model\'s internal knowledge, without needing external tools, real-time data, or research.\n\n## 2. Tool-Assisted Task\nThe task requires calling external tools or APIs (e.g., web search, calendar integration, file manipulation, command execution, window management) to complete a direct action.\n\n## 3. Autonomous Agent Task\nThe task is complex and requires a multi-step plan, extensive research, synthesis of information from multiple sources, or complex reasoning. Some tasks that would fall into this category:\n- Multi-step autonomous web tasks.\n- Tasks that take more than 5 tool calls.\n\n# Output\n\n## Examples\n\n### Simple Task\n```\nUser: What time is it?\nResponse: {\n  "difficulty": 1,\n  "reason": "Some common context is automatically provided such as time, date, ongoing events and computer state."\n}\n```\n\n### Tool-Assisted Task\n```\nUser: Can you add this event I\'m looking at to my calendar?\nResponse: {\n  "difficulty": 2,\n  "reason": "The AI needs a screenshot tool and calendar editing tool to complete this task."\n}\n```\n\n### Autonomous Agent Task\n```\nUser: Can you create a plan for a 2 week Japan trip?\nResponse: {\n  "difficulty": 3,\n  "reason": "This task requires complex planning, research and reasoning to be completed."\n}\n```'
    history_compression_system: str = "You are a memory-compression subsystem for an autonomous AI agent. Your task is to summarize the provided conversation log (which contains Human, AI, and Tool messages) into a dense, highly informative context block.\n\nThis summary will be prepended to the main AI agent's future prompts to serve as its memory of past events.\n\nCRITICAL DIRECTIVES:\n1. Retain the Core Objective: What is the user ultimately trying to accomplish?\n2. Preserve Crucial Data: Extract and keep specific file paths, URLs, code snippets, names, IDs, or environmental facts discovered via tools. Do not abstract these away.\n3. Track Progress: Briefly state what actions have successfully been completed.\n4. Note Failures/Constraints: Mention any tool failures or dead ends so the agent doesn't repeat the same mistakes.\n5. Identify Next Steps: Highlight any unresolved tasks, pending questions, or ongoing plans.\n6. Be Concise: Strip out all pleasantries, repetitive errors, and conversational filler. Use bullet points for readability.\n\nOUTPUT FORMAT:\n**User Objective:** [Brief statement of the overarching goal]\n**Established Context:** [Bullet points of crucial facts, exact paths, and extracted data]\n**Actions Taken:** [Brief list of what was already done/attempted]\n**Current State / Pending:** [What the agent was doing or needs to do right before this cutoff]"
    history_compression_human: str = "Summarize the current history."


class AgentConfig(BaseModel):
    infer_agent_mode: Literal[
        Difficulty.AUTONOMOUS_STRICT, Difficulty.AUTONOMOUS_TRAJECTORY
    ] = Difficulty.AUTONOMOUS_TRAJECTORY
    infer_default_fallback: Literal[1, 2, 3] = 2
    default_difficulty_fallback: Difficulty = Difficulty.AUTONOMOUS_TRAJECTORY

    continue_prompt: str = "[AGENT SYSTEM]: You haven't indicated completion intent. Continue the task or call the agent_complete_objective tool. Remeber to stay in the tool call loop by calling tools every response!"


class LogConfig(BaseModel):
    silenced_libraries: list[str] = Field(default_factory=list)
    level: str = "INFO"
    to_file: bool = False
    file_path: str = "core.log"


class StreamConfig(BaseModel):
    default_batch_task_title: str = "Info"
    max_iterations: int = 1000
    max_retries: int = 3
    retry_delay: float = 5.0


class HistoryConfig(BaseModel):
    length: int = 25
    path: str = "~/.null-shift/brain/history.json"
    compression: bool = True
    compression_threshold: int = 30
    compression_target_length: int = 15
    aggressive_compression: bool = False
    compression_timeout: float | None = None


class SocketConfig(BaseModel):
    input: str = "tcp://*:5555"
    output: str = "tcp://*:5556"
    default_response_topic: str = "response"


class PromptConfig(BaseModel):
    path: str = "prompts/"
    recursive: bool = False
    file_names: list[str] = [".py", ".md", ".xml"]
    function_name: str = "collect"
    function_timeout: float = 0.5


class ToolHandlingConfig(BaseModel):
    path: str = "tools/"
    recursive: bool = False
    per_module_timeout: float = 2.0
    min_refresh_delay: float = 5.0


class PermissionsConfig(BaseModel):
    yes_words: list[str] = Field(
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
    no_words: list[str] = Field(
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
    timeout: float = 30.0


class CoreConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    llm: LLM = Field(default_factory=LLM)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tool_config: ToolHandlingConfig = Field(default_factory=ToolHandlingConfig)
    prompt_config: PromptConfig = Field(default_factory=PromptConfig)
    prompts: Prompts = Field(default_factory=Prompts)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    socket: SocketConfig = Field(default_factory=SocketConfig)
    log: LogConfig = Field(default_factory=LogConfig)

    def get_model(self, identifier: ModelIdentifier) -> ModelInfo:
        """Get model based on identifier."""
        if model := self.llm.model_tiers.get(self.llm.default.name):
            self.llm.default = model

        if isinstance(identifier, ModelInfo):
            if identifier.name in self.llm.model_tiers:
                return self.llm.default.with_overrides(self.llm.model_tiers[identifier.name])
            return self.llm.default.with_overrides(identifier)
        elif identifier is None:
            return self.llm.default
        elif model := self.llm.model_tiers.get(identifier):
            return model

        return self.llm.default.with_overrides(ModelInfo(name=identifier))


class RuntimeState(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    shutdown_event: asyncio.Event = Field(default_factory=asyncio.Event)
    is_running: bool = False


manager = ConfigManager(Path("core_config.toml"), CoreConfig())
state = RuntimeState()
tool_manager = ConfigManager(Path("tool_config.toml"), ToolConfig())
