# Cognition Node

The **Vector Core** of Null-Shift. A standalone async process that is the only node that talks to an LLM. It knows nothing about audio, TTS, or UI — it reads inputs from a ZMQ socket, thinks, and streams output back on another ZMQ socket.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [Socket Protocol](#socket-protocol)
5. [Writing Tools](#writing-tools)
6. [Writing Context Plugins](#writing-context-plugins)
7. [Writing Prompt Modules](#writing-prompt-modules)
8. [LLM Backends](#llm-backends)
9. [Runtime State](#runtime-state)

---

## Quick Start

```bash
# 1 — Set up environment
cd /home/luuppi/Documents/coding/projects/null-shift
python -m venv cognition/.venv
source cognition/.venv/bin/activate
pip install -r cognition/dependencies.txt

# 2 — Configure
#     Edit cognition/cognition.toml and set your API key

# 3 — Add a personality
echo "You are a helpful assistant." > cognition/personality.md

# 4 — Run
python -m cognition --config cognition/cognition.toml
```

To send a test message from another terminal:

```python
import zmq, json

ctx = zmq.Context()
sock = ctx.socket(zmq.PUSH)
sock.connect("tcp://localhost:5555")
sock.send_multipart([
    b"input.instant",
    json.dumps({"type": "instant", "body": "Hello!"}).encode()
])
```

---

## Architecture

```
cognition/
├── main.py               Entry point — event loop + sequential turn queue
├── config.py             CognitionConfig dataclass + TOML loader
├── bus.py                ZMQ PULL + PUB socket owner
├── state.py              RuntimeState (published as state.changed events)
├── registry.py           Async hot-reload tool registry
├── context.py            Async hot-reload context plugins
├── prompt.py             System prompt assembler
├── history.py            LangChain typed conversation history
├── vector.py             Vector orchestrator (turn loop)
├── types.py              ToolCall, StreamChunk dataclasses
├── media.py              mime_type → provider format converters
│
├── backends/
│   ├── __init__.py       LLMBackend ABC + build_backend() factory
│   ├── openai.py         OpenAI-compatible backend
│   └── vertexai.py       Google VertexAI / Gemini backend
│
├── tools/                Hot-reload tool plugins  (.py)
├── context_plugins/      Hot-reload context plugins (.py)
├── prompts/              Prompt modules (.md)
└── memory/
    ├── memory.md         Long-term persistent memory
    ├── expiring_notes.json  Short-lived behavioural constraints
    └── history.json      Persisted conversation history
```

### Data Flow

```
  Perception / any node
        │  PUSH → tcp://localhost:5555
        ▼
  ┌──────────────────────────────┐
  │  _listen_loop (bus.py)       │
  │  routes by topic             │
  │    abort → vector.abort()    │
  │    action.result → queue     │
  │    other → input_queue       │
  └──────────────┬───────────────┘
                 │ asyncio.Queue (sequential)
  ┌──────────────▼───────────────┐
  │  _consumer_loop              │
  │  await vector.process_turn() │  ← one turn at a time
  └──────────────┬───────────────┘
                 │
  ┌──────────────▼───────────────┐
  │  Vector._run_turn()          │
  │  1. reload tools + context   │
  │  2. build system prompt      │
  │  3. LLM stream loop          │
  │     ├─ publish tokens →      │──► SUB: assistant.stream
  │     └─ tool call?            │
  │         ├─ local registry    │
  │         └─ action.request →  │──► Action Node
  │  4. publish done →           │──► SUB: assistant.stream.done
  │  5. publish state.changed →  │──► SUB: state.changed
  └──────────────────────────────┘
```

---

## Configuration

Configuration is two-layer: `config.py` defines all defaults as a typed dataclass. `cognition.toml` provides user overrides — only keys you want to change need to appear.

```toml
# cognition/cognition.toml

[llm]
provider    = "openai"                    # "openai" | "vertexai"
model       = "gpt-4o-mini"
api_key     = "sk-..."
base_url    = "https://api.openai.com/v1" # change for local models
temperature = 0.7
max_history    = 40   # max messages kept in history
max_iterations = 15   # max LLM→tool→LLM loops per turn

# VertexAI only (ignored when provider = "openai")
# vertex_project  = "my-gcp-project"
# vertex_location = "us-central1"

[zmq]
input_bind  = "tcp://*:5555"   # PULL — Cognition binds here
output_bind = "tcp://*:5556"   # PUB  — Cognition binds here

[paths]
personality    = "cognition/personality.md"
memory         = "cognition/memory/memory.md"
history        = "cognition/memory/history.json"
tools          = "cognition/tools"
context_plugins = "cognition/context_plugins"
prompts        = "cognition/prompts"

[prompts]
layout = ["personality", "prompts", "long_term", "short_term", "context"]

[behaviour]
log_level = "INFO"   # DEBUG | INFO | WARNING | ERROR
```

---

## Socket Protocol

Cognition **binds** both sockets. All other nodes **connect** to these addresses.

### Input — PULL socket (default: `tcp://localhost:5555`)

Every message is a **two-frame ZMQ multipart message**:

```
Frame 1: topic  (UTF-8 string, e.g. "input.user")
Frame 2: payload (UTF-8 JSON)
```

#### Input Topics

| Topic | Payload | Effect |
|---|---|---|
| `input.user` | `{body, content?}` | Immediate user turn. |
| `input.instant` | `{body, title?, content?, tools?}` | Immediate system event turn. `title` is optional. `tools` can be an array of OpenAPI-style function schemas to dynamically inject for this turn. |
| `input.batched` | `{body, title?, content?}` | Queued for later batch dispatch. |
| `input.abort` | `{}` | Aborts the current turn. |
| `input.command` | `{cmd: str, ...}` | Execute internal commands (e.g. `dispatch_batched` or `poll_state`). |

#### Media Attachments

Any message can carry multimodal content via a `content` array:

```json
{
  "body": "What do you see in this image?",
  "content": [
    {
      "mime_type": "image/png",
      "data": "<base64-encoded bytes>"
    }
  ]
}
```

Supported MIME types: `image/*` (all backends), `application/pdf` (OpenAI only).

#### Python Sender Example

```python
import zmq
import json
import base64

ctx = zmq.Context()
sock = ctx.socket(zmq.PUSH)
sock.connect("tcp://localhost:5555")

def send(topic: str, payload: dict) -> None:
    sock.send_multipart([
        topic.encode(),
        json.dumps(payload).encode()
    ])

# Plain text
send("input.user", {"body": "What is the weather today?"})

# System event
send("input.instant", {
    "title": "Calendar",
    "body": "You have a meeting in 10 minutes."
})

# With external tools dynamically injected
send("input.instant", {
    "body": "What is the temperature in Helsinki?",
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetch live weather for a city."
        }
    }]
})

# With image
with open("screenshot.png", "rb") as f:
    data = base64.b64encode(f.read()).decode()

send("input.user", {
    "body": "Describe this screenshot.",
    "content": [{"mime_type": "image/png", "data": data}]
})

# Abort current turn (e.g. from VAD onset)
send("input.abort", {})
```

---

### Output — PUB socket (default: `tcp://localhost:5556`)

All output is **published** as two-frame multipart messages (same format as input). Subscribers can use ZMQ topic-prefix filtering to receive only what they need.

#### Output Topics

| Topic | Payload | Description |
|---|---|---|
| `assistant.stream` | `{chunk: str, turn_id: str}` | Incremental LLM token |
| `assistant.stream.done` | `{turn_id, reason, full_text, tool_calls}` | Turn complete |
| `state.changed` | See [Runtime State](#runtime-state) | Any state field changed |
| `action.request` | `{call_id, tool, args, turn_id}` | Tool call for Action Node |

`reason` in `stream.done` will be the upstream API stop reason (`STOP`, `TOOL_CALL`, `LENGTH`), or a local orchestrator reason (`ABORT_LOCAL`, `TOOL_LIMIT`). `tool_calls` is a list of all tools executed during that turn.

#### Python Subscriber Example

```python
import zmq
import json

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://localhost:5556")

sock.subscribe(b"assistant.stream")   # token stream
sock.subscribe(b"state.changed")       # state events

while True:
    topic_b, payload_b = sock.recv_multipart()
    topic = topic_b.decode()
    payload = json.loads(payload_b)

    if topic == "assistant.stream":
        print(payload["chunk"], end="", flush=True)
    elif topic == "assistant.stream.done":
        print(f"\n[done: {payload['reason']}]")
    elif topic == "state.changed":
        print(f"[state] busy={payload['is_busy']}")
```

---

## Writing Tools

Tools are `async` Python functions placed in `cognition/tools/`. They are **hot-reloaded by mtime** at the start of every turn — drop a file in and it is available immediately, no restart needed.

### Rules

1. The file must be in `cognition/tools/` (or the configured `path_tools`)
2. Each tool function must be decorated with `@tool` from `cognition.registry`
3. Tool functions must be `async def`
4. Parameters must have type annotations — they are used to build the LangChain tool schema
5. The docstring becomes the tool's description visible to the LLM

### Example: `cognition/tools/system_info.py`

```python
from registry import tool
import platform

@tool
async def get_system_info() -> str:
    """Return basic information about the system this assistant runs on."""
    return f"{platform.system()} {platform.release()} — {platform.node()}"
```

### Example: With `Literal` parameters

```python
from registry import tool
from typing import Literal
import subprocess

@tool
async def set_volume(
    direction: Literal["up", "down", "mute"],
    amount: int = 5,
) -> str:
    """Adjust the system volume.

    Args:
        direction: Whether to raise, lower, or mute the volume.
        amount: Percentage to change by (ignored when muting).
    """
    if direction == "mute":
        subprocess.run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "1"])
        return "Muted."
    sign = "+" if direction == "up" else "-"
    subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{amount}%{sign}"])
    return f"Volume {direction} by {amount}%."
```

`Literal["up", "down", "mute"]` produces an enum constraint in the LangChain tool schema — the LLM is constrained to those values.

### Local vs. Action Node Tools

- **Local tools** (in `cognition/tools/`) run in-process inside Cognition. Use for fast, safe operations (reading files, calling system commands, querying APIs).
- **Action Node tools** are dispatched via `action.request` on the output PUB socket and the result is expected back on the input PULL socket as `action.result`. Use for heavy, isolated, or UI-affecting operations.

If the LLM calls a tool name that is **not** in the local registry, Cognition publishes `action.request` and waits up to 30 seconds for the Action Node to respond.

---

## Writing Context Plugins

Context plugins assemble the **Current Context** block injected into the system prompt every turn. Each plugin is a `.py` file in `cognition/context_plugins/`, hot-reloaded by mtime.

### Rules

1. Each file must expose `async def collect() -> str | None`
2. Return a non-empty string to add a line to the context block
3. Return `None` or `""` to be silently skipped
4. Each plugin has a **0.5 second timeout** — keep them fast

### Example: `cognition/context_plugins/current_time.py`

```python
from datetime import datetime

async def collect() -> str | None:
    return f"Time: {datetime.now().strftime('%H:%M on %A, %B %d, %Y')}"
```

### Example: `cognition/context_plugins/active_window.py`

```python
import asyncio

async def collect() -> str | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            "niri", "msg", "-j", "focused-window",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=0.3)
        import json
        data = json.loads(stdout)
        title = data.get("title", "")
        app = data.get("app_id", "")
        if title or app:
            return f"Active Window: '{title}' ({app})"
    except Exception:
        pass
    return None
```

All plugin outputs are joined with newlines and placed under `# Current Context` in the system prompt.

---

## Writing Prompt Modules

Prompt modules are `.md` files in `cognition/prompts/`. They are concatenated **alphabetically by filename** after the personality and before the context block.

### Rules

1. Files are sorted alphabetically — use numeric prefixes to control order: `00_`, `10_`, `20_`, ...
2. **The filename is not used as a header** — each file owns its own headings
3. Files are hot-reloaded on every `build()` call (every turn)

### Example: `cognition/prompts/00_tool_usage.md`

```markdown
# Tool Usage

When you need to perform an action, use the tools provided to you. Always
prefer tools over describing hypothetical steps. If a tool fails, try once
with adjusted arguments before reporting the failure to the user.
```

### Example: `cognition/prompts/10_response_style.md`

```markdown
# Response Style

- Be concise. Prefer short answers unless the user asks for depth.
- When using voice output, avoid lists, markdown, and special characters.
- Never start a response with "I" or "As an AI".
```

### System Prompt Structure

The final system prompt sent to the LLM is assembled exactly in the order specified by the `[prompts] layout` list array in your `cognition.toml`. 

The default layout list is `["personality", "prompts", "long_term", "short_term", "context"]`.

For example, using the default list, your output will look like this:

```
{personality.md content}

{prompts/00_*.md}
{prompts/10_*.md}
...

# Memory
{memory/memory.md content}

# Active Constraints
{memory/expiring_notes.json}

# Current Context
{output of all context plugins}
```

If an item is missing from the list (for instance, `["personality", "prompts"]`), the underlying memory and context plugins will still run but their output will be silently dropped from the final LLM prompt.

---

## LLM Backends

The backend is selected by `config.llm_provider`. Swapping providers requires only a config change.

### `openai` (default)

Works with any endpoint speaking the OpenAI API:

```toml
[llm]
provider = "openai"
model    = "gpt-4o"
api_key  = "sk-..."
base_url = "https://api.openai.com/v1"
```

For **local models** (Ollama, LM Studio, vLLM):

```toml
[llm]
provider = "openai"
model    = "llama3.2"
api_key  = "not-used"
base_url = "http://localhost:11434/v1"   # Ollama
```

### `vertexai`

```toml
[llm]
provider        = "vertexai"
model           = "gemini-2.0-flash"
vertex_project  = "my-gcp-project"
vertex_location = "us-central1"
```

Authentication uses Application Default Credentials (`gcloud auth application-default login`). No `api_key` needed.

### Adding a New Backend

1. Create `cognition/backends/my_provider.py`
2. Subclass `LLMBackend` from `cognition.backends`
3. Implement `async def stream(messages, tools) -> AsyncIterator[StreamChunk]`
4. Register it in `cognition/backends/__init__.py`'s `build_backend()`:

```python
if config.llm_provider == "my_provider":
    from backends.my_provider import MyProviderBackend
    return MyProviderBackend(config)
```

---

## Runtime State

State is **event-driven**. Whenever a field changes, Cognition publishes `state.changed` on the output PUB socket. Other nodes subscribe instead of polling.

### State Fields

| Field | Type | Description |
|---|---|---|
| `is_busy` | `bool` | `True` while a turn is in progress |
| `turn_id` | `str \| null` | UUID of the current turn |
| `is_aborting` | `bool` | `True` if an abort was requested |
| `tool_active` | `str \| null` | Name of the tool currently executing |
| `at_finished` | `float` | Unix timestamp of the last completed turn |

### State Polling

While the primary pattern is subscribing to `state.changed`, you can actively request the current state on demand:

```python
send("input.command", {
    "cmd": "poll_state",
    "reply_topic": "my.custom.reply"  # optional, defaults to "state.reply"
})
```

Cognition will immediately publish its full state dictionary to the specified topic on the PUB socket.

### Waiting for Idle (Agent Callback Pattern)

Subscribe to `state.changed` and wait for `is_busy == false`:

```python
sock.subscribe(b"state.changed")

while True:
    _, payload_b = sock.recv_multipart()
    state = json.loads(payload_b)
    if not state["is_busy"]:
        break  # Safe to send next turn
```
