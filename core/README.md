# Null-Shift Core (Vector Engine)

> **Note:** This module is currently in active development. This README provides an overview of the current operational state and setup instructions.

Null-Shift Core is a robust, high-performance LLM orchestration engine designed for agentic workflows. It serves as the central "brain" for the Null-Shift ecosystem, managing LLM interactions, dynamic tool execution, state persistence, and task inference via an asynchronous ZeroMQ (ZMQ) message bus.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- ZeroMQ (ZMQ) libraries

### Setup & Run
1. Navigate to the `core` directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r dependencies.txt
   ```
4. Start the engine:
   ```bash
   python main.py
   ```

## ✨ Key Features & Architecture

- **ZMQ-Based Message Bus**: Uses `asyncio` ZeroMQ loops (`listener_loop`, `consumer_loop`, `message_loop`) for high-throughput, non-blocking communication over configured PULL/PUSH sockets.
- **Task Inference Engine**: Automatically evaluates user input using a fast, lightweight LLM to categorize requests into:
  1. *Simple* (Internal knowledge only)
  2. *Tool-Assisted* (Direct action via tool APIs)
  3. *Autonomous Agent* (Complex reasoning, multi-step plans)
- **Multiple LLM Backends**: Natively supports **VertexAI**, **OpenAI**, and **LiteLLM** providers via modular integrations.
- **Hot-Reloadable Configuration**: Managed via `core_config.toml`. Changes to UI settings, prompt parameters, socket bindings, or LLM targets apply instantly at runtime without restarting the main process.

## 📂 System Components

### 🧠 Dynamic Prompts (`prompts/`)
Vector dynamically assembles its system prompt by concatenating files from the `prompts/` directory in alphabetical order. 
- **Static Context**: Standard `.md`, `.txt`, or `.xml` files are read as raw text and appended to the system message.
- **Dynamic Context**: Python (`.py`) files can export a `collect() -> Optional[str]` function. Vector executes this at runtime to inject programmatic, real-time context (e.g., dynamic context, memory, role definition).

### 🛠️ Extensible Tools (`tools/`)
Tools are loaded dynamically from the `tools/` directory. 
- Built-in capabilities include foundational agent mechanisms (`agent_tools.py`), system-level integrations (`niri.py`, `windows.py`), and filesystem access (`files.py`, `linux_file_read.py`).
- Adding a new capability is as simple as dropping a LangChain-compatible BaseTool implementation into the folder.

### ⚙️ Backends (`backends/`)
Houses the implementation wrappers for LLM providers. By toggling `llm_provider` in the configuration, you can seamlessly switch between VertexAI, OpenAI, or LiteLLM endpoints to handle stream generation and tool calling functions.