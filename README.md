# Null-Shift

A modular AI agent system with multiple LLM backends, voice I/O, and tool execution capabilities.

## Architecture

Null-Shift is built as a collection of independent components communicating through a ZeroMQ message bus:

- **[`core/`](core/)** - LLM backends, conversation history, configuration management, and agent loop
- **[`tools/`](tools/)** - File editing, browser automation, and system utilities
- **[`gui/`](gui/)** - Flet-based desktop interface
- **[`perception/`](perception/)** - Speech-to-text processing
- **[`synthesis/`](synthesis/)** - Text-to-speech output
- **[`text_chat/`](text_chat/)** - Terminal-based chat interface

## Features

- **Multi-provider LLM support** - OpenAI, Google Vertex AI, and LiteLLM backends
- **Adaptive conversation modes** - Simple Q&A, tool-assisted, or fully autonomous agentic tasks
- **Tool integration** - File operations, browser automation, and custom tool registry
- **Voice interface** - Real-time speech recognition and text-to-speech
- **Conversation management** - JSON-backed history with automatic summarization and compression
- **Hot-reloadable config** - TOML-based configuration with live updates

## Quick Start

```bash
# Install dependencies
pip install -r dependencies.txt

# Run the core agent
python -m null_shift core

# Launch the GUI
python -m null_shift gui

# Start the CLI chat
python -m null_shift text
```

## Configuration

Configuration is stored in TOML and supports hot-reloading. See `global_tools.py` for the `ConfigManager` implementation.

## Project Structure

```
null_shift/
├── core/             # Core agent logic, backends, history
├── tools/            # File and browser automation tools
├── gui/              # Flet desktop interface
├── perception/       # Audio input processing
├── synthesis/        # Audio output processing
├── text_chat/        # Terminal-based chat frontend
└── global_*.py       # Shared utilities and type definitions
```
