# Project Null-Shift

A complete architectural rewrite designed to solve the structural bottlenecks of traditional monolithic assistants. **Null-Shift** moves beyond the linear "Listen-Think-Speak" loop, adopting a **reactive mesh of independent Nodes**.

---

## The "Vector" Philosophy

In Null-Shift, the LLM is no longer referred to as an "Agent" or "Assistant"—it is the **Vector**. It is a hybrid cognitive core that doesn't just respond to commands; it represents a trajectory of intent. The Vector is isolated from the hardware (Audio/UI), interacting with the world solely through a high-speed asynchronous data bus.

## The Tri-Node Architecture

Everything is split into three replaceable, specialized nodes that communicate over **ZeroMQ (ZMQ)** using a mix of Pub/Sub and Request/Response patterns.

### 1. Perception (The Ingest Layer)
Responsible for capturing the state of the world and feeding it to the Vector.

- **Automated STT**: Dedicated process running local Whisper (Vulkan/C++). It performs its own VAD (Voice Activity Detection) and publishes clean text.
- **Event Sourcing**: System events (notifications, calendar alerts, file changes) are treated as first-class inputs, identical to voice.
- **The "Barge-in" Signal**: Perception nodes publish an `ONSET` event the moment they detect user speech. This allows all other nodes to react instantly before a single word is even transcribed.

### 2. Cognition (The Vector Core)
The "Brain" of the project. It consumes inputs and coordinates the output flow.

- **Reactive Async Loop**: Built on `asyncio`. It listens to the `stt.transcript` and `system.events` topics.
- **Broadcast Output**: Instead of returning a string, Cognition acts as a **ZMQ Publisher**. It streams tokens/sentences to an `assistant.stream` topic.
    - **One-to-Many**: It doesn't care if there is one speaker, multiple UI screens, or a logging server connected. It simply shouts its thoughts into the bus.
- **Modular Action Sandbox**: Tools are decoupled from the main engine. When the Vector wants to act, it sends a `command.request` and waits for an asynchronous response from the Action Node.
- **Shared States**: A persistent "Registry" (Redis-backed) stores the global state (emotions, current focus, volume) accessible by all nodes.

### 3. Synthesis (The Manifestation Layer)
The output sinks that translate the Vector's digital thoughts into human-perceivable signals.

- **Decoupled TTS**: Multiple TTS nodes can coexist. One might use a fast, low-latency engine (Piper) for immediate feedback, while another generates a high-quality summary in the background.
- **Visual Sinks**: Real-time dashboards, Tauri desktop apps, or terminal UIs that subscribe to the same stream as the TTS.
- **Interruption Resilience**: Sinks subscribe to the `vad.onset` topic. When the user starts talking, the TTS node kills its audio output in **nanoseconds**, not seconds, eliminating the "talking-over-user" lag of V1.

---

## Communication Protocols (The IPC Bus)

Null-Shift uses **Unix Domain Sockets** with **ZeroMQ** for the underlying plumbing:

- **Broadcasting (`PUB/SUB`)**: Used for the Assistant's voice, thought stream, and system-wide VAD triggers.
- **Messaging (`REQ/REP`)**: Used for specific node configurations and tool-call handshakes.

This architecture means you can update the STT model, swap the TTS provider, or even move the entire Cognition engine to a different machine without restarting the other components.
