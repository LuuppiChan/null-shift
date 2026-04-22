"""
The Vector — Cognition's turn orchestrator.

Consumes :class:`~cognition.bus.BusMessage` events from the input queue,
drives the LLM streaming loop (including tool dispatch to the local registry
or the Action Node), and publishes results back onto the bus.

Types are in :mod:`cognition.types`. Media helpers are in :mod:`cognition.media`.
LLM backends are in :mod:`cognition.backends`.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING

from backends import LLMBackend, build_backend
from bus import BusMessage, CognitionBus
from config import CognitionConfig, load_config
from context import ContextAssembler
from custom_types import ToolCall
from history import HistoryManager
from prompt import PromptAssembler
from registry import ToolRegistry
from state import RuntimeState

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_TOPIC_STREAM = "assistant.stream"
_TOPIC_DONE = "assistant.stream.done"
_TOPIC_STATE = "state.changed"
_TOPIC_ACTION_REQ = "action.request"


class Vector:
    """The cognitive orchestrator.

    Consumes :class:`~cognition.bus.BusMessage` events, drives the LLM
    streaming loop — including tool dispatch — and publishes results back
    onto the bus.

    Args:
        config: Resolved configuration.
        bus: The :class:`~cognition.bus.CognitionBus` instance.
        state: Shared :class:`~cognition.state.RuntimeState`.
        registry: Hot-reload :class:`~cognition.registry.ToolRegistry`.
        context: :class:`~cognition.context.ContextAssembler`.
        prompt: :class:`~cognition.prompt.PromptAssembler`.
        history: :class:`~cognition.history.HistoryManager`.
        backend: The active :class:`~cognition.backends.LLMBackend`.
    """

    def __init__(
        self,
        config: CognitionConfig,
        bus: CognitionBus,
        state: RuntimeState,
        registry: ToolRegistry,
        context: ContextAssembler,
        prompt: PromptAssembler,
        history: HistoryManager,
        backend: LLMBackend,
        config_path: str | Path = "cognition/cognition.toml",
    ) -> None:
        from pathlib import Path

        self._config = config
        self._bus = bus
        self._state = state
        self._registry = registry
        self._context = context
        self._prompt = prompt
        self._history = history
        self._backend = backend

        self._config_path = Path(config_path)
        self._config_mtime = (
            self._config_path.stat().st_mtime if self._config_path.exists() else 0
        )

        self._batched_queue: list[BusMessage] = []
        self._pending_tool_results: asyncio.Queue[BusMessage] = asyncio.Queue()

        # Wire state mutations → publish state.changed on the bus.
        self._state._on_change = self._publish_state_sync

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def process_turn(self, event: BusMessage) -> None:
        """Process a single input event as a new conversation turn.

        Args:
            event: The inbound :class:`~cognition.bus.BusMessage` to process.
        """
        turn_id = str(uuid.uuid4())
        self._state.set_busy(turn_id)
        logger.info("Turn %s started — topic: %s", turn_id, event.topic)

        try:
            await self._run_turn(event, turn_id)
        except asyncio.CancelledError:
            logger.warning("Turn %s cancelled.", turn_id)
        except Exception as exc:
            logger.exception("Unhandled error during turn %s: %s", turn_id, exc)
        finally:
            self._state.set_idle()
            logger.info("Turn %s complete.", turn_id)

    async def abort(self) -> None:
        """Request an abort of the current turn at the next safe checkpoint."""
        if self._state.is_busy:
            logger.warning("Abort requested for turn %s.", self._state.turn_id)
            self._state.request_abort()

    def enqueue_batched(self, event: BusMessage) -> None:
        """Queue a batched event for later dispatch.

        Args:
            event: The batched :class:`~cognition.bus.BusMessage`.
        """
        self._batched_queue.append(event)
        logger.debug("Batched event queued (queue size: %d).", len(self._batched_queue))

    async def dispatch_batched(self) -> None:
        """Flush and process all queued batched events as a single turn.

        Bodies are merged into one ``instant``-type message. No-op if empty.
        """
        if not self._batched_queue:
            return

        events = list(self._batched_queue)
        self._batched_queue.clear()

        combined_body = "\n".join(
            f"## {e.payload.get('title', 'Event')}\n{e.payload.get('body', '')}"
            for e in events
        )
        merged = BusMessage(
            topic="input.batched",
            payload={
                "body": combined_body,
                "title": "Batched Events",
            },
        )
        logger.info("Dispatching %d batched events.", len(events))
        await self.process_turn(merged)

    async def feed_action_result(self, event: BusMessage) -> None:
        """Deliver an Action Node result to the in-flight turn.

        Args:
            event: The ``action.result`` :class:`~cognition.bus.BusMessage`.
        """
        await self._pending_tool_results.put(event)

    # ------------------------------------------------------------------
    # Turn internals
    # ------------------------------------------------------------------

    async def _run_turn(self, event: BusMessage, turn_id: str) -> None:
        """Core turn logic: build context, stream LLM, dispatch tools."""
        self._maybe_reload_config()
        self._registry.reload(self._config.path_tools)

        snapshot = await self._context.assemble()
        self._history.append_system(self._prompt.build(snapshot))

        text, media_parts, external_tools = self._parse_event(event)
        self._history.append_user(text, media_parts or None)
        self._history.trim()

        full_text = ""
        tool_schemas = self._registry.get_schemas() + external_tools
        logger.info(
            "Tool schemas for turn %s: %s",
            turn_id,
            [
                s.name if hasattr(s, "name") else s.get("function", {}).get("name")
                for s in tool_schemas
            ],
        )
        logger.debug("Full tool schemas: %s", tool_schemas)

        all_tool_calls_in_turn = []
        final_reason = "complete"

        for iteration in range(self._config.llm_max_iterations):
            if self._state.is_aborting:
                logger.info("Turn %s aborted at iteration %d.", turn_id, iteration)
                final_reason = "ABORT_LOCAL"
                break

            if iteration == self._config.llm_max_iterations - 1:
                final_reason = "TOOL_LIMIT"

            logger.debug(
                "LLM iteration %d / %d", iteration + 1, self._config.llm_max_iterations
            )
            delta_accumulator = ""
            got_tool_call = False

            async for chunk in self._backend.stream(
                self._history.to_messages(), tool_schemas
            ):
                if self._state.is_aborting:
                    final_reason = "ABORT_LOCAL"
                    break
                if chunk.is_done:
                    if chunk.finish_reason:
                        final_reason = chunk.finish_reason
                    break

                if chunk.delta_text:
                    delta_accumulator += chunk.delta_text
                    full_text += chunk.delta_text
                    await self._bus.publish(
                        _TOPIC_STREAM,
                        {"chunk": chunk.delta_text, "turn_id": turn_id},
                    )

                if chunk.tool_call:
                    got_tool_call = True
                    tc = chunk.tool_call
                    logger.info("Tool call: %s(%s)", tc.name, tc.args)
                    all_tool_calls_in_turn.append({"name": tc.name, "args": tc.args})

                    spoken_with_tool = delta_accumulator.strip() or None
                    delta_accumulator = ""

                    self._history.append_tool_call(
                        tc.call_id, tc.name, tc.args, content=spoken_with_tool
                    )
                    self._state.set_tool_active(tc.name)

                    result = await self._dispatch_tool(tc, turn_id)
                    self._history.append_tool_result(tc.call_id, result)
                    self._state.clear_tool()

                    # Save after every tool — tools may modify history directly.
                    await self._history.save()

            if delta_accumulator:
                self._history.append_assistant(delta_accumulator)

            if final_reason in ("ABORT_LOCAL", "TOOL_LIMIT"):
                break

            if not got_tool_call:
                break

        await self._bus.publish(
            _TOPIC_DONE,
            {
                "turn_id": turn_id,
                "reason": final_reason,
                "full_text": full_text,
                "tool_calls": all_tool_calls_in_turn,
            },
        )
        await self._history.save()

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    async def _dispatch_tool(self, tc: ToolCall, turn_id: str) -> str:
        """Execute a tool — local registry first, then Action Node via bus.

        Args:
            tc: The tool call to execute.
            turn_id: Current turn identifier (for correlation).

        Returns:
            str: The string result of the tool execution.
        """
        if tc.name in self._registry._tools:
            try:
                return await self._registry.call(tc.name, tc.args)
            except Exception as exc:
                logger.error("Local tool '%s' raised: %s", tc.name, exc)
                return f"Error executing {tc.name}: {exc}"

        await self._bus.publish(
            _TOPIC_ACTION_REQ,
            {
                "call_id": tc.call_id,
                "tool": tc.name,
                "args": tc.args,
                "turn_id": turn_id,
            },
        )
        logger.debug("Waiting for action.result call_id=%s", tc.call_id)

        deadline = asyncio.get_event_loop().time() + 30.0
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return f"Timeout waiting for tool '{tc.name}' result."
            try:
                result_event = await asyncio.wait_for(
                    self._pending_tool_results.get(),
                    timeout=min(remaining, 1.0),
                )
                if result_event.payload.get("call_id") == tc.call_id:
                    if result_event.payload.get("error"):
                        return f"Tool error: {result_event.payload['error']}"
                    return str(result_event.payload.get("result", ""))
                await self._pending_tool_results.put(result_event)
                await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------------
    # Configuration re-swapping
    # ------------------------------------------------------------------

    def _maybe_reload_config(self) -> None:
        """Check if cognition.toml has changed on disk; reload backend if so."""
        if not self._config_path.exists():
            return

        try:
            mtime = self._config_path.stat().st_mtime
        except OSError:
            return

        if mtime <= self._config_mtime:
            return

        logger.info("Configuration change detected — reloading.")
        old_provider = self._config.llm_provider
        old_model = self._config.llm_model

        load_config(self._config_path, into=self._config)
        self._config_mtime = mtime

        if (
            self._config.llm_provider != old_provider
            or self._config.llm_model != old_model
        ):
            logger.info(
                "Backend settings changed (%s/%s → %s/%s) — rebuilding backend.",
                old_provider,
                old_model,
                self._config.llm_provider,
                self._config.llm_model,
            )
            self._backend = build_backend(self._config)

    # ------------------------------------------------------------------
    # Input parsing
    # ------------------------------------------------------------------

    def _parse_event(
        self, event: BusMessage
    ) -> tuple[str, list[dict] | None, list[dict]]:
        """Convert a raw bus event to text, media, and external tools.

        Args:
            event: Inbound bus message.

        Returns:
            tuple: ``(text, media_parts, external_tools)``
        """
        payload = event.payload
        body: str = payload.get("body", "")
        title: str = payload.get("title", "")
        content: list[dict] | None = payload.get("content")
        external_tools: list[dict] = payload.get("tools", [])

        if event.topic in ("input.instant", "input.batched"):
            text = f"# {title}\n{body}" if title else body
        else:
            text = body

        return text, content, external_tools

    # ------------------------------------------------------------------
    # State publication
    # ------------------------------------------------------------------

    def _publish_state_sync(self, state_dict: dict) -> None:
        """Fire-and-forget state publisher wired into RuntimeState._on_change.

        Args:
            state_dict: Serialised state snapshot to broadcast.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._bus.publish(_TOPIC_STATE, state_dict))
        except RuntimeError:
            pass
