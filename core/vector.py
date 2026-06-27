import asyncio
import logging
from pathlib import Path
from typing import Any, Optional, cast

import openai
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import tool
from pydantic import ValidationError

from core.agent import AgentData, convert_difficulty, infer_difficulty
from core.backends import LLMBackend, get_backend
from core.config import manager, state
from core.context import get_context
from core.core_data import LocalData
from core.history import History
from core.registry import LLMTool, get_tools
from core.socket_system import socket_out
from global_tools import Signal
from global_types import (
    BusMessage,
    Commands,
    Difficulty,
    InputCommand,
    InputMessage,
    MessageTopic,
    MessageType,
    command_response,
    convert_to_langchain_media,
    is_autonomous,
)
from output_message import OutputMessage

logger = logging.getLogger(__name__)


class Vector:
    def __init__(self, input_queue: asyncio.Queue[BusMessage]) -> None:
        global vector
        vector = self

        self.llm: LLMBackend
        self.input_queue: asyncio.Queue[BusMessage] = input_queue
        self.message_queue: asyncio.Queue[InputMessage] = asyncio.Queue()
        self.history = History()
        self.batch: list[InputMessage] = []
        self.data: LocalData = LocalData()
        self.history.added.connect(self.data._add_history)
        self.abort: Signal = Signal()
        self.abort.connect(self._on_abort)

    def _on_abort(self):
        logger.info("Abort triggered!")
        logger.info("Passing through the abort message.")
        asyncio.create_task(socket_out.send(BusMessage(topic=MessageTopic.ABORT)))
        logger.info("Clearing message queue due to abort.")
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get_nowait()
                self.message_queue.task_done()
                logger.info("Clearing message: %s", message)
            except asyncio.QueueEmpty:
                break

    async def consumer_loop(self):
        """Handles input queue messages and passes them to the process_input."""
        while not state.shutdown_event.is_set():
            try:
                message = await asyncio.wait_for(self.input_queue.get(), timeout=0.1)
                logger.info("Got input from topic: %s", message.topic)
            except asyncio.TimeoutError:
                continue

            await self.process_input(message)

    async def message_loop(self):
        """Handles messages in the message_queue one by one."""
        while not state.shutdown_event.is_set():
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            await socket_out.send(BusMessage(topic=MessageTopic.STARTED))
            state.is_running = True
            handle = asyncio.create_task(self._send_instant(message))

            handle.add_done_callback(lambda task: self.abort.disconnect(task.cancel))
            self.abort.connect(handle.cancel)

            while not handle.done():
                await asyncio.sleep(0.1)

            if handle.cancelled():
                logger.info("Message stream task was cancelled.")
            else:
                # Idk if it returns the error or raises an error.
                try:
                    e = handle.exception()
                    if e:
                        logger.critical(
                            "Stream returned an exception: %s",
                            e,
                            stack_info=True,
                            exc_info=True,
                        )
                except Exception as e:
                    logger.error(
                        "Stream returned an exception: %s",
                        e,
                        stack_info=True,
                        exc_info=True,
                    )

            await socket_out.send(BusMessage(topic=MessageTopic.FINISHED))
            state.is_running = False

    async def process_input(self, message: BusMessage):
        """Takes in a raw BusMessage and processes it accordingly."""
        topic = message.topic.split(".")
        if topic[0] == MessageTopic.INPUT:
            try:
                input_message = InputMessage.model_validate(
                    message.payload, strict=False, extra="ignore"
                )
            except ValidationError as e:
                logger.error("Error validating input message from socket: %s", e)
                return

            await self._handle_input(input_message)

        elif topic[0] == MessageTopic.COMMAND:
            await self._handle_command(message)

        else:
            logger.warning("Unknown input topic: %s", message.topic)

    async def _handle_input(self, message: InputMessage):
        match message.type:
            case MessageType.BATCHED:
                logger.info(
                    "Batching message: %s",
                    (
                        message.title
                        if message.title
                        else "[No title]" + ": " + message.body
                    ),
                )
                self.batch.append(message)
            case MessageType.INSTANT:
                logger.info(
                    "Adding message to queue: %s",
                    (
                        message.title
                        if message.title
                        else "[No title]" + ": " + message.body
                    ),
                )
                await self.message_queue.put(message)
            case _:
                logger.warning("Unknown message type: %s", message.type)

    async def _send_instant(self, message: InputMessage):
        """
        Handles an instant message type assuming the current InputMessage is instant.
        Returns after the full message has been handled. (LLM response and tools.)
        """
        self.history.load()
        self.history.validate_history()
        compressed = await self.history.trim_history()
        if compressed:
            self.data.last_compression = compressed

        config = manager.get_config()

        content: list[str | dict[Any, Any]] = []
        formatted = []
        if message.title:
            formatted.append("# " + message.title)
        formatted.append(message.body)
        text = "\n".join(formatted)
        content.append({"type": "text", "text": text})

        if message.media:
            content.extend(convert_to_langchain_media(message.media))

        if message.difficulty is not None:
            if message.difficulty == Difficulty.INFER:
                inferred_difficulty = infer_difficulty(text)
                self.data.agent.difficulty = convert_difficulty(inferred_difficulty)
            else:
                self.data.agent.difficulty = message.difficulty
        else:
            self.data.agent.difficulty = config.agent.default_difficulty_fallback

        self.data.agent.completed = False
        self.data.agent.goal = message.goal
        self.data.agent.context = message.context

        self.history.append(HumanMessage(content))
        logger.info("USER: %s", text)

        i = 0
        retries = 0

        while True:
            config = manager.get_config()
            max_iterations = config.stream.max_iterations
            # Reload every turn
            if self.data.agent.difficulty == Difficulty.SIMPLE:
                # No tools for simple requests
                logger.info("Simple mode selected. No tools are given for the LLM.")
                tools = {}
            else:
                tools = await get_tools()
                logger.info("Got %s tool(s) for the LLM.", len(tools))

            logger.info(
                "LLM iteration %s/%s, retries %s/%s",
                i,
                max_iterations,
                retries,
                config.stream.max_retries,
            )
            try:
                full_response = await self._process_llm_stream(tools)
            except openai.BadRequestError as e:
                # Code 400 I think, this means the user config is invalid.
                logger.error("Bad request error: %s", e)
                break
            except Exception as e:
                logger.error("Stream error: %s", e)
                logger.debug("Stream hisotory at error time: %s", self.history.messages)
                retries += 1
                logger.info("Stream retries: %s/%s", retries, config.stream.max_retries)
                if retries >= config.stream.max_retries:
                    logger.error("Max retries reached, aborting stream.")
                    break
                else:
                    await asyncio.sleep(config.stream.retry_delay)
                    continue

            # Reset back after successful stream
            retries = 0

            if full_response is None:
                logger.error("Full response object is None after stream iteration.")
                return
            elif full_response.content == "" and not full_response.tool_calls:
                logger.error("Full response object is empty after stream iteration.")
                logger.debug("Full response object: %s", full_response)
                retries += 1
                continue
            else:
                msg = ""
                if isinstance(full_response.content, str):
                    msg = full_response.content
                elif isinstance(full_response.content, list):
                    if full_response.content:
                        for block in full_response.content:
                            # Never is a list of strings, look above (now deleted)
                            # What the fuck it was a list of strings.
                            # If the first chunk is not a string it goes here.
                            if isinstance(block, str):
                                msg += block
                            elif isinstance(block, dict):
                                for (
                                    k,
                                    v,
                                ) in block.items():
                                    if k == "text":
                                        msg += str(v)
                            else:
                                logger.error("Unhandled message block: %s", block)
                        full_response.content = msg
                    else:
                        msg = "[Empty content block]"
                else:
                    logger.error(
                        "Error finding message. Full content: %s", full_response.content
                    )
                logger.info("LLM: %s", msg.strip())

            self.history.append(full_response)
            # Save and validate here because of the validation system
            self.history.save()
            compressed = await self.history.trim_history()
            if compressed is not None:
                # this has the issue of the compression context not getting to the model
                # on the next compression
                self.data.last_compression = compressed

            if i >= max_iterations:
                logger.warning("Exeeded max LLM iterations (%s)", max_iterations)
                break

            if full_response.tool_calls:
                self.history.extend(
                    await self._handle_tool_calls(tools, full_response.tool_calls)
                )
            elif is_autonomous(self.data.agent.difficulty):
                if self.data.agent.completed:
                    break
                else:
                    logger.error("LLM nudge triggered.")
                    self.history.append(HumanMessage(config.agent.continue_prompt))
            else:
                break

            i += 1

        self.history.save()

    async def _process_llm_stream(
        self, tools: dict[str, LLMTool]
    ) -> Optional[AIMessage]:
        full_response: Optional[AIMessage] = None
        cfg = manager.get_config()
        model = cfg.get_model(cfg.llm.models.main)
        self.llm = get_backend(model)
        context = await get_context(self.data)
        logger.info(
            "Running with system prompt with a length of %s", len(context.content)
        )

        async for chunk in self.llm.stream(
            self.history.with_system_message(context),
            list(tools.values()),
        ):
            logger.debug("Full chunk: %s", chunk)
            # To only get the text delta
            content = chunk.content
            # I think this is valid, but the type checker doesn't like it so I'm casting.
            full_response = (
                chunk
                if full_response is None
                else cast(AIMessage, full_response + chunk)
            )
            out_msg = OutputMessage()

            match content:
                case str():
                    # It's a simple string (most common for text LLMs)
                    if content:
                        out_msg.text = content
                case []:
                    # Ignore empty content blocks
                    pass
                case list() if content and isinstance(content[0], dict):
                    # It's a list of dicts (Complex/Multi-modal/Tool blocks)
                    # Vertexai uses this
                    part: dict | str
                    for part in content:
                        if isinstance(part, str):
                            if part:
                                out_msg.text = part
                        else:
                            match part.get("type"):
                                case "text":
                                    out_msg.text = part.get("text")
                                case "thinking":
                                    out_msg.reasoning = part.get("thinking")
                                case None:
                                    logger.warning(
                                        "Type not found in answer dict part: %s", part
                                    )
                                case _:  # Handle other instances if they come
                                    logger.error(
                                        "Unhandled dict type: %s", part.get("type")
                                    )
                case list() if all(isinstance(item, str) for item in content):
                    # It's a list of strings
                    logger.warning(
                        "LLM message content is a list of strings. This was unexpected: %r",
                        content,
                    )
                case _:
                    logger.error("LLM message content not recognized: %r", content)

            if out_msg.reasoning is None:
                reasoning_content = chunk.additional_kwargs.get(
                    "reasoning_content"
                ) or chunk.additional_kwargs.get("reasoning")
                if reasoning_content:
                    out_msg.reasoning = reasoning_content

            await socket_out.send(out_msg.to_bus(MessageTopic.STREAM))

        # send full response
        out_msg = OutputMessage()
        if full_response is not None:
            out_msg.reasoning = full_response.additional_kwargs.get(
                "reasoning_content"
            ) or full_response.additional_kwargs.get("reasoning")

        if full_response and isinstance(full_response.content, list):
            text_parts = []
            reasoning_parts = []
            for part in full_response.content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict):
                    ptype = part.get("type")
                    if ptype == "text":
                        text_parts.append(part.get("text", ""))
                    elif ptype == "thinking":
                        reasoning_parts.append(part.get("thinking", ""))

            if reasoning_parts:
                existing_reasoning = out_msg.reasoning or ""
                out_msg.reasoning = existing_reasoning + "".join(reasoning_parts)

            out_msg.text = "".join(text_parts) if text_parts else None
        else:
            out_msg.text = str(full_response.content if full_response else None)

        await socket_out.send(out_msg.to_bus(MessageTopic.FULL))
        return full_response

    async def _handle_tool_calls(
        self, tools: dict[str, LLMTool], tool_calls: list[ToolCall]
    ) -> list[ToolMessage]:
        logger.info("LLM requested %s tool(s)", len(tool_calls))
        for tool_call in tool_calls:
            await socket_out.send(
                OutputMessage(
                    tool_call_id=tool_call["id"],
                    tool_name=tool_call["name"],
                    tool_args=tool_call["args"],
                ).to_bus(MessageTopic.TOOL_CALL)
            )

        tasks = []

        for tool_call in tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            call_id = tool_call["id"]

            @tool(name)
            def inexistent(*args: Any, _name: str = name, **kwargs: Any) -> str:
                """Inexistent tool placeholder."""
                logger.warning(
                    "LLM called tool %s with args %s and kwargs %s", _name, args, kwargs
                )
                return f"You tried to call an inexistent tool '{_name}'."

            func = tools.get(name, inexistent)

            tasks.append(self._handle_single_tool(func, args, call_id))

        results = await asyncio.gather(*tasks)

        for result in results:
            await socket_out.send(
                OutputMessage(
                    tool_result=result.content,
                    tool_call_id=result.tool_call_id,
                    tool_name=result.name,
                ).to_bus(MessageTopic.TOOL_RESULT)
            )

        return results

    async def _handle_single_tool(
        self, func: LLMTool, args: dict[str, Any], call_id: str | None
    ) -> ToolMessage:
        """Helper to run a single tool returning a tool message."""
        try:
            output = await asyncio.to_thread(lambda: func.run(args))
            # There's an argument called 'artifact' which is a system context field not sent to the LLM.
            return ToolMessage(content=output, tool_call_id=call_id, name=func.name)
        except Exception as e:
            logger.error("Error executing tool '%s': %s", func.name, e)
            return ToolMessage(
                content=f"Error executing tool: {e}",
                tool_call_id=call_id,
                name=func.name,
            )

    async def _send_batched(self):
        if not self.batch:
            logger.warning("Tried to flush an empty batch.")
            return

        logger.info("Flushing %s batched message(s).", len(self.batch))
        parts = []

        for item in self.batch:
            title = (
                item.title
                if item.title
                else manager.get_config().stream.default_batch_task_title
            )
            parts.append(f"## {title}\n" + item.body)

        body = "# Batched events\n\n" + "\n\n".join(parts)
        message = InputMessage(body=body)
        self.batch.clear()
        await self.message_queue.put(message)

    async def _handle_command(self, message: BusMessage):
        """Handle a command type message assuming BusMessage is a command."""
        try:
            command = InputCommand.model_validate(
                message.payload, strict=True, extra="ignore"
            )
        except ValidationError as e:
            logger.error("Error validating input command from socket: %s", e)
            return

        match command.command.lower():
            case Commands.FLUSH_BATCH:
                await self._send_batched()
            case Commands.ABORT:
                self.abort.emit()
            case Commands.IS_RUNNING:
                if command.args:
                    response_topic = command.args[1]
                else:
                    response_topic = MessageTopic.COMMAND_RESPONSE

                await socket_out.send(
                    BusMessage(
                        topic=response_topic,
                        payload=command_response(is_running=state.is_running),
                    )
                )
            case _:
                logger.warning("Invalid command: %s", command.command)


# global reference to the main vector instance
# this is assigned by the main function.
vector: Vector
