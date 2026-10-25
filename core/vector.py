import asyncio
import json
import logging
from typing import Any, Optional, cast

import openai
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import tool
from openrouter.errors.badrequestresponse_error import BadRequestResponseError
from pydantic import ValidationError

from core.agent import convert_difficulty, infer_difficulty
from core.backends import LLMBackend, get_backend
from core.config import manager, state
from core.context import get_context
from core.core_data import LocalData
from core.dynamic_context import get_dynamic_context
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
    PendingPermissionRequest,
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
        self.cancelled_messages: list[str] = []
        self.messages_in_queue: list[str] = []
        """To see whether we even have the input message with that id."""
        self.data: LocalData = LocalData()
        self.history.added.connect(self.data._add_history)
        self.abort: Signal = Signal()
        self.abort.connect(self._on_abort)

    def _on_abort(self) -> None:
        logger.info("Abort triggered!")
        logger.info("Passing through the abort message.")
        state.is_running = False
        asyncio.create_task(socket_out.send(BusMessage(topic=MessageTopic.ABORT)))
        logger.info("Clearing message queue due to abort.")
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get_nowait()
                self.message_queue.task_done()
                logger.info("Clearing message: %s", message)
            except asyncio.QueueEmpty:
                break

    async def consumer_loop(self) -> None:
        """Handles input queue messages and passes them to the process_input."""
        while not state.shutdown_event.is_set():
            try:
                message = await asyncio.wait_for(self.input_queue.get(), timeout=0.1)
                logger.info("Got input from topic: %s", message.topic)
            except asyncio.TimeoutError:
                continue

            await self.process_input(message)

    async def message_loop(self) -> None:
        """Handles messages in the message_queue one by one."""
        while not state.shutdown_event.is_set():
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                if self.message_queue.empty():
                    # to prevent memory leaks from other bad code
                    self.messages_in_queue.clear()
                    self.cancelled_messages.clear()
                continue

            if message.id in self.messages_in_queue:
                self.messages_in_queue.remove(message.id)
            else:
                logger.warning(
                    "Invalid state. Message that was in the queue, was not in the queue id list. %s",
                    message,
                )

            if message.id in self.cancelled_messages:
                logger.info("Found cancelled message. '%s'", message.id)
                try:
                    self.cancelled_messages.remove(message.id)
                except ValueError as e:
                    logger.warning("Invalid state. Message id not in ids: %s", e)
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
                    logger.critical(
                        "Stream returned an exception: %s",
                        e,
                        stack_info=True,
                        exc_info=True,
                    )

            state.is_running = False
            await socket_out.send(BusMessage(topic=MessageTopic.FINISHED))

    async def process_input(self, message: BusMessage) -> None:
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

        elif topic[0] == MessageTopic.PERMISSION_RESPONSE:
            await self._handle_permission(message)

        else:
            logger.warning("Unknown input topic: %s", message.topic)

    async def _handle_permission(self, response: BusMessage):
        logger.info("Adding permission request response to state list.")
        try:
            msg = PendingPermissionRequest.model_validate(**response.payload)
            for req in state.pending_permission_requests:
                if req.id == msg.id:
                    state.pending_permission_responses.append(msg)
                    break
            else:
                logger.warning(
                    "Didn't find a matching permission request for the response."
                )

        except ValidationError:
            logger.error(
                "Error validating PendingPermissionRequest: %s", response.payload
            )

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
                self.messages_in_queue.append(message.id)
                logger.info(
                    "Adding message to queue (%s): %s",
                    message.id,
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

        self.data.current_turn = HumanMessage
        dyn_ctx = await get_dynamic_context(self.data)
        text = message.build(dyn_ctx)
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

        human_message = HumanMessage(content)
        self.history.append(human_message)
        logger.info("USER: %s", text)

        i = 0
        retries = 0
        cost = 0.0
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        reasoning_tokens = 0

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
                if full_response is not None:
                    request_cost = full_response.response_metadata.get("cost")

                    if isinstance(request_cost, float):
                        cost += request_cost
                    else:
                        logger.warning("Malformed request cost: %s", request_cost)

                    if (usage_metadata := full_response.usage_metadata) is not None:
                        request_input_tokens = usage_metadata.get("input_tokens")
                        request_output_tokens = usage_metadata.get("output_tokens")
                        input_token_details: dict | Any = usage_metadata.get(
                            "input_token_details"
                        )
                        output_token_details: dict | Any = usage_metadata.get(
                            "output_token_details"
                        )
                        if isinstance(request_input_tokens, int):
                            input_tokens += request_input_tokens
                        else:
                            logger.warning(
                                "Malformed request input tokens: %s",
                                request_input_tokens,
                            )

                        if isinstance(request_output_tokens, int):
                            output_tokens += request_output_tokens
                        else:
                            logger.warning(
                                "Malformed request output tokens: %s",
                                request_output_tokens,
                            )

                        if isinstance(input_token_details, dict):
                            if isinstance(
                                request_cached_tokens := input_token_details.get(
                                    "cache_read", 0
                                ),
                                int,
                            ):
                                cached_tokens += request_cached_tokens
                            else:
                                logger.warning(
                                    "Malformed cached_tokens: %s", request_cached_tokens
                                )
                        else:
                            logger.warning(
                                "Malformed input_token_details: %s", input_token_details
                            )

                        if isinstance(output_token_details, dict):
                            if isinstance(
                                request_reasoning_tokens := output_token_details.get(
                                    "reasoning", 0
                                ),
                                int,
                            ):
                                reasoning_tokens += request_reasoning_tokens
                            else:
                                logger.warning(
                                    "Malformed reasoning tokens: %s",
                                    request_reasoning_tokens,
                                )
                        else:
                            logger.warning(
                                "Malformed output_token_details: %s",
                                output_token_details,
                            )

            except openai.BadRequestError as e:
                # Code 400 I think, this means the user config is invalid.
                logger.error("Bad request error: %s", e)
                break
            except BadRequestResponseError as e:
                logger.error(
                    "Openrouter status=%s message=%s body=%s",
                    e.status_code,
                    e.message,
                    e.body,
                )
                break
            except Exception as e:
                logger.exception("Stream error: %s", e)
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

            logger.info("Total cost: %s", cost)
            logger.info("Total input tokens: %s", input_tokens)
            logger.info("Total output tokens: %s", output_tokens)
            logger.info("Total cached tokens: %s", cached_tokens)
            logger.info("Total reasoning tokens: %s", reasoning_tokens)
            # to avoid zero division error
            logger.info(
                f"Cache hit: {cached_tokens / (input_tokens or 1) * 100.0:.2f}%"
            )

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

        if (
            not manager.get_config().dynamic_context.keep_context
            and dyn_ctx is not None
        ):
            for block in human_message.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    block["text"] = block["text"].replace(dyn_ctx, "").strip()

        self.history.save()

    async def _process_llm_stream(
        self, tools: dict[str, LLMTool]
    ) -> Optional[AIMessage]:
        full_response: Optional[AIMessage] = None
        cfg = manager.get_config()
        model = cfg.get_model(cfg.llm.models.main)
        self.llm = get_backend(model)
        self.data.current_turn = None
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
                                    # why am I not saving it to an easily accessible place?
                                    out_msg.reasoning = part.get("thinking")
                                case "reasoning":
                                    # content=[{'summary': [{'index': 0, 'type': 'summary_text', 'text': '**Answering philosophical questions**\n\nI'}], 'index': 0, 'type': 'reasoning'}]
                                    summary_list: list[dict[str, int | str]] | None = (
                                        part.get("summary")
                                    )
                                    if isinstance(summary_list, list):
                                        for item in summary_list:
                                            if out_msg.reasoning is None:
                                                out_msg.reasoning = item.get("text")
                                            else:
                                                out_msg.reasoning += item.get(
                                                    "text", ""
                                                )
                                            if (
                                                "reasoning_content"
                                                in full_response.additional_kwargs
                                                and isinstance(
                                                    full_response.additional_kwargs[
                                                        "reasoning_content"
                                                    ],
                                                    str,
                                                )
                                            ):
                                                full_response.additional_kwargs[
                                                    "reasoning_content"
                                                ] += item.get("text", "")
                                            else:
                                                full_response.additional_kwargs[
                                                    "reasoning_content"
                                                ] = item.get("text")
                                case "function_call":
                                    ...
                                    # content=[{'type': 'function_call', 'name': 'browser_close_tab', 'arguments': '{"tab_index', 'call_id': 'call_zsjXgqfMEkupjtc3jp1Gui2L', 'id': 'fc_tmp_xgub93viza', 'index': 0}] additional_kwargs={} response_metadata={'model_provider': 'openai', 'id': 'gen-1783107070-5ZayIrvuS5VGXEu1yQXu'} id='gen-1783107070-5ZayIrvuS5VGXEu1yQXu' tool_calls=[{'name': 'browser_close_tab', 'args': {}, 'id': 'call_zsjXgqfMEkupjtc3jp1Gui2L', 'type': 'tool_call'}] invalid_tool_calls=[] tool_call_chunks=[{'name': 'browser_close_tab', 'args': '{"tab_index', 'id': 'call_zsjXgqfMEkupjtc3jp1Gui2L', 'index': 0, 'type': 'tool_call_chunk'}]
                                case None:
                                    logger.warning(
                                        "Type not found in answer dict part: %s", part
                                    )
                                case _:  # Handle other instances if they come
                                    logger.error(
                                        "Unhandled dict type: %s", part.get("type")
                                    )
                                    logger.error("Dict content: %s", part)
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

        if full_response:
            logger.info(
                "Usage metadata: %s",
                json.dumps(full_response.usage_metadata, ensure_ascii=False, indent=2),
            )
            logger.info(
                "Response metadata: %s",
                json.dumps(
                    full_response.response_metadata, ensure_ascii=False, indent=2
                ),
            )

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

        results: list[ToolMessage] = []

        for tool_call in tool_calls:
            await socket_out.send(
                OutputMessage(
                    tool_call_id=tool_call["id"],
                    tool_name=tool_call["name"],
                    tool_args=tool_call["args"],
                ).to_bus(MessageTopic.TOOL_CALL)
            )

            name = tool_call["name"]
            args = tool_call["args"]
            call_id = tool_call["id"]

            logger.info("Calling tool: %s", name)
            logger.debug("Tool call id: %s", call_id)
            logger.debug(
                "Tool arguments: %s", json.dumps(args, indent=2, ensure_ascii=False)
            )

            @tool(name)
            def inexistent(
                *args: Any,
                _name: str = name,
                _args: dict[str, Any] = args,
                **kwargs: Any,
            ) -> str:
                """Inexistent tool placeholder."""
                logger.warning("LLM called tool %s with args %s", _name, _args)
                return f"You tried to call an inexistent tool '{_name}'."

            func = tools.get(name, inexistent)

            result = await self._handle_single_tool(func, args, call_id)
            results.append(result)

            # Before there were mulithreading, but for now I'll disable it because it doesn't bring that many benefits compared to the disadvantages it gives with model hallucination.
            # results = await asyncio.gather(*tasks)

            # user message injection here
            # tool context is ran after the tool for a reason.
            self.data.current_turn = ToolMessage
            context = await get_dynamic_context(self.data)
            try:
                # stealing the messages from the message queue muhahahahahahaa
                # only quirk is that the prompts aren't getting updated based on the input message at least for now.
                user_message = self.message_queue.get_nowait()
                if user_message.id in self.cancelled_messages:
                    logger.info(
                        "Message '%s' was cancelled, not sending.", user_message.id
                    )
                    logger.debug("Full message: %s", user_message)
                    self.cancelled_messages.remove(user_message.id)
                    # too lazy to catch invalid sate
                    self.messages_in_queue.remove(user_message.id)
                    user_message = None

            except asyncio.QueueEmpty:
                user_message = None

            content = ""
            if isinstance(result.content, str):
                if context:
                    result.content += "\n\n" + context
                if user_message is not None:
                    result.content += "\n\nUser:\n" + user_message.build()

                content = result.content
            elif isinstance(result.content, list):
                if all(isinstance(block, str) for block in result.content):
                    if context:
                        result.content.append("\n\n" + context)

                    if user_message is not None:
                        assert isinstance(result.content[-1], str)
                        result.content[-1] += (
                            "\n\nUser (you should acknowledge this immediately):\n"
                            + user_message.build()
                        )
                else:  # either when all are dicts or all aren't dicts
                    if context:
                        result.content.append({"type": "text", "text": context})

                    if user_message is not None:
                        assert isinstance(result.content[-1], dict)
                        result.content.append(
                            {"type": "text", "text": "User:\n" + user_message.build()}
                        )

                for block in result.content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            content += block.get("text", "")
                        else:
                            logger.debug(
                                "Tools has %s content type.", block.get("type")
                            )
                    elif isinstance(block, str):
                        content += block
            logger.debug("Tool content: %s", result.content)
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

            # Add support for the tools to give out raw tool messages
            if isinstance(output, (ToolMessage)):
                return output
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
            case Commands.CANCEL_MESSAGE:
                if not command.args:
                    logger.error(
                        "Malformed command. CANCEL_MESSAGE needs message id as the first argument."
                    )
                id = command.args[0]
                logger.info("Cancelling message with id '%s'", id)
                if id not in self.messages_in_queue:
                    logger.warning(
                        "Message with the id of '%s' cannot be cancelled.", id
                    )
                    return

                self.cancelled_messages.append(id)
            case _:
                logger.warning("Invalid command: %s", command.command)


# global reference to the main vector instance
# this is assigned by the main function.
vector: Vector
