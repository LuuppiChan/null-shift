from dataclasses import dataclass, field
from datetime import datetime
import json
import asyncio
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Literal, Optional
from copy import deepcopy

import flet as ft
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
    messages_from_dict,
)
import zmq.asyncio

from global_tools import Signal
from global_types import BusMessage, Commands, Difficulty, InputMessage, MessageTopic
from gui.stt import Transcriber
from gui.tts import TextToSpeech
from output_message import OutputMessage
from gui.config import manager

logger = logging.getLogger(__name__)


def open_in_default_editor(file_path: str):
    """Open a file in the system's default text editor."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if sys.platform.startswith("win"):
        os.startfile(file_path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", file_path])
    else:
        # Linux/Unix: Try EDITOR env var, then xdg-open
        # editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
        # if editor:
        #     subprocess.Popen([editor, file_path])
        # else:
        subprocess.Popen(["xdg-open", file_path])


def good_markdown(value: str = "", visible: bool = True) -> ft.Markdown:
    return ft.Markdown(
        value,
        soft_line_break=True,
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        code_theme=ft.MarkdownCodeTheme.ATOM_ONE_DARK,
        md_style_sheet=ft.MarkdownStyleSheet(
            blockquote_decoration=ft.BoxDecoration(ft.Colors.GREY_500)
        ),
        visible=visible,
    )


@dataclass
class ConfigValue:
    value: str | bool | list[str]
    selected: str | bool | str | None = None
    title: str = ""
    description: str = ""
    key: str = ""


@ft.control
class AutoConfig(ft.Column):
    """Create a settings menu based on a config dict."""

    config: dict[str, ConfigValue] = field(default_factory=dict)
    title: str = ""

    def init(self):
        self.build_settings()
        self.scroll = ft.ScrollMode.AUTO

    def build_settings(self):
        if self.title:
            self.controls.append(ft.Text(self.title))

        for k, v in self.config.items():
            v.key = k
            match v.value:
                case str():
                    self.controls.append(
                        ft.TextField(
                            v.value,
                            data=v,
                            on_change=self.on_update,
                            tooltip=v.description,
                            hint_text=v.title,
                            multiline=True,
                        )
                    )
                case bool():
                    self.controls.append(
                        ft.Checkbox(
                            v.title,
                            v.value,
                            data=v,
                            on_change=self.on_update,
                            tooltip=v.description,
                        )
                    )
                case list():
                    self.controls.append(
                        ft.Dropdown(
                            options=[ft.DropdownOption(opt) for opt in v.value],
                            label=v.title,
                            on_select=self.on_update,
                            data=v,
                            tooltip=v.description,
                        )
                    )

    def on_update(self, e: ft.Event):
        assert isinstance(e.control, (ft.TextField, ft.Checkbox, ft.Dropdown))
        k: ConfigValue = e.control.data

        match e.control:
            case ft.TextField(value=value):
                k.selected = value
            case ft.Checkbox(value=value):
                k.selected = value
            case ft.Dropdown(value=value):
                k.selected = value


@ft.control
class Input(ft.Container):
    def init(self):
        # bug: no scrolling currently.
        self.input = ft.TextField(
            multiline=True,
            hint_text="Send a Message",
            border=ft.InputBorder.NONE,
            expand=True,
            on_submit=self.send,
            on_change=self.text_changed,
            shift_enter=True,
        )
        self.shape = ft.BoxShape.RECTANGLE
        self.border_radius = 16
        self.bgcolor = ft.Colors.BLACK_54
        self.padding = ft.Padding(16, 12, 16, 12)

        self.voice_text = "Start voice input"
        self.mic = ft.IconButton(
            ft.Icons.MIC,
            on_click=self.on_voice_button,
            tooltip=self.voice_text,
            on_long_press=self.send_long_press,
        )
        self.mute = ft.IconButton(
            icon=ft.Icons.MIC,
            on_click=self.mute_toggle,
            tooltip="Mute mic",
            visible=False,
        )
        self.muted = False
        self.transcriber = Transcriber()
        self.transcriber.on_input.connect(self.on_voice)

        # todo: attach files
        # file selecting, pasting, dragging
        self.file_button = ft.IconButton(
            ft.Icons.ADD, tooltip="Attach files", on_click=self.open_file_picker
        )
        self.file_preview = ft.Row(visible=False)

        self.row = ft.Row(
            [
                self.file_button,
                self.input,
                self.mute,
                self.mic,
            ]
        )
        self.column = ft.Column([self.file_preview, self.row])
        self.content = self.column
        self.on_send: Signal[str, None] = Signal(str)
        self.chat: Chat

    def mute_toggle(self):
        self.muted = not self.muted
        self.transcriber.mic_listener.muted = self.muted
        if self.muted:
            self.mute.icon = ft.Icons.MIC_OFF
            self.mute.tooltip = "Unmute mic"
        else:
            self.mute.icon = ft.Icons.MIC
            self.mute.tooltip = "Mute mic"

    async def open_file_picker(self):
        file_picker = ft.FilePicker()
        files = await file_picker.pick_files("Attach Media")
        for file in files:
            if file.path is None:
                logger.error("Cannot pick file, path is None: %s", file)
                continue

            path = Path(file.path)
            # self.file_preview.controls.append()

    async def send_long_press(self):
        text = self.input.value
        if text:
            await self.transcriber.start()
            await self.chat.tts.start_stream()
            self.mute.visible = True
            self.text_changed()
            self.update()

    async def on_voice_button(self):
        text = self.input.value
        if self.transcriber.running:
            self.mute.visible = False
            await self.transcriber.stop()
        elif text:
            self.send()
        else:
            self.mute.visible = True
            await self.transcriber.start()
            await self.chat.tts.start_stream()
        self.text_changed()

    def text_changed(self):
        text = self.input.value
        if self.transcriber.running:
            self.mic.icon = ft.Icons.STOP
            self.mic.tooltip = "Stop voice input"
        elif text:
            self.mic.icon = ft.Icons.SEND
            self.mic.tooltip = "Send message"
        else:
            self.mic.tooltip = self.voice_text
            self.mic.icon = ft.Icons.MIC

    async def on_voice(self, text: str):
        logger.debug("Got voice on gui: %s", text)
        # check for commands
        cmd = text.lower()
        if re.search(r"(?:slash|\/)\s?sen(?:d|t)\s?(?:message)?", cmd):
            self.send()
            return
        elif re.search(r"(?:slash|\/)\s?abort", cmd):
            await self.chat.on_send("/a")
            cfg = manager.get_config()
            if cfg.speak.audio_feedback:
                await self.chat.tts.abort()
                await self.chat.tts.speak_single("Aborted.")
            return
        elif re.search(r"(?:slash|\/)\s?clear", cmd):
            self.input.value = ""
            self.update()
            cfg = manager.get_config()
            if cfg.speak.audio_feedback:
                await self.chat.tts.abort()
                await self.chat.tts.speak_single("Cleared.")
            return
        elif re.search(r"(?:slash|\/)\s?current\s?(?:text|message)?", cmd):
            await self.chat.tts.abort()
            await self.chat.tts.speak_single("Current message: " + self.input.value)
            return

        self.input.value += "\n" + text if self.input.value else text
        self.update()
        cfg = manager.get_config()
        text = self.input.value.lower()
        # the check is more readable this way
        contains_wake_word = False
        if not cfg.voice.wake_word:
            contains_wake_word = True
        elif isinstance(cfg.voice.wake_word, str):
            contains_wake_word = re.search(cfg.voice.wake_word, text) is not None
        elif isinstance(cfg.voice.wake_word, list):
            for wake_word in cfg.voice.wake_word:
                match = re.search(wake_word, text)
                if match:
                    contains_wake_word = True
                    break

        if isinstance(cfg.voice.wake_word, str) and cfg.voice.wake_word.lower() in text:
            contains_wake_word = True
        elif isinstance(cfg.voice.wake_word, list) and any(
            [word.lower() in text for word in cfg.voice.wake_word]
        ):
            contains_wake_word = True

        if cfg.auto_send_voice and contains_wake_word:
            self.send()

    def send(self):
        text = self.input.value
        if text:
            logger.info("Submitted text: %s", text)
            self.on_send.emit(text)


@ft.control
class Message(ft.Container):
    def init(self):
        cfg = manager.get_config()
        self.text = good_markdown(visible=False)
        self.padding = 16
        self.margin = 4
        self.shape = ft.BoxShape.RECTANGLE
        self.border_radius = 16
        self.bgcolor = ft.Colors.BLACK_54
        self.avatar = ft.Markdown("**Orion**", visible=False)
        self.loading = ft.Column([ft.ProgressRing()])
        self.space = ft.Divider(
            height=cfg.end_space_height, opacity=0, visible=cfg.end_space
        )
        self.is_ai: bool = False
        self.is_tool: bool = False
        self.tool_name: str = ""
        self.tool_id: str = ""
        self.thought_markdown = good_markdown(visible=False)
        self.tool_call = good_markdown(visible=False)
        self.tool_result: ft.Text | ft.Markdown = ft.Text(visible=False)
        self.thoughts = ft.ExpansionTile(
            "Reasoning",
            [
                thought_column := ft.Column(
                    [self.thought_markdown, self.tool_call, self.tool_result],
                    scroll=ft.ScrollMode.AUTO,
                    expand=True,
                )
            ],
            visible=False,
            on_change=self.change_size_limit,
            expanded=cfg.default_collapse_state,
        )
        self.thought_column = thought_column

        self.action_bar = ft.Row(
            [
                ft.IconButton(
                    ft.Icons.COPY,
                    tooltip="Copy to clipboard",
                    on_click=self.copy_to_clipboard,
                    icon_size=16,
                    margin=0,
                    padding=2,
                ),
                copy_thoughts := ft.IconButton(
                    ft.Icons.COPY_ALL,
                    tooltip="Copy reasoning to clipboard",
                    on_click=self.copy_thoughts_to_clipboard,
                    icon_size=16,
                    margin=0,
                    padding=2,
                    visible=False,
                ),
                speak := ft.IconButton(
                    ft.Icons.VOLUME_UP,
                    tooltip="Speak aloud",
                    icon_size=16,
                    margin=0,
                    padding=2,
                    on_click=self.speak_aloud,
                ),
                ft.IconButton(
                    ft.Icons.VISIBILITY_OFF,
                    tooltip="Hide message",
                    icon_size=16,
                    margin=0,
                    padding=2,
                    on_click=self.delete_message,
                ),
            ],
            visible=False,
            tight=True,
        )
        self.speak_button = speak
        self.speak_task: asyncio.Task | None = None
        self.copy_thoughts = copy_thoughts

        self.content = ft.Column(
            [
                self.avatar,
                self.thoughts,
                self.text,
                self.action_bar,
                self.loading,
                self.space,
            ]
        )
        self.on_hover = self.hover_event
        self.chat: Chat | None = None

    def delete_message(self):
        if self.chat is None:
            logger.error(
                "Cannot delete message. Hiding instead. Chat is None on message: %s",
                self.text.value,
            )
            self.visible = False
            self.update()
            return
        idx = self.chat.messages.controls.index(self)
        if idx == -1:
            logger.error(
                "Cannot delete message. Hiding instead. Message not found: %s",
                self.text.value,
            )
            self.visible = False
            self.update()
            return

        self.chat.messages.controls.pop(idx)
        logger.info("Removed message: %s", self.text.value)
        self.chat.messages.update()

    async def speak_aloud(self):
        if self.chat is None:
            logger.error(
                "Cannot speak aloud. Chat reference is None on message: %s",
                self.text.value,
            )
            return

        if self.speak_task is not None and not self.speak_task.done():
            self.speak_button.icon = ft.Icons.VOLUME_UP
            await self.chat.tts.abort()
            self.speak_task.cancel()
            self.update()
            return

        def done(_: asyncio.Task):
            self.speak_task = None
            self.speak_button.icon = ft.Icons.VOLUME_UP
            self.update()

        self.speak_button.icon = ft.Icons.STOP
        self.speak_task = asyncio.create_task(
            self.chat.tts.speak_single(self.text.value)
        )
        self.speak_task.add_done_callback(done)
        self.update()

    def hover_event(self, e: ft.Event[ft.Container]):
        # is hovering
        if e.data:
            self.action_bar.visible = True
        else:
            self.action_bar.visible = False
        self.update()

    async def copy_to_clipboard(self):
        await ft.Clipboard().set(self.text.value)
        self.page.show_dialog(ft.SnackBar("Copied answer to clipboard."))

    async def copy_thoughts_to_clipboard(self):
        await ft.Clipboard().set(self.thought_markdown.value)
        self.page.show_dialog(ft.SnackBar("Copied reasoning to clipboard."))

    def change_size_limit(self):
        if not self.is_tool:
            return
        if self.thoughts.height is None:
            self.thought_column.height = 500
        else:
            self.thought_column.height = None

    def append_text(self, text: str = "", thoughts: str = ""):
        if text:
            self.text.value += text
            self.loading.visible = False
            self.text.visible = True
        if thoughts:
            if not self.thoughts.visible:
                if self.chat is not None:
                    cfg = manager.get_config()
                    if cfg.speak.audio_feedback and self.chat.speaking_enabled():
                        self.chat.tts.put_stream("Reasoning. ")
                else:
                    logger.error(
                        "Cannot speak aloud, chat is None on message: %s",
                        self.text.value,
                    )
            self.thought_markdown.value += thoughts
            self.thought_markdown.visible = True
            cfg = manager.get_config()
            self.thoughts.tooltip = self.thought_markdown.value[-cfg.tooltip_len :]
            self.thoughts.visible = True
            self.copy_thoughts.visible = True

    @staticmethod
    def ai(
        text: str = "", thoughts: str = "", chat: Optional["Chat"] = None
    ) -> "Message":
        msg = Message()
        msg.chat = chat
        msg.append_text(text, thoughts)
        msg.align = ft.Alignment.CENTER_LEFT
        msg.bgcolor = None
        msg.avatar.visible = True
        msg.is_ai = True
        return msg

    @staticmethod
    def user(text: str = "") -> "Message":
        msg = Message()
        msg.append_text(text)
        msg.align = ft.Alignment.CENTER_RIGHT
        msg.space.visible = False
        return msg

    @staticmethod
    def tool(name: str, id: str, args: dict[str, Any]) -> "Message":
        cfg = manager.get_config()
        msg = Message()
        msg.thoughts.title = f"Tool call: `{name}`"
        msg.thoughts.tooltip = (
            f"Arguments:\njson\n{json.dumps(args, indent=4)[-cfg.tooltip_len :]}\n"
        )
        msg.thoughts.visible = True
        msg.tool_call.value = f"**Tool Name:** {name}\n\n**Arguments:**\n```json\n{json.dumps(args, indent=2)}\n```"
        msg.tool_call.visible = True
        msg.tool_name = name
        msg.tool_id = id
        msg.align = ft.Alignment.CENTER_LEFT
        msg.bgcolor = ft.Colors.BLACK_12
        msg.is_tool = True
        msg.loading.visible = False
        msg.space.visible = False
        return msg

    def add_tool_response(
        self, content: str | dict[str, Any] | list[dict[str, Any]] | list[str]
    ):
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2)
            content = f"```json\n{content}\n```"
            self.tool_result = good_markdown()

        self.tool_result.value = "Tool returned:\n\n" + content
        self.tool_result.visible = True


@ft.control
class Chat(ft.Container):
    def init(self):
        self.input = Input()
        self.input.chat = self
        self.input.on_send.connect(self.on_send)

        self.settings_reference = {
            "title": "Input message options.\nDon't touch if you don't know what you're doing.",
            "config": {
                "type": ConfigValue(
                    value=["", "instant", "batched"],
                    title="Message Type",
                    description="Type of the message.",
                ),
                "title": ConfigValue(
                    value="",
                    title="Message title",
                    description="Title for the message, usually used on batched messages.",
                ),
                "difficulty": ConfigValue(
                    value=["", *list(Difficulty)],
                    title="Difficulty",
                    description="Difficulty of the task",
                ),
                "goal": ConfigValue(
                    value="", title="Goal", description="Goal of the given task."
                ),
                "context": ConfigValue(
                    value="", title="Context", description="Context about the task."
                ),
            },
        }
        self.settings = AutoConfig(**deepcopy(self.settings_reference))
        self.preset_map: dict[str, AutoConfig] = {}

        def select_preset(e: ft.Event[ft.Dropdown]):
            assert isinstance(e.data, str), type(e.data)
            self.settings = self.preset_map.get(
                e.data, AutoConfig(**deepcopy(self.settings_reference))
            )

        def save_preset():
            if preset_name.value:
                self.preset_map[preset_name.value] = deepcopy(self.settings)
                presets.options.append(ft.DropdownOption(preset_name.value))
                preset_name.value = ""
                self.update()

        def delete_preset():
            if presets.value in self.preset_map:
                del self.preset_map[preset_name.value]
                preset_name.value = ""
                for i, preset in enumerate(presets.options):
                    if preset.text == presets.value:
                        presets.options.pop(i)
                        break
                self.settings = self.preset_map.get(
                    "", AutoConfig(**deepcopy(self.settings_reference))
                )
                self.update()
            else:
                logger.warning("Preset %s is not in preset map.", presets.value)

        self.settings_menu = ft.Column(
            [
                presets := ft.Dropdown(
                    "",
                    [ft.DropdownOption("")],
                    tooltip="Presets",
                    on_select=select_preset,
                    visible=False,
                ),
                preset_name := ft.TextField(hint_text="Preset name", visible=False),
                ft.Row(
                    [
                        ft.Button("Save", on_click=save_preset),
                        ft.Button("Delete", on_click=delete_preset),
                    ],
                    visible=False,
                ),
                ft.Divider(visible=False),
                self.settings,
            ],
            visible=False,
        )

        def toggle_visible():
            self.settings_menu.visible = not self.settings_menu.visible
            if self.settings_menu.visible:
                self.width = 1280
            else:
                self.width = 1024

        self.open_settings = ft.IconButton(
            ft.Icons.MENU,
            on_click=toggle_visible,
            align=ft.Alignment.TOP_RIGHT,
            padding=16,
            margin=8,
        )

        self.messages: ft.ListView = ft.ListView(
            controls=[],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            expand_loose=True,
            auto_scroll=True,
        )
        self.column = ft.Column(
            [self.messages, ft.Column([self.input])],
            expand=True,
        )
        self.row = ft.Row([self.column, self.settings_menu], expand=True)
        self.stack = ft.Stack([self.row])
        self.content = ft.SelectionArea(self.stack)
        self.align = ft.Alignment.CENTER
        self.width = 1024
        self.expand = True

        self.finished: bool = True
        self.was_aborted: bool = False
        self.streaming: bool = False
        self.tts = TextToSpeech(self.input.transcriber.mic_listener.speech_event)

        self.listening = asyncio.create_task(self.listen())
        self.config_poller_task = asyncio.create_task(self.config_poller())
        self.load_core_history()

    async def config_poller(self):
        while True:
            await asyncio.sleep(1)
            manager.get_config()

    async def on_send(self, text: str):
        if text.startswith("/"):
            parts = text.split(" ", 1)
            if len(parts) == 1:
                command = parts[0]
                arg = ""
            else:
                command, arg = parts

            match command:
                case "/abort" | "/a":
                    await sock_in.send_multipart(
                        BusMessage(
                            topic=MessageTopic.COMMAND,
                            payload={"command": Commands.ABORT},
                        ).encoded()
                    )
                case "/h" | "/history":
                    match arg:
                        case "mv":
                            Path(
                                "~/.null-shift/brain/history.json"
                            ).expanduser().rename(
                                Path(
                                    f"~/.null-shift/brain/history {datetime.now().strftime('%d-%m-%y %H-%M-%S')}.json"
                                ).expanduser()
                            )
                            self.add_message(Message.user("History moved."))
                        case "rm":
                            Path(
                                "~/.null-shift/brain/history.json"
                            ).expanduser().unlink(missing_ok=True)
                            self.add_message(Message.user("History deleted."))
                        case "c":
                            self.messages.controls.clear()
                        case "crm" | "rmc" | "rm c" | "c rm":
                            Path(
                                "~/.null-shift/brain/history.json"
                            ).expanduser().unlink(missing_ok=True)
                            self.messages.controls.clear()
                            self.page.show_dialog(
                                ft.SnackBar("History removed and cleared.")
                            )
                        case _:
                            self.add_message(Message.user(f"Unknown argument '{arg}'"))
                case _:
                    self.add_message(Message.user(f"Unknown command: '{command}'"))
            self.input.input.value = ""
            self.input.text_changed()
            self.update()
            return

        if self.finished:
            cfg = manager.get_config()
            self.finished = False
            self.input.input.value = ""
            self.input.text_changed()
            self.add_message(Message.user(text))
            c = self.settings.config
            options = {k: v.selected for k, v in c.items() if v.selected}
            options["body"] = text
            context = str(options.get("context", ""))

            if cfg.voice.context_injection and self.input.transcriber.running:
                context += "\n\n" + cfg.voice.context_injection
            if cfg.speak.context_injection and (
                (cfg.speak.with_stt and self.input.transcriber.running)
                or (cfg.speak.always)
            ):
                context += "\n\n" + cfg.speak.context_injection

            options["context"] = context.strip()
            if not options["context"]:
                del options["context"]

            logger.info("%s", json.dumps(options, indent=2))

            if self.streaming and self.input.transcriber.running:
                await sock_in.send_multipart(
                    BusMessage(
                        topic=MessageTopic.COMMAND,
                        payload={"command": Commands.ABORT},
                    ).encoded()
                )
            self.update()
            await sock_in.send_multipart(
                InputMessage.model_validate(options).to_bus().encoded()
            )
            cfg = manager.get_config()
            if cfg.speak.audio_feedback and self.speaking_enabled():
                self.tts.put_stream("Message sent. ")

    def speaking_enabled(self) -> bool:
        cfg = manager.get_config()
        return cfg.speak.always or (
            cfg.speak.with_stt and self.input.transcriber.running
        )

    async def listen(self):
        """Listen sock_out"""
        try:
            while True:
                try:
                    frames = await sock_out.recv_multipart()
                    msg = BusMessage.decoded(frames)
                    if msg is None:
                        logger.error("Error reading message from core. %s", frames)
                        continue

                    match msg.topic:
                        case MessageTopic.STREAM:
                            if self.is_last_tool():
                                self.add_message(Message.ai(chat=self))

                            out = OutputMessage.from_bus(msg)
                            if out is None:
                                logger.error("Error validating output message.")
                                continue

                            self.append_last(out.text or "", out.reasoning or "")
                            if out.text:
                                if self.speaking_enabled():
                                    self.tts.put_stream(out.text)

                        case MessageTopic.TOOL_CALL:
                            out = OutputMessage.from_bus(msg)
                            if out is None:
                                logger.error("Error validating output message.")
                                continue
                            self.append_text().loading.visible = False
                            self.add_message(
                                Message.tool(
                                    out.tool_name or "",
                                    out.tool_call_id or "",
                                    out.tool_args or {},
                                )
                            )
                            cfg = manager.get_config()
                            if cfg.speak.audio_feedback:
                                self.tts.put_stream(
                                    f"Tool call {str(out.tool_name).replace('_', ' ')}. "
                                )
                        case MessageTopic.TOOL_RESULT:
                            out = OutputMessage.from_bus(msg)
                            if out is None:
                                logger.error("Error validating output message.")
                                continue

                            tool = self.append_last(
                                type="tool", tool_id=out.tool_call_id or ""
                            )
                            if tool is None:
                                tool = Message.tool(
                                    out.tool_name or "", out.tool_call_id or "", {}
                                )
                            tool.add_tool_response(
                                out.tool_result or "(No return value)"
                            )
                            # cfg = manager.get_config()
                            # if cfg.speak.audio_feedback:
                            #     self.tts.put_stream("Tool result. ")
                        case MessageTopic.STARTED:
                            self.was_aborted = False
                            self.add_message(Message.ai(chat=self))
                            if self.speaking_enabled():
                                await self.tts.start_stream()
                                cfg = manager.get_config()
                                if cfg.speak.audio_feedback:
                                    self.tts.put_stream("Stream started. ")
                        case MessageTopic.FINISHED:
                            self.finished = True
                            self.append_text().loading.visible = False
                            last = self.append_last()
                            if last and last.text.value:
                                cfg = manager.get_config()
                                if cfg.speak.audio_feedback and self.speaking_enabled():
                                    self.tts.put_stream(" Stream ended. ")
                            await self.tts.stop_stream()
                            if not self.was_aborted:
                                self.load_core_history()
                        case MessageTopic.ABORT:
                            self.finished = True
                            self.was_aborted = True
                            ai = self.append_last()
                            if ai:
                                ai.loading.visible = False
                            self.add_message(Message.user("Abort"))
                            await self.tts.stop_stream()
                            await self.tts.abort()
                            cfg = manager.get_config()
                            if cfg.speak.audio_feedback and self.speaking_enabled():
                                await self.tts.speak_single("Stream aborted.")
                    self.update()
                except Exception as e:
                    logger.error("Error while listening: %s", e, exc_info=True)
        except asyncio.CancelledError:
            logger.info("Shutting down listener.")

    def add_message(self, message: Message):
        message.chat = self
        if self.messages.controls:
            assert isinstance(self.messages.controls[-1], Message)
            self.messages.controls[-1].space.visible = False
        self.messages.controls.append(message)

    def append_text(self, text: str = "", thoughts: str = "", idx: int = -1) -> Message:
        """Append text to last message or specified index."""
        if not self.messages.controls:
            logger.error("Appending to an empty message list!!!")

        logger.debug("Appending text to %s: %s", idx, text or thoughts)
        msg = self.messages.controls[idx]
        assert isinstance(msg, Message)
        msg.append_text(text, thoughts)
        return msg

    def append_last(
        self,
        text: str = "",
        thoughts: str = "",
        type: Literal["ai", "tool"] = "ai",
        tool_id: str = "",
    ) -> Message | None:
        """Append to last some message."""
        # appending fails if same tool name on multiple tools
        for i in range(len(self.messages.controls) - 1, -1, -1):
            msg = self.messages.controls[i]

            assert isinstance(msg, Message)
            match type:
                case "ai":
                    if msg.is_ai:
                        return self.append_text(text, thoughts, i)
                case "tool":
                    if msg.is_tool and msg.tool_id == tool_id:
                        return self.append_text(text, thoughts, i)
        return None

    def is_last_tool(self) -> bool:
        if not self.messages.controls:
            return False
        assert isinstance(self.messages.controls[-1], Message)
        return self.messages.controls[-1].is_tool

    # I think this should be the main history generator.
    def load_core_history(self):
        cfg = manager.get_config()
        if not cfg.core_history:
            return

        history_path = Path(cfg.core_history).expanduser().resolve()
        if not history_path.is_file():
            return

        try:
            self.messages.controls.clear()
            history_dict: list[dict[str, Any]] = json.loads(history_path.read_text())
            history = messages_from_dict(history_dict)
            for msg in history:
                if isinstance(msg, HumanMessage):
                    text: str = ""
                    if isinstance(msg.content, list):
                        for item in msg.content:
                            if isinstance(item, str):
                                text += item
                            else:
                                text += str(item.get("text"))
                    else:
                        text += msg.content
                    self.add_message(Message.user(text))
                elif isinstance(msg, (AIMessageChunk, AIMessage)):
                    thoughts = ""
                    text = ""
                    if isinstance(msg.content, str):
                        text = msg.content
                    elif isinstance(msg.content, list):
                        for chunk in msg.content:
                            if isinstance(chunk, str):
                                text += chunk
                            elif isinstance(chunk, dict):
                                text += chunk.get("text", "")
                                thoughts += chunk.get(
                                    "thoughts",
                                    chunk.get("think", chunk.get("reasoning", "")),
                                )
                    thoughts += str(
                        msg.additional_kwargs.get(
                            "reasoning",
                            msg.additional_kwargs.get("reasoning_content", ""),
                        )
                    )
                    gui_msg = Message.ai(text, thoughts, chat=self)
                    gui_msg.loading.visible = False
                    self.add_message(gui_msg)
                    for tool_call in msg.tool_calls:
                        self.add_message(
                            Message.tool(
                                tool_call["name"],
                                str(tool_call["id"]),
                                tool_call["args"],
                            )
                        )
                elif isinstance(msg, (ToolMessage)):
                    tool = self.append_last(type="tool", tool_id=msg.tool_call_id or "")
                    if tool is None:
                        tool = Message.tool(msg.name or "", msg.tool_call_id or "", {})
                    tool.add_tool_response(str(msg.content) or "(No return value)")
        except Exception as e:
            logger.error("Couldn't load history: %s", e, exc_info=True, stack_info=True)


async def main(page: ft.Page):
    global ctx, sock_in, sock_out
    ctx = zmq.asyncio.Context()
    sock_in = ctx.socket(zmq.PUSH)
    sock_out = ctx.socket(zmq.SUB)
    sock_in.connect("tcp://localhost:5555")
    sock_out.connect("tcp://localhost:5556")
    sock_out.subscribe(b"")
    sock_out.setsockopt(zmq.RCVHWM, 10_000)

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # too_verbose = ["flet_transport", "flet_controls", "flet", "flet_desktop"]
    # for log in too_verbose:
    #     logging.getLogger(log).setLevel("INFO")

    async def on_close(e: ft.WindowEvent):
        if e.type == ft.WindowEventType.CLOSE:
            logger.info("Closing...")
            sock_in.close()
            sock_out.close()
            ctx.term()
            chat.listening.cancel()
            await page.window.destroy()

    page.window.prevent_close = True
    page.title = "Null Shift"
    page.window.on_event = on_close
    page.window.frameless = True
    page.theme_mode = ft.ThemeMode.DARK
    # page.scroll = ft.ScrollMode.AUTO

    # no worky right now
    # page.window.bgcolor = ft.Colors.TRANSPARENT
    # page.bgcolor = ft.Colors.TRANSPARENT

    chat = Chat()
    page.add(chat)
    page.overlay.append(chat.open_settings)
    page.overlay.append(
        ft.IconButton(
            ft.Icons.SETTINGS,
            tooltip="Open GUI settings",
            on_click=lambda: open_in_default_editor(str(manager.path)),
            align=ft.Alignment.TOP_LEFT,
            padding=16,
            margin=8,
        )
    )


ctx: zmq.asyncio.Context
sock_in: zmq.asyncio.Socket
sock_out: zmq.asyncio.Socket


ft.run(main)
