from dataclasses import dataclass, field
from datetime import datetime
import json
import asyncio
import logging
from pathlib import Path
from typing import Any, Literal

import flet as ft
import zmq.asyncio

from global_tools import Signal
from global_types import BusMessage, Commands, Difficulty, InputMessage, MessageTopic
from output_message import OutputMessage
from gui.config import manager

logger = logging.getLogger(__name__)

AI_EXAMPLE = """Here is a complete, modern approach to creating a sticky header. 

Today, the actual "sticking" behavior is best done using pure **CSS** (`position: sticky`). However, **JavaScript** is often used to add visual effects when the user scrolls (like shrinking the header, adding a shadow, or slightly changing the background color). 

Here is the complete HTML, CSS, and JavaScript to achieve both.

### 1. The HTML
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sticky Header</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>

    <!-- The Header -->
    <header id="site-header">
        <div class="logo">MyBrand</div>
        <nav>
            <a href="#">Home</a>
            <a href="#">About</a>
            <a href="#">Contact</a>
        </nav>
    </header>

    <!-- Dummy content to allow scrolling -->
    <main>
        <h1>Scroll down to see the effect</h1>
        <p>Keep scrolling...</p>
    </main>

    <script src="script.js"></script>
</body>
</html>
```

### 2. The CSS
The CSS handles the layout, the actual sticky positioning, and a special `.scrolled` class that JavaScript will trigger later.

```css
/* Basic Reset */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: sans-serif;
}

/* --- HEADER STYLES --- */
header {
    background-color: #1a1a1a;
    color: white;
    padding: 30px 50px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    
    /* The Magic CSS to make it sticky */
    position: sticky;
    top: 0;
    z-index: 1000; /* Ensures it stays above other content */
    
    /* Smooth transition for when JavaScript changes the styles */
    transition: all 0.3s ease-in-out; 
}

header nav a {
    color: white;
    text-decoration: none;
    margin-left: 20px;
    font-weight: bold;
}

/* --- JAVASCRIPT TRIGGERED CLASS --- */
/* This class is added by JS when the user scrolls down */
header.scrolled {
    padding: 15px 50px; /* Shrinks the header */
    background-color: rgba(26, 26, 26, 0.95); /* Adds slight transparency */
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); /* Adds a drop shadow */
}

/* Dummy content styling to make the page scrollable */
main {
    height: 2000px;
    padding: 50px;
    background: linear-gradient(to bottom, #f4f4f4, #cccccc);
}
```

### 3. The JavaScript
The JavaScript listens for the user to scroll. If they scroll down further than 50 pixels, it adds the `.scrolled` class to the header. If they scroll back to the top, it removes the class.

```javascript
// Select the header element
const header = document.getElementById('site-header');

// Listen for the scroll event on the window
window.addEventListener('scroll', () => {
    
    // Check how far the user has scrolled down the Y axis
    if (window.scrollY > 50) {
        // If scrolled past 50px, add the 'scrolled' class
        header.classList.add('scrolled');
    } else {
        // If at the top of the page, remove the 'scrolled' class
        header.classList.remove('scrolled');
    }
    
});
```

### How it works:
1. **`position: sticky; top: 0;`**: This CSS rule tells the browser to treat the header as part of the normal document flow until the user scrolls past it. Once it reaches `0px` from the top of the screen, it "sticks" there.
2. **`z-index: 1000;`**: This ensures the header doesn't hide underneath your text or images as you scroll down.
3. **The JavaScript Event Listener**: By using `window.addEventListener('scroll')`, we can detect the exact moment a user starts scrolling and animate the header so it shrinks and gains a shadow, which is a very popular modern UI pattern."""
USER_EXAMPLE = (
    """Show me a code snippet of a website's sticky header in CSS and JavaScript."""
)


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
            shift_enter=True,
        )
        self.shape = ft.BoxShape.RECTANGLE
        self.border_radius = 16
        self.bgcolor = ft.Colors.BLACK_54
        self.padding = ft.Padding(24, 16, 24, 16)

        self.content = ft.Column([self.input])
        self.on_send: Signal[str, None] = Signal(str)

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
        self.space = ft.Divider(height=400, opacity=0, visible=cfg.end_space)
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

        self.content = ft.Column(
            [self.avatar, self.thoughts, self.text, self.loading, self.space]
        )

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
            self.thought_markdown.value += thoughts
            self.thought_markdown.visible = True
            cfg = manager.get_config()
            self.thoughts.tooltip = self.thought_markdown.value[-cfg.tooltip_len:]
            self.thoughts.visible = True

    @staticmethod
    def ai(text: str = "", thoughts: str = "") -> "Message":
        msg = Message()
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
        msg.thoughts.tooltip = f"Arguments:\njson\n{json.dumps(args, indent=4)[-cfg.tooltip_len:]}\n"
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
        self.input.on_send.connect(self.on_send)

        self.settings = AutoConfig(
            title="Input message options.\nDon't touch if you don't know what you're doing.",
            config={
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
            visible=False,
        )

        def toggle_visible():
            self.settings.visible = not self.settings.visible
            if self.settings.visible:
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
            controls=[], scroll=ft.ScrollMode.AUTO, expand=True, expand_loose=True, auto_scroll=True
        )
        self.column = ft.Column(
            [self.messages, ft.Column([self.input])],
            expand=True,
        )
        self.row = ft.Row([self.column, self.settings], expand=True)
        self.stack = ft.Stack([self.row])
        self.content = ft.SelectionArea(self.stack)
        self.align = ft.Alignment.CENTER
        self.width = 1024
        self.expand = True

        self.finished: bool = True

        # self.add_message(
        #     Message.user(USER_EXAMPLE),
        #     Message.ai(AI_EXAMPLE),
        # )
        self.listening = asyncio.create_task(self.listen())

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
                            Path("/home/luuppi/.null-shift/brain/history.json").rename(
                                f"/home/luuppi/.null-shift/brain/history {datetime.now().strftime('%d-%m-%y %H-%M-%S')}.json"
                            )
                            self.add_message(Message.user("History moved."))
                        case "rm":
                            Path("/home/luuppi/.null-shift/brain/history.json").unlink(
                                missing_ok=True
                            )
                            self.add_message(Message.user("History deleted."))
                        case "c":
                            self.messages.controls.clear()
                        case "crm" | "rmc" | "rm c" | "c rm":
                            Path("/home/luuppi/.null-shift/brain/history.json").unlink(
                                missing_ok=True
                            )
                            self.messages.controls.clear()
                        case _:
                            self.add_message(Message.user(f"Unknown argument '{arg}'"))
                case _:
                    self.add_message(Message.user(f"Unknown command: '{command}'"))
            self.input.input.value = ""
            self.update()
            return

        if self.finished:
            self.finished = False
            self.input.input.value = ""
            self.add_message(Message.user(text))
            c = self.settings.config
            options = {k: v.selected for k, v in c.items() if v.selected}
            options["body"] = text
            logger.info("%s", json.dumps(options, indent=2))
            self.update()
            await sock_in.send_multipart(
                InputMessage.model_validate(options).to_bus().encoded()
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
                                self.add_message(Message.ai())

                            out = OutputMessage.from_bus(msg)
                            if out is None:
                                logger.error("Error validating output message.")
                                continue

                            self.append_last(out.text or "", out.reasoning or "")
                        case MessageTopic.TOOL_CALL:
                            out = OutputMessage.from_bus(msg)
                            if out is None:
                                logger.error("Error validating output message.")
                                continue
                            self.append_text().loading.visible = False
                            self.add_message(
                                Message.tool(out.tool_name or "", out.tool_call_id or "", out.tool_args or {})
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
                                tool = Message.tool(out.tool_name or "", out.tool_call_id or "", {})
                            tool.add_tool_response(
                                out.tool_result or "(No return value)"
                            )
                        case MessageTopic.STARTED:
                            self.add_message(Message.ai())
                        case MessageTopic.FINISHED:
                            self.finished = True
                            self.append_text().loading.visible = False
                        case MessageTopic.ABORT:
                            ai = self.append_last()
                            if ai:
                                ai.loading.visible = False
                            self.add_message(Message.user("Abort"))
                    self.update()
                except Exception as e:
                    logger.error("Error while listening: %s", e)
        except asyncio.CancelledError:
            logger.info("Shutting down listener.")

    def add_message(self, message: Message):
        if self.messages.controls:
            assert isinstance(self.messages.controls[-1], Message)
            self.messages.controls[-1].space.visible = False
        self.messages.controls.append(message)

    def append_text(self, text: str = "", thoughts: str = "", idx: int = -1) -> Message:
        """Append text to last message or specified index."""
        if not self.messages.controls:
            logger.error("Appending to an empty message list!!!")

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


async def main(page: ft.Page):
    global ctx, sock_in, sock_out
    ctx = zmq.asyncio.Context()
    sock_in = ctx.socket(zmq.PUSH)
    sock_out = ctx.socket(zmq.SUB)
    sock_in.connect("tcp://localhost:5555")
    sock_out.connect("tcp://localhost:5556")
    sock_out.subscribe(b"")

    logging.basicConfig(level="INFO")

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


ctx: zmq.asyncio.Context
sock_in: zmq.asyncio.Socket
sock_out: zmq.asyncio.Socket


ft.run(main)
