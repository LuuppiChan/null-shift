import asyncio
from typing import Any, Literal

import regex
import zmq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from core.config import tool_manager
from core.helpers import ask_ai, enforce_character_limit
from global_types import BusMessage
from tools.browser.message_types import Action, TabDict


def send_browser_request(_action: str, **kwargs: Any) -> Any:
    """
    Sends a request to the browser tool via ZeroMQ and returns the result.

    Args:
        _action: The action to perform (e.g., from Action enum like Action.LIST_TABS).
        **kwargs: Arguments to pass as the payload for the action.

    Returns:
        The 'result' field from the browser's response payload.
    """
    ctx = zmq.Context()
    # I guess this works lol
    # from tools.browser.config import manager
    from core.config import tool_manager as manager

    config = manager.get_config()

    # The browser server binds to e.g., tcp://*:5557, but the client must connect to localhost
    # address = config.socket_path.replace("*", "localhost")
    address = "tcp://127.0.0.1:5557"

    socket = ctx.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, config.brower_socket_timeout_ms)
    socket.connect(address)

    try:
        # Construct and send the outgoing message
        msg = BusMessage(topic=str(_action), payload=kwargs)
        socket.send_multipart(msg.encoded())

        # Wait for and decode the incoming response
        frames = socket.recv_multipart()
        response = BusMessage.decoded(frames)

        if response is None:
            raise ValueError(
                "Failed to decode BusMessage response from the browser tool."
            )

        return response.payload.get("result")
    finally:
        socket.close()
        ctx.destroy()


@tool(
    description="""Gets the interactive Document Object Model (DOM) of the current active page as if it was a file.

Arguments:
    start_line: Optional line to start the read at. Starts at the first line if not specified. (1-indexed)
    end_line: Optional line to end the read at. Reads to the end if not specified. (exclusive)
    show_line_numbers: Show virtual line numbers.
    character_limit: Limits the max character count. Most of the time doesn't need to be touched.
"""
)
def browser_get_dom(
    start_line: int | None = None,
    end_line: int | None = None,
    show_line_numbers: bool = False,
    character_limit: int = tool_manager.get_config().browser_dom_char_limit,
) -> str:
    dom: str = send_browser_request(Action.DOM)

    lines = dom.splitlines()
    length = len(lines)

    if show_line_numbers:
        lines = [f"{i + 1} {line}" for i, line in enumerate(lines)]

    if end_line is None:
        dom = "\n".join(lines[start_line:])
    else:
        dom = "\n".join(lines[start_line:end_line])

    if not dom:
        return f"Empty output. DOM has {length} line(s)."

    return enforce_character_limit(dom, character_limit)


@tool(
    description="""Simple grep-like tool to find things from the DOM easier. Parameters map to grep equivalents. Matches are per-line only."""
)
def browser_grep_dom(
    pattern: str,
    ignore_case: bool = True,
    after: int | None = None,
    before: int | None = None,
    context: int | None = None,
) -> str:
    compiled = regex.compile(pattern, regex.IGNORECASE if ignore_case else 0)
    dom: str = send_browser_request(Action.DOM)

    # these could be moved to the end, but eh
    lines = [f"{i + 1} {line}" for i, line in enumerate(dom.splitlines())]
    matches = [j for j, line in enumerate(lines) if compiled.search(line)]
    context_lines = set()

    before = context if context is not None else (after or 0)
    after = context if context is not None else (before or 0)

    for idx in matches:
        context_lines.update(
            range(max(0, idx - before), min(len(lines), idx + after + 1))
        )

    return "\n".join([lines[idx] for idx in sorted(context_lines)])


@tool(
    description="""Return an AI summary of the current page.
An economical convenience tool to prevent token wasting.
Use this for page overviews and other simple questions about the page.
You can ask any question about the currently focused website.
Note that the model doesn't have any external context besides the DOM and instructions to summarize web pages."""
)
def browser_page_summary(message: str = "Summarize the contents of this page") -> str:
    return asyncio.run(
        ask_ai(
            message,
            "You are an AI assistant that summarizes web pages. "  # note no comma
            "Your task is to answer the user's question about the current page or summarize the web page.",
            "low",
            [
                HumanMessage(str(send_browser_request(Action.DOM))),
                AIMessage("Page content acknowledged"),
            ],
        )
    )


@tool(description="""Clicks an interactive element on the page using its ID.

Example:
    `browser_click(element_id=15)`
    Simulates a mouse click on the element marked as [15].""")
def browser_click(element_id: int) -> str:
    return send_browser_request(Action.CLICK, element_id=element_id)


@tool(description="""High level text typing tool.

Clears an input field or textarea and types the provided text into it.
Overwrite controls whether to set the field content or append to it.
Works on any shown input fields on the DOM including Google docs.

Example:
    `browser_type(element_id=4, text="search query", press_enter=True)`
    Types "search query" into input [4] and automatically submits by pressing Enter.""")
def browser_type(
    element_id: int, text: str, press_enter: bool = False, overwrite: bool = True
) -> str:
    return send_browser_request(
        Action.TYPE,
        element_id=element_id,
        text=text,
        press_enter=press_enter,
        overwrite=overwrite,
    )


@tool(
    description="""Extracts a specific HTML attribute from an element (e.g., 'src', 'href', 'alt').

Example:
    `browser_extract_attribute(element_id=12, attribute="href")`
    Returns the destination URL of link [12]."""
)
def browser_extract_attribute(element_id: int, attribute: str) -> str:
    return send_browser_request(
        Action.EXTRACT_ELEMENT, element_id=element_id, attribute=attribute
    )


# @tool(description="""Presses a specific keyboard key on the active page.
#
# Example:
#     `browser_press_key(key="Escape")`
#     Presses the ESC key (useful for closing modals).
#     Other common keys: "Enter", "Tab", "ArrowDown", "ArrowUp".""")
def browser_press_key(key: str) -> str:
    return send_browser_request(Action.PRESS_KEY, key=key)


@tool(
    description="""Presses each key in the sequence in order on the active page.

Each entry is a complete key press: a key name, or a combination joined with '+' such as "Control+e". Modifier names like "Control" alone do NOT combine with following entries.

Examples:
    `browser_press_keys(keys=["a", "b", "Enter"])` -> types a, b, then Enter.
    `browser_press_keys(keys=["Control+a", "Delete"])` -> select all, then delete.
    `browser_press_keys(keys=["ArrowDown", "ArrowDown", "Enter"])` -> navigate a menu and confirm.

Presses are sent one by one and are not atomic: if an entry fails, earlier presses have already been applied. Stops at the first invalid key and reports its index and name."""
)
def browser_press_keys(keys: list[str]) -> str:
    return send_browser_request(Action.PRESS_KEYS, keys=keys)


@tool(description="""Scrolls the active page up or down.
This is only useful when taking screenshots of the visible page.

Example:
    `browser_scroll(direction="down")`
    Scrolls down one page length.""")
def browser_scroll(direction: str = "down") -> str:
    return send_browser_request(Action.SCROLL, direction=direction)


@tool(
    description="""Hovers the mouse over an element. Useful for opening CSS-based dropdown menus.

Example:
    `browser_hover(element_id=8)`
    Moves the mouse pointer over element [8] and triggers hover styles."""
)
def browser_hover(element_id: int) -> str:
    return send_browser_request(Action.HOVER, element_id=element_id)


@tool(
    description="""Executes miscellaneous actions or dispatches raw HTML/DOM events on an element.
This is highly useful for specific UI interactions that aren't a simple left-click.

Supported native actions: 'dblclick', 'rightclick', 'focus', 'blur', 'check', 'uncheck'
Supported DOM events: 'submit', 'mouseenter', 'mouseleave', 'change', 'input', 'keydown'

Examples:
    `browser_misc_action(element_id=14, action_event="rightclick")` -> Opens context menu.
    `browser_misc_action(element_id=3, action_event="dblclick")` -> Double clicks an item.
    `browser_misc_action(element_id=5, action_event="check")` -> Toggles a checkbox on.
    `browser_misc_action(element_id=9, action_event="blur")` -> Removes focus from an input."""
)
def browser_misc_action(element_id: int, action_event: str) -> str:
    return send_browser_request(
        Action.MISC_ACTION, element_id=element_id, action_event=action_event
    )


@tool(
    description="""Run javascript on the current page. This is your last resort and using this tool is highly discouraged.

expression: JavaScript expression to be evaluated in the browser context. If the expression evaluates to a function, the function is automatically invoked.
"""
)
def browser_page_javascript(expression: str) -> str:
    return send_browser_request(Action.PAGE_JAVASCRIPT, expression=expression)


def _browser_list_tabs() -> str:
    return "\n".join(
        [
            f"[{tab['index']} (active: {tab['active']})]: {tab['title']} ({tab['url']})"
            for tab in send_browser_request(Action.LIST_TABS)
        ]
    )


@tool(
    description="""Lists all currently open tabs with their index, title, url, and active state."""
)
def browser_list_tabs() -> str:
    return _browser_list_tabs()


@tool(description="""Switches the active browser focus to the tab at the given index.

Example:
    `browser_switch_tab(tab_index=1)`
    Brings the second tab (index 1) to the front.""")
def browser_switch_tab(tab_index: int) -> str:
    return send_browser_request(Action.SWITCH_TAB, tab_index=tab_index)


@tool(description="""Closes the browser tab with the given index and url.""")
def browser_close_tab(url: str, index: int) -> str:
    tabs: list[TabDict] = send_browser_request(Action.LIST_TABS)
    for i, tab in enumerate(tabs):
        if url == tab["url"] and i == index:
            return send_browser_request(Action.CLOSE_TAB, tab_index=tab["index"])
    return f"Invalid pattern.\nAvailable tabs:\n{_browser_list_tabs()}"


@tool(
    description="""Opens a new browser tab and optionally navigates to a URL immediately.

Example:
    `browser_new_tab(url="https://example.com")`
    Creates a new tab and loads example.com."""
)
def browser_new_tab(url: str | None = None) -> str:
    return send_browser_request(Action.NEW_TAB, url=url)


@tool(description="""Navigates the current active tab to a new URL.

Example:
    `browser_navigate(url="https://wikipedia.org")`
    Redirects the current page to Wikipedia.""")
def browser_navigate(url: str) -> str:
    return send_browser_request(Action.NAVIGATE, url=url)


@tool(description="""Sets the value of a slider (<input type="range">) element.

Example:
    `browser_set_slider(element_id=7, value="50")`
    Moves the slider at ID 7 to the 50 mark.""")
def browser_set_slider(element_id: int, value: str) -> str:
    return send_browser_request(Action.SET_SLIDER, element_id=element_id, value=value)


@tool(
    description="""Takes a full page screenshot and returns it. Useful for reading the screen when the browser_get_dom doesn't return enough context."""
)
def browser_page_screenshot() -> Any:
    return send_browser_request(Action.PAGE_SCREENSHOT)


@tool(description="""Takes a screenshot of a specific element and returns it.""")
def browser_element_screenshot(element_id: int) -> Any:
    return send_browser_request(Action.ELEMENT_SCREENSHOT, element_id=element_id)


@tool(
    description="""Selects an option in a combo box. Works with native <select> elements and React custom dropdowns (e.g. react-select) in a single action.

Example:
    `browser_select_option(element_id=9, option_value="United States")`
    Selects the "United States" option from dropdown [9]."""
)
def browser_select_option(element_id: int, option_value: str) -> str:
    return send_browser_request(
        Action.SELECT_OPTION, element_id=element_id, option_value=option_value
    )


@tool(
    description="""Controls a video element. Valid actions: 'play', 'pause', 'seek', 'mute', 'unmute'.
'seek' requires a value in seconds.

Examples:
    `browser_video_control(element_id=4, action="play")` -> Starts the video.
    `browser_video_control(element_id=4, action="seek", value=30.5)` -> Skips to 30.5 seconds."""
)
def browser_video_control(
    element_id: int, action: str, value: float | None = None
) -> str:
    return send_browser_request(
        Action.VIDEO_CONTROL, element_id=element_id, action=action, value=value
    )


@tool(
    description="""Moves the visible agent cursor, clicks at viewport coordinates, or reports where it is. action="move" with element_id moves to that element; action="move" with x and y moves to viewport coordinates. action="click" with x and y moves to those coordinates and clicks there; use it only for targets not exposed in the DOM (e.g. canvas), otherwise prefer the element click tool. action="info" returns the cursor position and the element under it, which is useful for checking a target before clicking."""
)
def browser_cursor_action(
    action: Literal["move", "click", "info"],
    element_id: int | None = None,
    x: float | None = None,
    y: float | None = None,
) -> Any:
    return send_browser_request(
        Action.CURSOR_ACTION, action=action, element_id=element_id, x=x, y=y
    )
