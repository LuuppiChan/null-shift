from typing import Any, Optional

import zmq
from langchain_core.tools import tool

from global_types import BusMessage
from tools.browser.config import manager
from tools.browser.message_types import Action

# Module-level ZMQ context to avoid creating new contexts for every request
_zmq_ctx = zmq.Context()


def send_browser_request(_action: str, **kwargs: Any) -> Any:
    """
    Sends a request to the browser tool via ZeroMQ and returns the result.

    Args:
        _action: The action to perform (e.g., from Action enum like Action.LIST_TABS).
        **kwargs: Arguments to pass as the payload for the action.

    Returns:
        The 'result' field from the browser's response payload.
    """
    # I guess this works lol
    config = manager.get_config()

    # The browser server binds to e.g., tcp://*:5557, but the client must connect to localhost
    address = config.socket_path.replace("*", "localhost")

    socket = _zmq_ctx.socket(zmq.REQ)
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


@tool
def browser_get_dom() -> str:
    """
    Gets the interactive Document Object Model (DOM) of the current active page.

    Example:
        `browser_get_dom()`
        Returns a string representation of the DOM with interactive elements tagged with their [ID].
    """
    return send_browser_request(Action.DOM)


@tool
def browser_click(element_id: int) -> str:
    """
    Clicks an interactive element on the page using its ID.

    Example:
        `browser_click(element_id=15)`
        Simulates a mouse click on the element marked as [15].
    """
    return send_browser_request(Action.CLICK, element_id=element_id)


@tool
def browser_type(element_id: int, text: str, press_enter: bool = False) -> str:
    """
    Clears an input field or textarea and types the provided text into it.

    Example:
        `browser_type(element_id=4, text="search query", press_enter=True)`
        Types "search query" into input [4] and automatically submits by pressing Enter.
    """
    return send_browser_request(
        Action.TYPE, element_id=element_id, text=text, press_enter=press_enter
    )


@tool
def browser_extract_attribute(element_id: int, attribute: str) -> str:
    """
    Extracts a specific HTML attribute from an element (e.g., 'src', 'href', 'alt').

    Example:
        `browser_extract_attribute(element_id=12, attribute="href")`
        Returns the destination URL of link [12].
    """
    return send_browser_request(
        Action.EXTRACT_ELEMENT, element_id=element_id, attribute=attribute
    )


@tool
def browser_press_key(key: str) -> str:
    """
    Presses a specific keyboard key on the active page.

    Example:
        `browser_press_key(key="Escape")`
        Presses the ESC key (useful for closing modals).
        Other common keys: "Enter", "Tab", "ArrowDown", "ArrowUp".
    """
    return send_browser_request(Action.PRESS_KEY, key=key)


@tool
def browser_scroll(direction: str = "down") -> str:
    """
    Scrolls the active page up or down to reveal hidden content.

    Example:
        `browser_scroll(direction="down")`
        Scrolls down one page length.
    """
    return send_browser_request(Action.SCROLL, direction=direction)


@tool
def browser_hover(element_id: int) -> str:
    """
    Hovers the mouse over an element. Useful for opening CSS-based dropdown menus.

    Example:
        `browser_hover(element_id=8)`
        Moves the mouse pointer over element [8] and triggers hover styles.
    """
    return send_browser_request(Action.HOVER, element_id=element_id)


@tool
def browser_misc_action(element_id: int, action_event: str) -> str:
    """
    Executes miscellaneous actions or dispatches raw HTML/DOM events on an element.
    This is highly useful for specific UI interactions that aren't a simple left-click.

    Supported native actions: 'dblclick', 'rightclick', 'focus', 'blur', 'check', 'uncheck'
    Supported DOM events: 'submit', 'mouseenter', 'mouseleave', 'change', 'input', 'keydown'

    Examples:
        `browser_misc_action(element_id=14, action_event="rightclick")` -> Opens context menu.
        `browser_misc_action(element_id=3, action_event="dblclick")` -> Double clicks an item.
        `browser_misc_action(element_id=5, action_event="check")` -> Toggles a checkbox on.
        `browser_misc_action(element_id=9, action_event="blur")` -> Removes focus from an input.
    """
    return send_browser_request(
        Action.MISC_ACTION, element_id=element_id, action_event=action_event
    )


@tool
def browser_list_tabs() -> Any:
    """
    Lists all currently open tabs with their index, title, url, and active state.

    Example:
        `browser_list_tabs()`
        Returns a list of dictionaries detailing the open tabs.
    """
    return send_browser_request(Action.LIST_TABS)


@tool
def browser_switch_tab(tab_index: int) -> str:
    """
    Switches the active browser focus to the tab at the given index.

    Example:
        `browser_switch_tab(tab_index=1)`
        Brings the second tab (index 1) to the front.
    """
    return send_browser_request(Action.SWITCH_TAB, tab_index=tab_index)


@tool
def browser_close_tab(tab_index: int) -> str:
    """
    Closes the browser tab at the given index.

    Example:
        `browser_close_tab(tab_index=2)`
        Closes the third tab (index 2).
    """
    return send_browser_request(Action.CLOSE_TAB, tab_index=tab_index)


@tool
def browser_new_tab(url: Optional[str] = None) -> str:
    """
    Opens a new browser tab and optionally navigates to a URL immediately.

    Example:
        `browser_new_tab(url="https://example.com")`
        Creates a new tab and loads example.com.
    """
    return send_browser_request(Action.NEW_TAB, url=url)


@tool
def browser_navigate(url: str) -> str:
    """
    Navigates the current active tab to a new URL.

    Example:
        `browser_navigate(url="https://wikipedia.org")`
        Redirects the current page to Wikipedia.
    """
    return send_browser_request(Action.NAVIGATE, url=url)


@tool
def browser_set_slider(element_id: int, value: str) -> str:
    """
    Sets the value of a slider (<input type="range">) element.

    Example:
        `browser_set_slider(element_id=7, value="50")`
        Moves the slider at ID 7 to the 50 mark.
    """
    return send_browser_request(Action.SET_SLIDER, element_id=element_id, value=value)


@tool
def browser_page_screenshot() -> Any:
    """
    Takes a full page screenshot and returns it. Usable by multimodal LLMs.

    Example:
        `browser_page_screenshot()`
        Returns a dictionary containing the base64 encoded image payload.
    """
    return send_browser_request(Action.PAGE_SCREENSHOT)


@tool
def browser_element_screenshot(element_id: int) -> Any:
    """
    Takes a screenshot of a specific element and returns it. Usable by multimodal LLMs.

    Example:
        `browser_element_screenshot(element_id=22)`
        Takes a bounded screenshot of element 22 (e.g., an image or a captcha).
    """
    return send_browser_request(Action.ELEMENT_SCREENSHOT, element_id=element_id)


@tool
def browser_select_option(element_id: int, option_value: str) -> str:
    """
    Selects an option in a combo box (<select> element) by label or value.

    Example:
        `browser_select_option(element_id=9, option_value="United States")`
        Selects the "United States" option from dropdown [9].
    """
    return send_browser_request(
        Action.SELECT_OPTION, element_id=element_id, option_value=option_value
    )


@tool
def browser_video_control(
    element_id: int, action: str, value: Optional[float] = None
) -> str:
    """
    Controls a video element. Valid actions: 'play', 'pause', 'seek', 'mute', 'unmute'.
    'seek' requires a value in seconds.

    Examples:
        `browser_video_control(element_id=4, action="play")` -> Starts the video.
        `browser_video_control(element_id=4, action="seek", value=30.5)` -> Skips to 30.5 seconds.
    """
    return send_browser_request(
        Action.VIDEO_CONTROL, element_id=element_id, action=action, value=value
    )
