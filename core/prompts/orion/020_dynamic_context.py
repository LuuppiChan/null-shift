"""
Definitions and Specifications: Provide clear, explicit, specific, and complete descriptions of any definitions and/or specifications unique to the problem.
"""

from typing import Any

import zmq
from core.helpers import PromptHelper
from core.core_data import data
from global_types import BusMessage
from tools.browser.message_types import Action

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
    # config = manager.get_config()

    # The browser server binds to e.g., tcp://*:5557, but the client must connect to localhost
    # address = config.socket_path.replace("*", "localhost")
    address = "tcp://localhost:5557"

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


def browser_list_tabs() -> list[dict[str, int | bool | str]]:
    """
    Lists all currently open tabs with their index, title, url, and active state.

    Example:
        `browser_list_tabs()`
        Returns a list of dictionaries detailing the open tabs.
    """
    return send_browser_request(Action.LIST_TABS)


def collect() -> str | None:
    prompt = PromptHelper("dynamic_context")
    prompt.add_part(
        "This section contains dynamic data about the surrounding environment meaning it's always up-to-date.\nThis information may or may not be relevant to your task, it is up for you to decide.",
        "description",
    )
    prompt.add_part(data.datetime(), "datetime")
    prompt.add_part(data.home_path(), "user_home_path")
    prompt.add_part(
        data.scratchpad(),
        "assistant_scratchpad_path",
        "This is your dedicated workspace. You can freely read, write, and modify files here. Use this directory to draft code, store intermediate data, write down step-by-step plans, or keep track of your thoughts during complex tasks. Consider it your working memory.",
    )
    try:
        focused = [tab for tab in browser_list_tabs() if tab.get("active")][0]
        prompt.add_part(focused, "currently_focused_browser_tab", "Currently focused browser tab. To read the contents use the get DOM tool.")
    except (IndexError, AttributeError):
        print("Error indexing browser tabs. No active tab.")

    return prompt.compile()
