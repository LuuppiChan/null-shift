"""
Definitions and Specifications: Provide clear, explicit, specific, and complete descriptions of any definitions and/or specifications unique to the problem.
"""

from typing import Any

import zmq
from core.helpers import PromptHelper, xml_tag
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
    socket.setsockopt(zmq.RCVTIMEO, 400)
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


def browser_list_tabs() -> str:
    """
    Lists all currently open tabs with their index, title, url, and active state.

    Example:
        `browser_list_tabs()`
        Returns a list of dictionaries detailing the open tabs.
    """
    return send_browser_request(Action.LIST_TABS)


def collect() -> str | None:
    try:
        focused = [tab for tab in browser_list_tabs().split("\n") if "active" in tab]
        if not focused:
            return None

        return xml_tag(
            focused,
            "currently_focused_browser_tab",
            "Possibly the currently focused browser tab. To read the contents use the get DOM tool.",
        )
    except Exception as e:
        print("Error indexing browser tabs.", e)
