import logging
from enum import StrEnum
from typing import Any

import zmq
from langchain_core.tools import StructuredTool

from global_types import BusMessage

logger = logging.getLogger(__name__)


class Action(StrEnum):
    LIST_TOOLS = "list_tools"
    CALL_TOOL = "call_tool"
    GET_PROMPT = "get_prompt"
    CLOSE_SESSION = "close_session"


# Must cover the server's init_timeout_s (fresh daemon spawns the MCP
# subprocess on the first LIST_TOOLS) and call_timeout_s plus headroom.
LIST_TIMEOUT_MS = 60_000
CALL_TIMEOUT_MS = 300_000

_SOCKET_ADDRESS = "tcp://127.0.0.1:5558"

_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def send_mcp_request(action: str, timeout_ms: int, **payload: Any) -> Any:
    """
    Sends a request to the MCP tool server via ZeroMQ and returns the result.

    Same pattern as browser.send_browser_request: a fresh REQ socket per call,
    so tool modules stay stateless and reload-safe. The long-lived MCP sessions
    live in the tools/mcp daemon.
    """
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.connect(_SOCKET_ADDRESS)

    try:
        socket.send_multipart(
            BusMessage(topic=str(action), payload=payload).encoded()
        )
        frames = socket.recv_multipart()
        response = BusMessage.decoded(frames)
        if response is None:
            raise ValueError("Failed to decode BusMessage response from the MCP tool.")
        return response.payload.get("result")
    except zmq.Again:
        return f"Response timed out after {timeout_ms} ms"
    finally:
        socket.close()
        ctx.destroy()


def load_mcp_tools(server: str) -> list[StructuredTool]:
    """
    Queries the MCP daemon for its tool list and wraps every tool as a
    StructuredTool whose body is a plain ZMQ request, mirroring how the
    browser @tool wrappers delegate to send_browser_request.
    """
    defs = send_mcp_request(Action.LIST_TOOLS, LIST_TIMEOUT_MS, server=server)
    tools: list[StructuredTool] = []
    for d in defs:
        def _run(_server=server, _name=d["name"], **kwargs: Any) -> str:
            return send_mcp_request(
                Action.CALL_TOOL, CALL_TIMEOUT_MS,
                server=_server, tool=_name, kwargs=kwargs,
            )

        tools.append(
            StructuredTool.from_function(
                func=_run,
                name=d["name"],
                description=d.get("description") or d["name"],
                args_schema=d.get("inputSchema") or {},
            )
        )
    logger.info("Loaded %d MCP tools from server '%s'.", len(tools), server)
    return tools


# for tool in load_mcp_tools("rea"):
#     globals()[tool.name] = tool
