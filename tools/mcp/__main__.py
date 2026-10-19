import asyncio
import base64
import logging
import tempfile
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, AsyncIterable

import zmq.asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from global_types import BusMessage
from tools.mcp.config import ServerConfig, manager

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


class Action:
    RETURN = "return"
    LIST_TOOLS = "list_tools"
    CALL_TOOL = "call_tool"
    GET_PROMPT = "get_prompt"
    CLOSE_SESSION = "close_session"


class MCPSession:
    """One stdio MCP server. Lives entirely on the daemon's event loop."""

    def __init__(
        self, name: str, command: str, args: list[str], url: str | None = None
    ) -> None:
        self.name = name
        self.command = command
        self.args = args
        self.url = url
        self.session: ClientSession | None = None
        self._stack = AsyncExitStack()

    @property
    def _params(self) -> StdioServerParameters:
        return StdioServerParameters(command=self.command, args=self.args)

    async def connect(self) -> None:
        if self.session is not None:
            return
        logger.info(
            "Starting MCP server '%s' (%s %s)...", self.name, self.command, self.args
        )
        if self.url:
            read, write, _ = await self._stack.enter_async_context(
                streamable_http_client(self.url)
            )
        else:
            read, write = await self._stack.enter_async_context(
                stdio_client(self._params)
            )

        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await asyncio.wait_for(
            self.session.initialize(), timeout=manager.get_config().init_timeout_s
        )
        logger.info("MCP server '%s' ready.", self.name)

    async def close(self) -> None:
        self.session = None
        await self._stack.aclose()

    async def list_tools(self) -> list[dict[str, Any]]:
        await self.connect()

        if self.session is None:
            logger.error(
                "Cannot list tools! Session is None on mcp '%s', returning an empty list.",
                self.name,
            )
            return []

        result = await self.session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema,
            }
            for t in result.tools
        ]

    async def call(self, tool_name: str, kwargs: dict[str, Any]) -> list:
        await self.connect()

        if self.session is None:
            logger.error("Trying to call session that's None on server '%s'", self.name)
            return []

        try:
            result = await asyncio.wait_for(
                self.session.call_tool(tool_name, kwargs),
                timeout=manager.get_config().call_timeout_s,
            )
        except Exception:
            # Transport died or hung: drop the session so the next call reconnects.
            logger.exception("Call to '%s' failed, dropping session.", tool_name)
            await self.close()
            raise

        if result.isError:
            # since it checks whether the attribute exists
            texts = [
                c.text  # pyright: ignore[reportAttributeAccessIssue]
                for c in result.content
                if getattr(c, "text", None)
            ]
            raise RuntimeError("\n".join(texts) or "Unknown tool error.")
        return result.content

    async def get_prompt(self, prompt_name: str) -> str:
        await self.connect()

        if self.session is None:
            logger.error(
                "Trying to get_prompt on session that's None on server '%s'", self.name
            )
            return ""

        result = await self.session.get_prompt(prompt_name)
        # same here it checks whether that field exists
        return "\n".join(
            m.content.text  # pyright: ignore[reportAttributeAccessIssue]
            for m in result.messages
            if getattr(m.content, "text", None)
        )


class Registry:
    def __init__(self) -> None:
        self._sessions: dict[str, MCPSession] = {}

    def _config_for(self, name: str) -> ServerConfig | None:
        for server in manager.get_config().servers:
            if server.name == name:
                return server
        logger.error("No config for server '%s'", name)
        return None

    def get(self, name: str) -> MCPSession | None:
        cfg = self._config_for(name)
        if cfg is None:
            return None

        session = self._sessions.get(name)
        if session is not None and (session.command, session.args) != (
            cfg.command,
            cfg.args,
        ):
            # Config changed since the session was created; respawn.
            asyncio.get_event_loop().create_task(session.close())
            session = None
        if session is None:
            session = MCPSession(name, cfg.command, cfg.args, cfg.url)
            self._sessions[name] = session
        return session

    async def close(self, name: str) -> None:
        session = self._sessions.pop(name, None)
        if session:
            await session.close()

    async def close_all(self) -> None:
        for name in list(self._sessions):
            await self.close(name)


def render_content(content_items: list) -> str:
    """Text goes through as-is; images are decoded to PNG files on disk."""
    parts: list[str] = []
    image_dir = (
        Path(manager.get_config().image_dir or tempfile.gettempdir()) / "mcp_images"
    )
    for i, item in enumerate(content_items):
        kind = getattr(item, "type", None)
        if kind == "text":
            parts.append(item.text)
        elif kind == "image":
            image_dir.mkdir(parents=True, exist_ok=True)
            ext = (item.mimeType or "image/png").split("/")[-1]
            path = image_dir / f"{item.__class__.__name__.lower()}_{i}.{ext}"
            path.write_bytes(base64.b64decode(item.data))
            parts.append(f"[image ({item.mimeType}) saved to `{path}`. View it with the file read tool.]")
        else:
            parts.append(str(item))
    return "\n".join(parts) if parts else "OK (no content returned)."


class MCPControl:
    def __init__(self) -> None:
        self.registry = Registry()
        self.sock: zmq.asyncio.Socket | None = None

    async def handle(self, msg: BusMessage) -> Any:
        payload = msg.payload
        server = payload.get("server", "")

        if not server:
            return "Error: Payload should include a server field."

        match msg.topic:
            case Action.LIST_TOOLS:
                session = self.registry.get(server)
                if session is None:
                    return f"Error: No server named '{server}'"

                return await session.list_tools()
            case Action.CALL_TOOL:
                session = self.registry.get(server)
                if session is None:
                    return f"Error: No server named '{server}'"

                content = await session.call(
                    payload["tool"], payload.get("kwargs") or {}
                )
                return render_content(content)
            case Action.GET_PROMPT:
                session = self.registry.get(server)
                if session is None:
                    return f"Error: No server named '{server}'"

                return await session.get_prompt(payload["name"])
            case Action.CLOSE_SESSION:
                await self.registry.close(server)
                return f"Success: Closed session '{server}'."
            case _:
                return f"Error: Unknown action '{msg.topic}'."

    async def send(self, value: Any) -> None:
        if self.sock is None:
            logger.error("Cannot send message, socket is not initialized")
            return
        self.sock.send_multipart(
            BusMessage(topic=Action.RETURN, payload={"result": value}).encoded()
        )

    async def listen(self) -> AsyncIterable[BusMessage]:
        ctx = zmq.asyncio.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(manager.get_config().socket_path)
        self.sock = sock
        try:
            while True:
                frames = await sock.recv_multipart()
                message: BusMessage | None = BusMessage.decoded(frames)
                if message is None:
                    logger.error("Error parsing bus message.")
                    await self.send("Error: Invalid bus message received.")
                    continue
                yield message
        finally:
            await self.registry.close_all()
            sock.close()
            ctx.term()

    async def run(self):
        async for msg in self.listen():
            try:
                await self.send(await self.handle(msg))
            except KeyError as e:
                await self.send(f"Error: {e.args[0]}")
            except RuntimeError as e:
                # Tool-level errors (isError results) are readable, pass them on.
                await self.send(f"Error: {e}")
            except Exception as e:
                logger.exception("Unexpected error processing action %s", msg.topic)
                await self.send(f"Error: An unexpected internal error occurred: {e}")


def run():
    control = MCPControl()
    try:
        asyncio.run(control.run())
    except KeyboardInterrupt:
        logger.info("Shutting down MCP control...")


if __name__ == "__main__":
    run()
