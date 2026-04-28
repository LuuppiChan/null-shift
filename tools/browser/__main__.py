import asyncio
import logging
import os
from typing import Any, AsyncIterable

import zmq.asyncio
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from global_types import BusMessage
from tools.browser.config import manager
from tools.browser.message_types import Action, BrowserMessage
from tools.browser.tools import (
    click_element,
    control_video,
    execute_misc_action,
    extract_attribute,
    fill_input,
    get_agent_dom,
    hover_element,
    press_keyboard_key,
    scroll_page,
    select_combo_option,
    set_slider,
    take_element_screenshot,
    take_page_screenshot,
)

os.environ["NODE_OPTIONS"] = "--no-deprecation"
logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


class BrowserControl:
    def __init__(self) -> None:
        self.port = 9222
        self.p: Playwright
        self.browser: Browser
        self.ctx: BrowserContext
        self.sock: zmq.asyncio.Socket

    async def active_page(self) -> Page | None:
        """Get current active page."""
        api_ctx = await self.p.request.new_context()
        response = await api_ctx.get(f"http://localhost:{self.port}/json/list")
        targets = await response.json()
        await api_ctx.dispose()

        active_id = None
        for target in targets:
            if target.get("type") == "page" and not target.get("url", "").startswith(
                "devtools://"
            ):
                active_id = target["id"]
                break

        if active_id is None:
            return None

        for page in self.ctx.pages:
            try:
                # Open a temporary, lightweight CDP session to ask the page for its true ID
                client = await self.ctx.new_cdp_session(page)
                target_info = await client.send("Target.getTargetInfo")
                await client.detach()  # Clean up the session

                # If the IDs match, this is definitively our active tab
                if target_info["targetInfo"]["targetId"] == active_id:
                    return page
            except Exception:
                # Ignore pages that have crashed or restrict CDP (like chrome:// extensions)
                continue
        return None

    async def run(self):
        async with async_playwright() as p:
            # init stuff
            self.p = p
            logger.info("Connecting to browser...")
            browser = await p.chromium.connect_over_cdp(f"http://localhost:{self.port}")
            logger.info("Connected")
            self.browser = browser
            ctx = browser.contexts[0]
            self.ctx = ctx

            # actual code
            async for msg in self.listen():
                match msg.action:
                    case Action.LIST_TABS:
                        tabs: list[dict[str, int | bool | str]] = []
                        active = await self.active_page()
                        for i, p in enumerate(self.ctx.pages):
                            try:
                                title = await p.title()
                            except Exception:
                                title = "Error"
                            tabs.append(
                                {
                                    "index": i,
                                    "title": title,
                                    "url": p.url,
                                    "active": p == active,
                                }
                            )
                        await self.send(
                            "\n".join(
                                [
                                    f"[{tab['index']}]: {tab['title']} ({tab['url']})"
                                    for tab in tabs
                                ]
                            )
                        )
                        continue
                    case Action.SWITCH_TAB:
                        idx = msg.kwargs.get("tab_index")
                        if idx is not None and 0 <= idx < len(self.ctx.pages):
                            await self.ctx.pages[idx].bring_to_front()
                            await self.send(f"Success: Switched to tab {idx}")
                        else:
                            await self.send("Error: Invalid tab index")
                        continue
                    case Action.CLOSE_TAB:
                        idx = msg.kwargs.get("tab_index")
                        if idx is not None and 0 <= idx < len(self.ctx.pages):
                            await self.ctx.pages[idx].close()
                            await self.send(f"Success: Closed tab {idx}")
                        else:
                            await self.send("Error: Invalid tab index")
                        continue
                    case Action.NEW_TAB:
                        url = msg.kwargs.get("url")
                        new_page = await self.ctx.new_page()
                        if url:
                            await new_page.goto(url)
                        await self.send("Success: Opened new tab")
                        continue

                page = await self.active_page()
                if page is None:
                    await self.send(
                        "Error: No active page available. The browser might have no open tabs, or the current tab crashed/is restricted. Use the 'browser_new_tab' tool to open a fresh page."
                    )
                    continue
                match msg.action:
                    case Action.DOM:
                        await self.send(await get_agent_dom(page))
                    case Action.CLICK:
                        await self.send(await click_element(page, **msg.kwargs))
                    case Action.TYPE:
                        await self.send(await fill_input(page, **msg.kwargs))
                    case Action.EXTRACT_ELEMENT:
                        await self.send(await extract_attribute(page, **msg.kwargs))
                    case Action.PRESS_KEY:
                        await self.send(await press_keyboard_key(page, **msg.kwargs))
                    case Action.SCROLL:
                        await self.send(await scroll_page(page, **msg.kwargs))
                    case Action.HOVER:
                        await self.send(await hover_element(page, **msg.kwargs))
                    case Action.MISC_ACTION:
                        await self.send(await execute_misc_action(page, **msg.kwargs))
                    case Action.NAVIGATE:
                        url = msg.kwargs.get("url")
                        if url:
                            await page.goto(url)
                            await self.send(f"Success: Navigated to {url}")
                        else:
                            await self.send("Error: No URL provided")
                    case Action.SET_SLIDER:
                        await self.send(await set_slider(page, **msg.kwargs))
                    case Action.PAGE_SCREENSHOT:
                        await self.send(await take_page_screenshot(page))
                    case Action.ELEMENT_SCREENSHOT:
                        await self.send(
                            await take_element_screenshot(page, **msg.kwargs)
                        )
                    case Action.SELECT_OPTION:
                        await self.send(await select_combo_option(page, **msg.kwargs))
                    case Action.VIDEO_CONTROL:
                        await self.send(await control_video(page, **msg.kwargs))

    async def send(self, value: dict[str, Any] | str):
        self.sock.send_multipart(
            BusMessage(topic=Action.RETURN, payload={"result": value}).encoded()
        )

    async def listen(self) -> AsyncIterable[BrowserMessage]:
        ctx = zmq.asyncio.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(manager.get_config().socket_path)
        self.sock = sock
        while True:
            frames = await sock.recv_multipart()
            message: BusMessage | None = BusMessage.decoded(frames)
            if message is None:
                logger.error("Error parsing bus message.")
                continue
            action = BrowserMessage.from_bus_msg(message)
            if action is None:
                logger.error("Error parsing BrowserMessage.")
                continue
            yield action


def run():
    bc = BrowserControl()
    asyncio.run(bc.run())


if __name__ == "__main__":
    run()
