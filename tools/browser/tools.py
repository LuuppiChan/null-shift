import asyncio
import base64
from typing import Any

from playwright.async_api import Page


async def get_agent_dom(page: Page):
    agent_text = await page.evaluate(r"""() => {
        let idCounter = 1;

        const processNode = (node, hasInteractiveParent = false, inheritedRole = '', isInsideRichText = false) => {
            if (node.nodeType === Node.TEXT_NODE) {
                let text = node.textContent.replace(/\s+/g, ' ');
                if (text === ' ') return ' ';
                if (!text.trim()) return '';
                return text;
            }

            if (node.nodeType !== Node.ELEMENT_NODE) return '';

            const tag = node.tagName.toLowerCase();

            // 1. Removed 'video' from the ignore list
            if (['script', 'style', 'noscript', 'svg', 'canvas'].includes(tag)) return '';

            try {
                const style = window.getComputedStyle(node);
                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return '';
            } catch (e) {}

            // 2. Added 'video' to the Semantic Roles
            let currentRole = inheritedRole;
            if (tag === 'a') currentRole = 'link';
            else if (tag === 'button' || node.getAttribute('role') === 'button') currentRole = 'button';
            else if (tag === 'img' && !currentRole) currentRole = 'image';
            else if (tag === 'video' && !currentRole) currentRole = 'video';

            let isInt = false;
            let isRichText = false;
            try {
                // 3. Tell the script that <video> is a targetable, interactive element
                if (['a', 'button', 'input', 'select', 'textarea', 'img', 'video'].includes(tag)) isInt = true;
                else if (node.hasAttribute('onclick') || node.getAttribute('role') === 'button') isInt = true;
                else if (node.isContentEditable || node.getAttribute('role') === 'textbox') {
                    isInt = true;
                    isRichText = true;
                }
                else {
                    const style = window.getComputedStyle(node);
                    if (style.cursor === 'pointer') {
                        const parentStyle = node.parentElement ? window.getComputedStyle(node.parentElement) : null;
                        if (!parentStyle || parentStyle.cursor !== 'pointer') isInt = true;
                    }
                }

                if (isInt) {
                    const rect = node.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) isInt = false;
                }
            } catch (e) {
                isInt = false;
            }

            let childrenText = [];
            const passInteractiveState = hasInteractiveParent || (isInt && !isRichText);
            const passIsInsideRichText = isInsideRichText || isRichText;

            if (tag === 'iframe') {
                try {
                    let iframeBody = node.contentDocument && node.contentDocument.body;
                    if (iframeBody) {
                        let iframeRes = processNode(iframeBody, passInteractiveState, currentRole, passIsInsideRichText);
                        if (iframeRes) childrenText.push(iframeRes);
                    } else {
                        childrenText.push(" [IFRAME: Cross-Origin Restricted] ");
                    }
                } catch (e) {
                    childrenText.push(" [IFRAME: Cross-Origin Restricted] ");
                }
            } else {
                for (let child of node.childNodes) {
                    let childRes = processNode(child, passInteractiveState, currentRole, passIsInsideRichText);
                    if (childRes) childrenText.push(childRes);
                }
            }
            let combinedText = childrenText.join('');

            if (tag === 'img') {
                let alt = node.getAttribute('alt') || node.getAttribute('title') || "Image";
                combinedText = combinedText ? `${combinedText} ${alt}` : alt;
            }

            if (isInt && !hasInteractiveParent) {
                try {
                    let currentId = idCounter++;
                    node.setAttribute('data-agent-id', currentId);

                    let label = combinedText.trim();
                    if (!label) {
                        label = node.getAttribute('aria-label') || node.getAttribute('title') || node.name || "";
                    }

                    if (tag === 'select') {
                        let options = Array.from(node.options).map(opt => opt.text).join(", ");
                        let val = node.options[node.selectedIndex]?.text || "";
                        return ` [${currentId}] SELECT (value: "${val}", options: [${options}]) `;
                    }
                    else if (tag === 'input' || tag === 'textarea') {
                        let type = (node.type || "text").toLowerCase();
                        let val = node.value || node.placeholder || label || "";
                        if (['submit', 'button', 'reset', 'image'].includes(type)) {
                            val = val || "Submit";
                            return ` [${currentId}] BUTTON (${val}) `;
                        } else if (type === "checkbox" || type === "radio") {
                            let checked = node.checked ? "[x]" : "[ ]";
                            return ` [${currentId}] ${checked} ${type} ${val} `;
                        } else if (type === "range") {
                            let min = node.min || "0";
                            let max = node.max || "100";
                            return ` [${currentId}] SLIDER (value: "${val}", min: "${min}", max: "${max}") `;
                        } else {
                            return ` [${currentId}] INPUT_FIELD (value: "${val}") `;
                        }
                    }
                    else if (isRichText) {
                        let val = node.textContent.substring(0, 1000).replace(/\s+/g, ' ').trim() || label || "";
                        if (node.textContent.length > 1000) val += "...";

                        if (combinedText.match(/\[\d+\]/)) {
                            return ` [${currentId}] RICH_TEXT_EDITOR (value: "${val}") ${combinedText}`;
                        }
                        return ` [${currentId}] RICH_TEXT_EDITOR (value: "${val}") `;
                    }
                    else if (currentRole === 'link') {
                        return ` [${currentId}] LINK (${label || "Unlabeled"}) `;
                    }
                    else if (currentRole === 'image') {
                        if (isInsideRichText) {
                            return ` [${currentId}] RICH_TEXT_IMAGE (${label || "Unlabeled"}) `;
                        }
                        return ` [${currentId}] IMAGE (${label || "Unlabeled"}) `;
                    }
                    // 4. Explicitly format the Video output for the LLM
                    else if (currentRole === 'video') {
                        let paused = node.paused ? "Paused" : "Playing";
                        let time = Math.floor(node.currentTime || 0);
                        let duration = Math.floor(node.duration || 0);
                        return ` [${currentId}] VIDEO (${paused}, ${time}s / ${duration}s) `;
                    }
                    else {
                        return ` [${currentId}] BUTTON (${label || "Unlabeled"}) `;
                    }
                } catch (e) {}
            }

            const blockTags = ['div', 'p', 'h1', 'h2', 'h3', 'h4', 'ul', 'li', 'nav', 'header', 'footer', 'section', 'tr', 'td'];
            if (blockTags.includes(tag)) {
                let clean = combinedText.trim();
                if (!clean) return '';

                if (['h1', 'h2', 'h3'].includes(tag)) {
                    return `\n# ${clean}\n`;
                }
                return `\n${clean}\n`;
            }

            return combinedText;
        };

        if (!document.body) return "Page not fully loaded.";
        let rawText = processNode(document.body);
        return rawText.replace(/\n\s*\n/g, '\n').trim();
    }""")

    return agent_text


async def _get_locator(page: Page, element_id: int):
    """Finds an element across the main page and all iframes."""
    selector = f'[data-agent-id="{element_id}"]'
    for frame in page.frames:
        loc = frame.locator(selector).first
        try:
            if await loc.count() > 0:
                return loc
        except Exception:
            pass
    return page.locator(selector).first


async def click_element(page: Page, element_id: int) -> str:
    """
    Clicks an interactive element on the page using its ID.
    """
    locator = await _get_locator(page, element_id)

    try:
        # Standard click: waits for element to be visible, stable, and receive events.
        await locator.click(timeout=3000)
        return f"Success: Clicked element {element_id}."

    except TimeoutError:
        try:
            # Fallback: Modern UIs often have invisible overlapping divs that block clicks.
            # force=True bypasses Playwright's actionability checks and clicks the exact X/Y coordinate.
            await locator.click(force=True, timeout=3000)
            return f"Success: Force-clicked element {element_id} (standard click was blocked)."
        except Exception as e:
            return f"Error: Could not click element {element_id}. Details: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


async def fill_input(
    page: Page,
    element_id: int,
    text: str,
    press_enter: bool = False,
    overwrite: bool = True,
) -> str:
    """
    Clears an input field or textarea and types the provided text into it.
    Overwrite clears the field. If it's off, the content is appended.
    """
    # google docs text not working.
    locator = await _get_locator(page, element_id)

    try:
        is_rich_text = await locator.evaluate(
            "(el) => el.isContentEditable || el.getAttribute('role') === 'textbox'"
        )

        if is_rich_text:
            await locator.click(timeout=3000)
            if overwrite:
                await page.keyboard.press("Control+A")
                await page.keyboard.press("Meta+A")
                await page.keyboard.press("Backspace")
            else:
                await page.keyboard.press("Control+A")
                await page.keyboard.press("Meta+A")
                await page.keyboard.press("ArrowRight")
            await page.keyboard.type(text, delay=5)
        else:
            if overwrite:
                # .fill() automatically clears the existing text first
                await locator.fill(text, timeout=3000)
            else:
                await locator.press_sequentially(text, timeout=3000)

        if press_enter:
            await locator.press("Enter")
            return (
                f"Success: Typed '{text}' into element {element_id} and pressed Enter."
            )

        return f"Success: Typed '{text}' into element {element_id}."

    except Exception as e:
        return f"Error: Failed to type into element {element_id}. Details: {str(e)}"


async def extract_attribute(page: Page, element_id: int, attribute: str) -> str:
    """
    Extracts a specific HTML attribute from an element (e.g., 'src' for images, 'href' for links).
    """
    locator = await _get_locator(page, element_id)

    try:
        value = await locator.get_attribute(attribute, timeout=3000)
        if value is None:
            return (
                f"Error: Element {element_id} does not have a '{attribute}' attribute."
            )

        return f"Success: Element {element_id} '{attribute}' is: {value}"

    except Exception as e:
        return (
            f"Error: Failed to extract attribute from {element_id}. Details: {str(e)}"
        )


async def press_keyboard_key(page: Page, key: str) -> str:
    """
    Presses a specific keyboard key (e.g., 'Escape', 'Enter', 'Tab', 'ArrowDown').
    """
    # Either multiple tool calls or a sequence input
    try:
        await page.keyboard.press(key)
        return f"Success: Pressed the '{key}' key."
    except Exception as e:
        return f"Error: Failed to press key '{key}'. Details: {str(e)}"


async def scroll_page(page: Page, direction: str = "down") -> str:
    """
    Scrolls the active page up or down.
    """
    try:
        if direction.lower() == "down":
            await page.mouse.wheel(0, 800)
        else:
            await page.mouse.wheel(0, -800)

        # Give lazy-loaded UI elements a moment to render after scrolling
        await asyncio.sleep(0.5)
        return f"Success: Scrolled {direction}."
    except Exception as e:
        return f"Error: Failed to scroll. Details: {str(e)}"


async def hover_element(page: Page, element_id: int) -> str:
    """
    Hovers the mouse over an element. Useful for opening CSS-based dropdown menus.
    """
    locator = await _get_locator(page, element_id)

    try:
        await locator.hover(timeout=3000)
        # Wait a moment for the hover animation/dropdown to appear
        await asyncio.sleep(0.5)
        return f"Success: Hovered over element {element_id}."
    except Exception as e:
        return f"Error: Failed to hover over element {element_id}. Details: {str(e)}"


async def execute_misc_action(page: Page, element_id: int, action_event: str) -> str:
    """
    Executes miscellaneous actions or dispatches raw HTML/DOM events on an element.

    Valid action_event strings include:
    - Native pseudo-events: 'dblclick', 'rightclick', 'focus', 'blur', 'check', 'uncheck'
    - Raw DOM events: 'submit', 'mouseenter', 'mouseleave', 'change', 'input', 'keydown'
    """
    locator = await _get_locator(page, element_id)

    try:
        # Normalize the LLM's string
        action = action_event.lower().strip()

        # 1. Handle common UI actions with Playwright's native methods first
        if action in ["dblclick", "doubleclick", "double_click"]:
            await locator.dblclick(timeout=3000)
            return f"Success: Double-clicked element {element_id}."

        elif action in ["rightclick", "right_click", "contextmenu"]:
            await locator.click(button="right", timeout=3000)
            return f"Success: Right-clicked element {element_id}."

        elif action == "focus":
            await locator.focus(timeout=3000)
            return f"Success: Focused element {element_id}."

        elif action == "blur":
            await locator.blur(timeout=3000)
            return f"Success: Blurred (removed focus from) element {element_id}."

        elif action == "check":
            await locator.check(timeout=3000)
            return f"Success: Checked (toggled on) element {element_id}."

        elif action == "uncheck":
            await locator.uncheck(timeout=3000)
            return f"Success: Unchecked (toggled off) element {element_id}."

        # 2. The Catch-All: Dispatch raw DOM events directly into the browser engine
        else:
            # This handles 'submit', 'mouseover', 'change', 'dragstart', etc.
            await locator.dispatch_event(action, timeout=3000)

            # Wait a tiny bit in case the DOM event triggers an animation or network request
            await asyncio.sleep(0.5)

            return f"Success: Dispatched raw DOM event '{action_event}' on element {element_id}."

    except Exception as e:
        return f"Error: Failed to execute '{action_event}' on element {element_id}. Details: {str(e)}"


async def select_combo_option(page: Page, element_id: int, option_value: str) -> str:
    """Selects an option in a combo box (<select> element)."""
    locator = await _get_locator(page, element_id)
    try:
        await locator.select_option(label=option_value, timeout=3000)
        return f"Success: Selected option '{option_value}' in element {element_id}."
    except Exception:
        try:
            # Fallback to selecting by value
            await locator.select_option(value=option_value, timeout=3000)
            return f"Success: Selected option '{option_value}' in element {element_id}."
        except Exception as e:
            return f"Error: Failed to select option '{option_value}'. Details: {str(e)}"


async def set_slider(page: Page, element_id: int, value: str) -> str:
    """Sets the value of a slider (<input type="range">)."""
    locator = await _get_locator(page, element_id)
    try:
        await locator.evaluate(
            f"(el) => {{ el.value = '{value}'; el.dispatchEvent(new Event('input', {{ bubbles: true }})); el.dispatchEvent(new Event('change', {{ bubbles: true }})); }}"
        )
        return f"Success: Set slider {element_id} to {value}."
    except Exception as e:
        return f"Error: Failed to set slider {element_id}. Details: {str(e)}"


async def take_page_screenshot(page: Page) -> dict[str, Any] | str:
    """Takes a screenshot of the entire page and returns it as a dict or error string."""
    try:
        screenshot_bytes = await page.screenshot(
            type="jpeg", quality=80, full_page=True
        )
        b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        }
    except Exception as e:
        return f"Error: Failed to take page screenshot. Details: {str(e)}"


async def take_element_screenshot(page: Page, element_id: int) -> dict[str, Any] | str:
    """Takes a screenshot of a specific element and returns it as a dict or error string."""
    locator = await _get_locator(page, element_id)
    try:
        screenshot_bytes = await locator.screenshot(type="jpeg", quality=80)
        b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        }
    except Exception as e:
        return f"Error: Failed to take screenshot of element {element_id}. Details: {str(e)}"


async def control_video(
    page: Page, element_id: int, action: str, value: float | None = None
) -> str:
    """Controls a video element (play, pause, seek, mute, unmute)."""
    locator = await _get_locator(page, element_id)
    action = action.lower()
    try:
        if action == "play":
            await locator.evaluate("(el) => el.play()")
            return f"Success: Played video {element_id}."
        elif action == "pause":
            await locator.evaluate("(el) => el.pause()")
            return f"Success: Paused video {element_id}."
        elif action == "seek" and value is not None:
            await locator.evaluate(f"(el) => el.currentTime = {value}")
            return f"Success: Seeked video {element_id} to {value}s."
        elif action == "mute":
            await locator.evaluate("(el) => el.muted = true")
            return f"Success: Muted video {element_id}."
        elif action == "unmute":
            await locator.evaluate("(el) => el.muted = false")
            return f"Success: Unmuted video {element_id}."
        else:
            return f"Error: Unknown video action '{action}' or missing value."
    except Exception as e:
        return f"Error: Failed to execute '{action}' on video {element_id}. Details: {str(e)}"
