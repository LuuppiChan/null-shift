# Contains the actual browser tool implementations.
# The tool definitions don't live here.
import asyncio
import base64
import weakref
from datetime import datetime
from typing import Any, Literal

import cv2
import numpy as np
from playwright.async_api import Locator, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from tools.browser.config import manager


def hide_sequences(text: str) -> str:
    """Hide possibly sensitive strings based on the config."""
    for s in manager.get_config().replace_b64_strings:
        try:
            text = text.replace(
                base64.b64decode(s[0]).decode(errors="replace"),
                base64.b64decode(s[1]).decode(errors="replace"),
            )
        except IndexError:
            pass
    return text


def compress_image_buffer(base64_str: str, quality: int = 50) -> str:
    """base64 image buffer."""
    img_bytes = base64.b64decode(base64_str)

    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    assert img is not None, "Failed to encode base64 image"
    success, compressed_img = cv2.imencode(
        ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    assert success, "Failed to compress base64 image"
    compressed_b64 = base64.b64encode(compressed_img).decode("utf-8")
    return compressed_b64


async def get_agent_dom(page: Page):
    # Custom Text Extraction for Google Docs
    is_google_docs = "docs.google.com/document/" in page.url
    docs_text = ""
    if is_google_docs:
        try:
            docs_text = await page.evaluate(r"""() => {
                let textBlocks = [];
                // Look for common Google Docs line/paragraph classes
                let nodes = document.querySelectorAll('.kix-lineview, .kix-paragraphrenderer, .kix-wordhtmlgenerator-word-node, [role="paragraph"]');
                if (nodes.length > 0) {
                    nodes.forEach(node => {
                        let textContent = node.innerText || node.textContent;
                        if (textContent && textContent.trim()) {
                            textBlocks.push(textContent.replace(/\u200C/g, ''));
                        }
                    });
                }
                return textBlocks.join('\n');
            }""")

            # If the DOM is completely canvas rendered, fallback to simulating a copy operation
            if not docs_text.strip():
                # Focus the editor canvas
                editor = page.locator(".kix-appview-editor").first
                if await editor.count() > 0:
                    await editor.click(timeout=3000)
                    await page.keyboard.press("Control+A")
                    await page.keyboard.press("Meta+A")  # For mac
                    await page.keyboard.press("Control+C")
                    await page.keyboard.press("Meta+C")

                    # Unselect
                    await page.keyboard.press("ArrowRight")

                    # Read clipboard
                    # Note: this requires clipboard permissions in the browser context
                    docs_text = await page.evaluate("navigator.clipboard.readText()")

        except Exception as e:
            docs_text = f"Error reading Google Docs content: {str(e)}"

        if not docs_text.strip():
            docs_text = "(Document appears empty or text is entirely canvas-rendered and unreadable)"

    agent_text: str = await page.evaluate(r"""() => {
        document.querySelectorAll('[data-agent-id]').forEach(el => el.removeAttribute('data-agent-id'));
        const activeEl = document.activeElement;

        let idCounter = 1;

        const isReactInteractive = (node) => {
            const propsKey = Object.keys(node).find(k => k.startsWith('__reactProps$'));
            if (!propsKey) return false;
            const p = node[propsKey];
            // Only direct click handlers. onKeyDown-only = widget wrapper, onSubmit = form
            // container (tagging those would suppress IDs of everything inside).
            const meaningful = ['onClick', 'onMouseDown'];
            if (!meaningful.some(k => typeof p?.[k] === 'function')) return false;
            // Skip huge containers that happen to carry onClick (whole-panel wrappers);
            // real controls have short labels.
            const text = (node.innerText || '').replace(/\s+/g, ' ').trim();
            return text.length <= 300;
        };

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
            if (['script', 'style', 'noscript', 'svg', 'canvas'].includes(tag)) {
                // SPECIAL CASE: Treat Google Docs Editor as interactable
                if (node.classList && node.classList.contains('kix-appview-editor')) {
                   // do not return ''
                } else {
                    return '';
                }
            }

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
            let isDocsEditor = false;
            try {
                // 3. Tell the script that <video> is a targetable, interactive element
                if (node.classList && node.classList.contains('kix-appview-editor')) {
                    isInt = true;
                    isDocsEditor = true;
                }
                else if (['a', 'button', 'input', 'select', 'textarea', 'img', 'video'].includes(tag)) isInt = true;
                else if (node.hasAttribute('onclick') || node.getAttribute('role') === 'button') isInt = true;
                // React detection: handlers live in __reactProps$* expando properties, not DOM attributes
                else if (isReactInteractive(node)) isInt = true;
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
                    const childInteractive = Array.from(node.children || []).some(c => {
                        const pk = Object.keys(c).find(k => k.startsWith('__reactProps$'));
                        if (!pk) return false;
                        const p = c[pk];
                        return typeof p?.onClick === 'function' || typeof p?.onMouseDown === 'function';
                    });
                    if (childInteractive) isInt = false; // recurse into children instead
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
                        let frameSrc = node.src || node.getAttribute('src') || '';
                        if (frameSrc.includes('google.com/recaptcha')) {
                            let captchaId = idCounter++;
                            node.setAttribute('data-agent-id', captchaId);
                            childrenText.push(` [${captchaId}] BUTTON (reCAPTCHA checkbox) `);
                        } else {
                            childrenText.push(" [IFRAME: Cross-Origin Restricted] ");
                        }
                    }
                } catch (e) {
                    let frameSrc = node.src || node.getAttribute('src') || '';
                    if (frameSrc.includes('google.com/recaptcha')) {
                        let captchaId = idCounter++;
                        node.setAttribute('data-agent-id', captchaId);
                        childrenText.push(` [${captchaId}] BUTTON (reCAPTCHA checkbox) `);
                    } else {
                        childrenText.push(" [IFRAME: Cross-Origin Restricted] ");
                    }
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
                    const render = () => {
                        let currentId = idCounter++;
                        node.setAttribute('data-agent-id', currentId);

                        if (isDocsEditor) {
                             return ` [${currentId}] GOOGLE_DOCS_EDITOR `;
                        }

                        let label = combinedText.trim();
                        if (!label) {
                            label = node.getAttribute('aria-label') || node.getAttribute('title') || node.name || "";
                        }

                        if (tag === 'select') {
                            let options = Array.from(node.options).map(opt => opt.text).join(", ");
                            let val = node.options[node.selectedIndex]?.text || "";
                            return ` [${currentId}] SELECT (value: "${val}", options: [${options}]) `;
                        }
                        else if (
                            node.getAttribute('role') === 'combobox' ||
                            ['listbox', 'menu', 'tree', 'grid', 'dialog', 'true'].includes(node.getAttribute('aria-haspopup')) ||
                            (typeof node.className === 'string' &&
                             /select__control|combobox|autocomplete|dropdown/i.test(node.className))
                        ) {
                            let expanded = node.getAttribute('aria-expanded');

                            // Collect options
                            let options = [];
                            try {
                                // react-select: options live in the component's props, climb the fiber tree
                                let fk = Object.keys(node).find(k => k.startsWith('__reactFiber$'));
                                if (fk) {
                                    let flatten = (arr) => arr.forEach(o => {
                                        if (o && Array.isArray(o.options)) flatten(o.options); // grouped options
                                        else {
                                            let text = (typeof o === 'string' ? o : String(o?.label ?? o?.value ?? ""))
                                                .replace(/\s+/g, " ").trim();
                                            options.push('"' + text + '"');
                                        }
                                    });
                                    for (let f = node[fk]; f; f = f.return) {
                                        let p = f.memoizedProps;
                                        if (p && Array.isArray(p.options) && p.options.length &&
                                            p.options.every(o => o && (typeof o === 'string' || o.label !== undefined || o.value !== undefined))) {
                                            flatten(p.options);
                                            break;
                                        }
                                    }
                                }
                            } catch (e) {}
                            // Fallback: options actually rendered in the DOM (open menus / custom dropdowns)
                            if (!options.length && expanded === 'true') {
                                let listboxId = node.getAttribute('aria-owns') || node.getAttribute('aria-controls');
                                let opts = listboxId
                                    ? document.querySelectorAll(`#${CSS.escape(listboxId)} [role="option"]`)
                                    : document.querySelectorAll('[role="listbox"] [role="option"]');
                                options = Array.from(opts).map(o => '"' + o.textContent.replace(/\s+/g, " ").trim() + '"');
                            }

                            // Current value: react-select renders it as single-value or placeholder
                            let val = (node.querySelector('[class*="singleValue"], [class*="placeholder"]')?.textContent
                                || node.textContent).replace(/\s+/g, " ").trim();

                            let optsStr = options.length ? `, options: [${options.join(", ")}]` : "";
                            let expStr = expanded !== null ? ` [${expanded === 'true' ? 'open' : 'closed'}]` : "";
                            return ` [${currentId}] DROPDOWN (${val}${expStr}${optsStr}) `;
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
                    };
                    let rendered = render();
                    if (node === activeEl) {
                        rendered = rendered.replace(/\[(\d+)\]/, '[$1] FOCUSED:');
                    }
                    return rendered;
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
    agent_text = hide_sequences(agent_text)

    if is_google_docs:
        return f"{agent_text}\n\n[Google Docs Content]\n{docs_text}\n[/Google Docs Content]"
    return agent_text


async def _get_locator(page: Page, element_id: int):
    """Finds an element across the main page and all iframes."""
    selector = f'[data-agent-id="{element_id}"]'
    for frame in page.frames:
        loc = frame.locator(selector)
        try:
            n = await loc.count()
        except Exception:
            continue
        if n == 0:
            continue
        for i in range(n):
            candidate = loc.nth(i)
            try:
                if await candidate.is_visible():
                    return candidate
            except Exception:
                continue
        return loc.first  # fallback: all matches hidden
    return page.locator(selector).first


async def page_javascript(page: Page, expression: str) -> str:
    """
    Run javascript on the current page.
    Args:
        expression: JavaScript expression to be evaluated in the browser context. If the expression evaluates to a function, the
    """
    return await page.evaluate(expression)


# Cursor overlay: Playwright doesn't draw the OS cursor, so a fixed marker is
# injected into the page and moved to wherever the tools act. Cosmetic only:
# failures here are swallowed so they never break a tool action.

CURSOR_JS: str = """
() => {
  // Recreate if an older overlay is left over without the show/hide helpers (e.g. after a server restart).
  if (document.getElementById('__orion_cursor') && window.__orionShow) return false;
  document.getElementById('__orion_cursor')?.remove();
  const el = document.createElement('div');
  el.id = '__orion_cursor';
  el.setAttribute('aria-hidden', 'true');
  Object.assign(el.style, {
    position: 'fixed', left: '0px', top: '0px',
    width: '18px', height: '18px',
    marginLeft: '-9px', marginTop: '-9px',
    borderRadius: '50%',
    border: '2px solid #ff4d4d',
    background: 'rgba(255, 77, 77, 0.3)',
    boxSizing: 'border-box',
    pointerEvents: 'none',
    zIndex: '2147483647',
    opacity: '0',
    transform: 'scale(0.5)',
  });
  document.documentElement.appendChild(el);

  // Runs one animation, then pins the element to finalStyle. Returns false if a newer
  // animation cancelled this one; the newer call then owns the final state.
  const run = async (frames, options, finalStyle) => {
    el.getAnimations().forEach((a) => a.cancel());
    const anim = el.animate(frames, { ...options, fill: 'forwards' });
    try {
      await anim.finished;
    } catch (e) {
      return false;
    }
    Object.assign(el.style, finalStyle);
    anim.cancel();
    return true;
  };

  window.__orionShow = async (opts) => {
    const { fromX, fromY, toX, toY, idleMs, appearMs, glideMs, disappearMs } = opts;
    clearTimeout(window.__orionCursorTimer);
    if (window.__orionCursorVisible) {
      // Already up: force full opacity so a glide that cancels a running appear can't leave it invisible.
      Object.assign(el.style, { opacity: '1', transform: 'scale(1)' });
    } else {
      window.__orionCursorVisible = true;
      el.style.left = fromX + 'px';
      el.style.top = fromY + 'px';
      const appeared = await run(
        [{ opacity: '0', transform: 'scale(0.5)' }, { opacity: '1', transform: 'scale(1)' }],
        { duration: appearMs, easing: 'ease-out' },
        { opacity: '1', transform: 'scale(1)' }
      );
      if (!appeared) return false;
    }
    const cs = getComputedStyle(el);
    const sx = parseFloat(cs.left);
    const sy = parseFloat(cs.top);
    if (Math.abs(sx - toX) > 0.5 || Math.abs(sy - toY) > 0.5) {
      const glided = await run(
        [{ left: sx + 'px', top: sy + 'px' }, { left: toX + 'px', top: toY + 'px' }],
        { duration: glideMs, easing: 'ease-in-out' },
        { left: toX + 'px', top: toY + 'px' }
      );
      if (!glided) return false;
    }
    if (idleMs !== null && idleMs !== undefined) {
      window.__orionCursorTimer = setTimeout(() => window.__orionHide(disappearMs), idleMs);
    }
    return true;
  };

  window.__orionHide = async (disappearMs) => {
    clearTimeout(window.__orionCursorTimer);
    if (!window.__orionCursorVisible) return false;
    window.__orionCursorVisible = false;
    return run(
      [{ opacity: '1', transform: 'scale(1)' }, { opacity: '0', transform: 'scale(0.6)' }],
      { duration: disappearMs, easing: 'ease-in' },
      { opacity: '0', transform: 'scale(0.6)' }
    );
  };
  return true;
}
"""

# Timings in milliseconds. The animations run in-page via the Web Animations API,
# so Python never sleeps for them.
CURSOR_VISIBLE_MS: int = 1500  # idle time before the overlay fades out
CURSOR_APPEAR_MS: int = 150
CURSOR_GLIDE_MS: int = 200
CURSOR_DISAPPEAR_MS: int = 150

SHOW_JS: str = "(opts) => window.__orionShow?.(opts)"
HIDE_JS: str = "(ms) => window.__orionHide?.(ms)"

PULSE_JS: str = """
() => {
  const el = document.getElementById('__orion_cursor');
  if (!el) return;
  el.animate(
    [{ transform: 'scale(1)' }, { transform: 'scale(1.6)', opacity: 0.4 }, { transform: 'scale(1)' }],
    { duration: 300 }
  );
}
"""

REMOVE_JS: str = "() => document.getElementById('__orion_cursor')?.remove()"

INSPECT_JS: str = """
([x, y]) => {
  const target = document.elementFromPoint(x, y);
  if (!target) return null;
  const el = target.closest('[data-agent-id]') ?? target;
  return {
    tag: el.tagName.toLowerCase(),
    agent_id: el.getAttribute('data-agent-id'),
    text: (el.innerText || el.getAttribute('aria-label') || '').trim().slice(0, 80),
  };
}
"""

# Last known cursor position per page. Forgotten when the overlay is recreated
# (i.e. after a navigation), so stale positions are never reported.
_cursor_positions: weakref.WeakKeyDictionary[Page, tuple[float, float]] = (
    weakref.WeakKeyDictionary()
)


async def ensure_cursor(page: Page) -> None:
    """Injects the overlay if missing. Forgets the old position if it had to recreate it."""
    created: bool = await page.evaluate(CURSOR_JS)
    if created:
        _cursor_positions.pop(page, None)


async def show_cursor(
    page: Page, x: float, y: float, idle_ms: int | None = CURSOR_VISIBLE_MS
) -> None:
    """
    Shows the overlay at viewport coordinates (x, y). If it's already placed on this
    page, it glides from the last position. Pass idle_ms=None to keep it up until
    hide_cursor is called. Never raises.
    """
    try:
        await ensure_cursor(
            page
        )  # may forget the old position if it had to recreate the overlay
        last: tuple[float, float] | None = _cursor_positions.get(page)
        from_x, from_y = last if last is not None else (x, y)
        await page.evaluate(
            SHOW_JS,
            {
                "fromX": from_x,
                "fromY": from_y,
                "toX": x,
                "toY": y,
                "idleMs": idle_ms,
                "appearMs": CURSOR_APPEAR_MS,
                "glideMs": CURSOR_GLIDE_MS,
                "disappearMs": CURSOR_DISAPPEAR_MS,
            },
        )
        _cursor_positions[page] = (x, y)
    except Exception:
        pass


async def hide_cursor(page: Page) -> None:
    """Fades the overlay out. Never raises."""
    try:
        await page.evaluate(HIDE_JS, CURSOR_DISAPPEAR_MS)
    except Exception:
        pass


async def move_cursor(page: Page, x: float, y: float) -> None:
    """Moves the overlay to viewport coordinates (x, y) with the full appear/glide/idle sequence."""
    await show_cursor(page, x, y)


async def move_cursor_to_locator(page: Page, locator: Locator) -> None:
    """Moves the overlay to the centre of an element's bounding box. Never raises."""
    try:
        box: dict[str, float] | None = await locator.bounding_box()
        if box is None:
            return
        await move_cursor(
            page, box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        )
    except Exception:
        pass


async def pulse_cursor(page: Page) -> None:
    """Short visual pulse to mark a click. Never raises."""
    try:
        await page.evaluate(PULSE_JS)
    except Exception:
        pass


async def remove_cursor(page: Page) -> None:
    """Removes the overlay from the page. Never raises."""
    try:
        await page.evaluate(REMOVE_JS)
    except Exception:
        pass


async def get_cursor_info(page: Page) -> dict[str, Any] | None:
    """Returns the cursor position and the element under it, or None if unavailable."""
    pos: tuple[float, float] | None = _cursor_positions.get(page)
    if pos is None:
        return None
    x, y = pos
    try:
        element: dict[str, Any] | None = await page.evaluate(INSPECT_JS, [x, y])
    except Exception:
        return None
    return {"x": round(x), "y": round(y), "element": element}


def format_cursor_info(info: dict[str, Any] | None) -> str:
    """One-line summary for tool results."""
    if info is None:
        return "Cursor: not placed yet."
    el: dict[str, Any] | None = info["element"]
    if el is None:
        return f"Cursor at ({info['x']}, {info['y']}), no element under it."
    agent: str = f" [{el['agent_id']}]" if el["agent_id"] else ""
    label: str = f" '{el['text']}'" if el["text"] else ""
    return f"Cursor at ({info['x']}, {info['y']}) over <{el['tag']}>{agent}{label}."


async def cursor_action(
    page: Page,
    action: str | Literal["move", "click", "info"],
    element_id: int | None = None,
    x: float | None = None,
    y: float | None = None,
) -> str:
    """
    Moves the agent cursor overlay, clicks at viewport coordinates, or reports where it is.
    action="move": moves to element_id (if given) or to viewport coordinates x, y.
    action="click": moves to viewport coordinates x, y and clicks there. Use for targets that are not exposed in the DOM.
    action="info": reports the cursor position and the element under it.
    """
    if action == "info":
        return format_cursor_info(await get_cursor_info(page))
    if action == "click":
        if x is None or y is None:
            return "Error: Click needs both x and y (viewport coordinates)."
        await move_cursor(page, x, y)
        # Capture what's under the cursor before clicking, so the result shows the real target.
        before: str = format_cursor_info(await get_cursor_info(page))
        try:
            # Raw mouse event at the coordinate: works on elements that have no DOM ID (canvas, shadow DOM, etc.).
            await page.mouse.click(x, y)
        except Exception as e:
            return f"Error: Could not click at ({x}, {y}). Details: {str(e)}"
        await pulse_cursor(page)
        return f"Success: Clicked at ({x}, {y}). {before}"
    if action != "move":
        return (
            f"Error: Unknown cursor action '{action}'. Use 'move', 'click' or 'info'."
        )
    if element_id is not None:
        try:
            locator = await _get_locator(page, element_id)
        except Exception as e:
            return f"Error: Could not find element {element_id}. Details: {str(e)}"
        await move_cursor_to_locator(page, locator)
    elif x is not None and y is not None:
        await move_cursor(page, x, y)
    else:
        return "Error: Move needs element_id or both x and y."
    return f"Success: Cursor moved. {format_cursor_info(await get_cursor_info(page))}"


async def click_element(page: Page, element_id: int) -> str:
    """
    Clicks an interactive element on the page using its ID.
    """
    locator = await _get_locator(page, element_id)
    await move_cursor_to_locator(page, locator)
    # Capture what's under the cursor before clicking, so the result shows the real target.
    before: str = format_cursor_info(await get_cursor_info(page))

    try:
        # Standard click: waits for element to be visible, stable, and receive events.
        await locator.click(timeout=3000)
        await pulse_cursor(page)
        return f"Success: Clicked element {element_id}. {before}"

    except PlaywrightTimeoutError:
        try:
            # Fallback: Modern UIs often have invisible overlapping divs that block clicks.
            # force=True bypasses Playwright's actionability checks and clicks the exact X/Y coordinate.
            await locator.click(force=True, timeout=3000)
            await pulse_cursor(page)
            return f"Success: Force-clicked element {element_id} (standard click was blocked). {before}"
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
    is_google_docs = "docs.google.com/document/" in page.url
    locator = await _get_locator(page, element_id)
    config = manager.get_config()

    try:
        if is_google_docs:
            await locator.click(timeout=3000)
            if overwrite:
                await page.keyboard.press("Control+A")
                await page.keyboard.press("Meta+A")
                await page.keyboard.press("Backspace")
            else:
                await page.keyboard.press("Control+A")
                await page.keyboard.press("Meta+A")
                await page.keyboard.press("ArrowRight")
            await page.keyboard.type(text, delay=config.typing_delay_ms)
        else:
            is_rich_text = await locator.evaluate(
                "(el) => el.isContentEditable || el.getAttribute('role') === 'textbox'"
            )

            if is_rich_text:
                await locator.click(timeout=config.typing_timeout_ms)
                if overwrite:
                    await page.keyboard.press("Control+A")
                    await page.keyboard.press("Meta+A")
                    await page.keyboard.press("Backspace")
                else:
                    await page.keyboard.press("Control+A")
                    await page.keyboard.press("Meta+A")
                    await page.keyboard.press("ArrowRight")
                await page.keyboard.type(text, delay=config.typing_delay_ms)
            else:
                if overwrite:
                    # .fill() automatically clears the existing text first
                    await locator.fill(text, timeout=config.typing_timeout_ms)
                else:
                    await locator.press_sequentially(
                        text,
                        timeout=config.typing_timeout_ms,
                        delay=config.typing_delay_ms,
                    )

        if press_enter:
            if is_google_docs:
                await page.keyboard.press("Enter")
            else:
                await locator.press("Enter")
            return f"Success: Typed given text into element {element_id} and pressed Enter."

        return f"Success: Typed given text into element {element_id}."

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


async def press_keyboard_keys(page: Page, keys: list[str]) -> str:
    """
    Presses a specific keyboard key (e.g., 'Escape', 'Enter', 'Tab', 'ArrowDown').
    """
    press_delay = manager.get_config().typing_delay_ms / 1000
    for key in keys:
        try:
            start = datetime.now()
            await page.keyboard.press(key)
            needs_wait = (datetime.now() - start).total_seconds() - press_delay
            if needs_wait > 0:
                await asyncio.sleep(needs_wait)
        except Exception as e:
            return f"Error: Failed to press key '{key}'. Details: {str(e)}"
    return "Success: Pressed the keys."


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
    await move_cursor_to_locator(page, locator)

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

    # Native <select>: use Playwright's native support
    try:
        tag = await locator.evaluate("el => el.tagName.toLowerCase()", timeout=3000)
        if tag == "select":
            try:
                await locator.select_option(label=option_value, timeout=3000)
                return f"Success: Selected option '{option_value}' in element {element_id}."
            except Exception:
                await locator.select_option(value=option_value, timeout=3000)
                return f"Success: Selected option '{option_value}' in element {element_id}."
    except Exception:
        pass

    # Custom/React dropdown (e.g. react-select): open the menu, then click the option
    try:
        await locator.click(timeout=3000)
        await asyncio.sleep(0.4)  # wait for the menu to render

        # Deepest exact-text match first: clicking a child still bubbles up to
        # the option's onClick. role/class based matches as fallbacks.
        candidates = [
            page.get_by_text(option_value, exact=True),
            page.locator('[role="option"]', has_text=option_value),
            page.locator('div[class*="option"]', has_text=option_value),
        ]
        for cand in candidates:
            try:
                await cand.first.click(timeout=2000)
                return f"Success: Selected option '{option_value}' in element {element_id} (custom dropdown)."
            except Exception:
                continue

        # Option not found: close the menu so the page isn't left in a weird state
        await page.keyboard.press("Escape")
        return f"Error: Option '{option_value}' not found in the open menu of element {element_id}."
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


async def take_page_screenshot(page: Page) -> list[dict[str, Any]] | str:
    """Takes a screenshot of the entire page and returns it as a dict or error string."""
    # Show the overlay at its last position with no idle timer, so it can't fade mid-capture
    # on a slow full-page shot. It's hidden explicitly afterwards.
    last: tuple[float, float] | None = None
    try:
        # Forgets the position if the overlay was recreated (e.g. after navigation).
        await ensure_cursor(page)
        last = _cursor_positions.get(page)
    except Exception:
        pass
    if last is not None:
        await show_cursor(page, last[0], last[1], idle_ms=None)
    try:
        screenshot_bytes = await page.screenshot(
            type="jpeg", quality=80, full_page=True
        )
        b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        b64 = compress_image_buffer(b64)
        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        ]
    except Exception as e:
        return f"Error: Failed to take page screenshot. Details: {str(e)}"
    finally:
        if last is not None:
            await hide_cursor(page)


async def take_element_screenshot(
    page: Page, element_id: int
) -> list[dict[str, Any]] | str:
    """Takes a screenshot of a specific element and returns it as a dict or error string."""
    locator = await _get_locator(page, element_id)
    try:
        screenshot_bytes = await locator.screenshot(type="jpeg", quality=80)
        b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        b64 = compress_image_buffer(b64)
        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        ]
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
