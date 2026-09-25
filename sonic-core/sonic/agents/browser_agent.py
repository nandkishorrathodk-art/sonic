"""
SONIC-REDA — Browser Vision Agent (Playwright + Screenshot Proofs)
====================================================================
Autonomous browser interaction agent using Playwright for:
    - Rendering SPAs (React, Vue, Angular) and extracting dynamic DOM
    - Capturing screenshot proof for XSS, UI bugs, and visual vulnerabilities
    - Clicking, typing, navigating, and waiting for dynamic content
    - Extracting hidden API routes from JavaScript bundles
"""

from __future__ import annotations

import base64
import inspect
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from sonic.logger import get_logger
from sonic.net.tls import tls_verify
from sonic.sandbox.egress import is_target_allowed

logger = get_logger(__name__)


@dataclass
class PageSnapshot:
    """Captured state of a web page."""
    url: str
    title: str
    status_code: int
    html_content: str
    screenshot_b64: str  # Base64 PNG
    cookies: list[dict[str, Any]] = field(default_factory=list)
    console_logs: list[str] = field(default_factory=list)
    network_requests: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class DOMElement:
    """An interactive element found in the DOM."""
    tag: str
    text: str
    selector: str
    attributes: dict[str, str] = field(default_factory=dict)
    is_visible: bool = True
    bbox: tuple[int, int, int, int] | None = None


class BrowserAgent:
    """
    Autonomous browser controller for visual testing and proof capture.
    Uses Playwright when available, falls back to httpx for basic HTTP.
    """

    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30000,
        vault: Any = None,
        *,
        scope_checker: Any | None = None,
        scope_config: dict[str, Any] | None = None,
        allow_javascript: bool = False,
        require_scope: bool = False,
    ):
        self.headless = headless
        self.timeout = timeout
        self.vault = vault
        self._browser = None
        self._context = None
        self._page = None
        self._playwright = None
        self._using_playwright = False
        self._attached_cdp = False
        self.scope_checker = scope_checker
        self.scope_config = scope_config
        self.allow_javascript = allow_javascript
        self.require_scope = require_scope

    def _validate_url(self, url: str) -> None:
        """Reject unsafe browser destinations before any network request."""
        clean = (url or "").strip()
        if clean == "about:blank":
            return
        parsed = urlparse(clean)
        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError("browser navigation only supports http and https URLs")
        if not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("browser navigation requires a host without embedded credentials")
        if self.require_scope and (self.scope_checker is None or not self.scope_config):
            raise ValueError("browser navigation requires an explicit engagement scope")
        if self.scope_checker is not None and self.scope_config:
            if not self.scope_checker.is_target_in_scope(parsed.hostname, self.scope_config):
                raise ValueError("browser URL is outside the configured engagement scope")
        allowed, reason = is_target_allowed(url)
        if not allowed:
            raise ValueError(f"browser URL blocked by egress policy: {reason}")

    async def launch(self) -> bool:
        """Launch or attach to browser instance (prefers visible :99 CDP port 9222 when available)."""
        try:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            engine_name = os.environ.get("SONIC_BROWSER_ENGINE", "chromium")
            browser_engine = getattr(self._playwright, engine_name)
            cdp_url = os.environ.get("SONIC_BROWSER_CDP_URL", "http://127.0.0.1:9222")

            saved_state = None
            if self.vault is not None and hasattr(self.vault, "load_browser_state"):
                try:
                    loaded = self.vault.load_browser_state()
                    if inspect.isawaitable(loaded):
                        loaded = await loaded
                    if loaded:
                        saved_state = loaded
                except Exception as exc:
                    logger.debug("browser_vault_load_failed", error=str(exc))

            ctx_kwargs: dict[str, Any] = {
                "viewport": {"width": 1280, "height": 800},
                "ignore_https_errors": False,
                "accept_downloads": True,
            }
            if saved_state is not None:
                ctx_kwargs["storage_state"] = saved_state

            connected_cdp = False
            if not self.headless or os.environ.get("SONIC_ATTACH_VISIBLE_BROWSER") == "1":
                try:
                    self._browser = await browser_engine.connect_over_cdp(cdp_url, timeout=2500)
                    self._attached_cdp = True
                    contexts = self._browser.contexts
                    self._context = contexts[0] if contexts else await self._browser.new_context(**ctx_kwargs)
                    if contexts and saved_state and isinstance(saved_state, dict) and saved_state.get("cookies"):
                        try:
                            await self._context.add_cookies(saved_state["cookies"])
                        except Exception:
                            pass
                    pages = self._context.pages
                    self._page = pages[0] if pages else await self._context.new_page()
                    connected_cdp = True
                    logger.info("browser_agent_attached_cdp", cdp_url=cdp_url)
                except Exception:
                    self._attached_cdp = False
                    connected_cdp = False

            if not connected_cdp:
                self._attached_cdp = False
                self._browser = await browser_engine.launch(headless=self.headless)
                self._context = await self._browser.new_context(**ctx_kwargs)
                self._page = await self._context.new_page()

            await self._page.route("**/*", self._guard_request)
            self._using_playwright = True
            logger.info("browser_agent_launched", engine="playwright", attached_cdp=connected_cdp)
            return True
        except ImportError:
            logger.warning("playwright_not_installed", fallback="httpx")
            self._using_playwright = False
            return True  # Fall back to httpx mode
        except Exception as e:
            logger.error("browser_launch_failed", error=str(e))
            return False

    async def save_state_to_vault(self) -> None:
        """Export current browser context storage_state to self.vault when available."""
        if self.vault is None or self._context is None:
            return
        if not hasattr(self.vault, "save_browser_state"):
            return
        try:
            state = await self._context.storage_state()
            if state is not None:
                res = self.vault.save_browser_state(state)
                if inspect.isawaitable(res):
                    await res
        except Exception as exc:
            logger.debug("browser_vault_save_failed", error=str(exc))

    async def _guard_request(self, route: Any) -> None:
        """Apply scheme/egress guard to navigations and SSRF guard to subresources."""
        request_url = route.request.url
        parsed = urlparse(request_url)
        if parsed.scheme.lower() in {"data", "blob", "about"}:
            await route.continue_()
            return
        try:
            is_nav = True
            try:
                is_nav = bool(route.request.is_navigation_request())
            except Exception:
                is_nav = True
            if is_nav:
                self._validate_url(request_url)
            else:
                # For CDN subresources (scripts, stylesheets, fonts, images),
                # still enforce SSRF/metadata egress protection without blocking third-party CDNs.
                allowed, reason = is_target_allowed(request_url)
                if not allowed:
                    raise ValueError(reason)
        except ValueError:
            await route.abort("blockedbyclient")
            return
        await route.continue_()

    async def navigate(self, url: str) -> PageSnapshot:
        """Navigate to URL and capture full page state."""
        self._validate_url(url)
        if self._using_playwright and self._page:
            snap = await self._playwright_navigate(url)
        else:
            snap = await self._httpx_navigate(url)
        await self.save_state_to_vault()
        return snap

    async def _playwright_navigate(self, url: str) -> PageSnapshot:
        """Full browser navigation with Playwright."""
        console_logs: list[str] = []
        network_reqs: list[dict[str, Any]] = []

        self._page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))
        self._page.on("request", lambda req: network_reqs.append({
            "url": req.url, "method": req.method, "resource_type": req.resource_type,
        }))

        response = await self._page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")
        status = response.status if response else 200
        final_url = self._page.url or url
        self._validate_url(final_url)

        # Wait for dynamic content
        await self._page.wait_for_timeout(1000)

        title = await self._page.title()
        html = await self._page.content()
        screenshot = await self._page.screenshot(full_page=True, type="png")
        screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")

        cookies = await self._context.cookies()

        return PageSnapshot(
            url=final_url,
            title=title,
            status_code=status,
            html_content=html[:50000],
            screenshot_b64=screenshot_b64,
            cookies=[{"name": c["name"], "value": c["value"], "domain": c["domain"]} for c in cookies],
            console_logs=console_logs[:50],
            network_requests=network_reqs[:100],
        )

    async def _httpx_navigate(self, url: str) -> PageSnapshot:
        """Fallback HTTP-only navigation without browser rendering."""
        import httpx
        async with httpx.AsyncClient(
            verify=tls_verify(url),
            timeout=15,
            follow_redirects=False,
        ) as client:
            current_url = url
            response = None
            for _ in range(10):
                self._validate_url(current_url)
                response = await client.get(current_url)
                if response.status_code not in {301, 302, 303, 307, 308}:
                    break
                location = response.headers.get("location")
                if not location:
                    break
                current_url = str(response.url.join(location))
            if response is None:
                raise RuntimeError("browser navigation did not receive a response")
            self._validate_url(str(response.url))
            return PageSnapshot(
                url=str(response.url),
                title="",
                status_code=response.status_code,
                html_content=response.text[:50000],
                screenshot_b64="",
                cookies=[],
                console_logs=[],
                network_requests=[],
            )

    async def evaluate_javascript(self, expression: str) -> Any:
        """Evaluate JavaScript only when explicitly enabled by the caller."""
        if not self.allow_javascript:
            raise PermissionError("arbitrary JavaScript execution is disabled by default")
        if not self._using_playwright or not self._page:
            raise RuntimeError("JavaScript evaluation requires a Playwright page")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("JavaScript expression cannot be empty")
        return await self._page.evaluate(expression)

    async def click(self, selector: str) -> bool:
        """Click an element by CSS selector."""
        if not self._using_playwright or not self._page:
            return False
        try:
            await self._page.click(selector, timeout=5000)
            await self.save_state_to_vault()
            return True
        except Exception as e:
            logger.debug("click_failed", selector=selector, error=str(e))
            return False

    async def type_text(self, selector: str, text: str) -> bool:
        """Type text into an input field."""
        if not self._using_playwright or not self._page:
            return False
        try:
            await self._page.fill(selector, text)
            await self.save_state_to_vault()
            return True
        except Exception as e:
            logger.debug("type_failed", selector=selector, error=str(e))
            return False

    async def select_option(self, selector: str, value: str) -> bool:
        """Select an option from a <select> dropdown by value or visible label."""
        if not self._using_playwright or not self._page:
            return False
        try:
            try:
                await self._page.select_option(selector, value=value, timeout=5000)
            except Exception:
                await self._page.select_option(selector, label=value, timeout=5000)
            return True
        except Exception as e:
            logger.debug("select_option_failed", selector=selector, value=value, error=str(e))
            return False

    async def press_key(self, key: str) -> bool:
        """Press a keyboard key or shortcut on the active browser page."""
        if not self._using_playwright or not self._page:
            return False
        try:
            await self._page.keyboard.press(key)
            await self.save_state_to_vault()
            return True
        except Exception as e:
            logger.debug("press_key_failed", key=key, error=str(e))
            return False

    async def scroll(self, delta_x: int = 0, delta_y: int = 500) -> bool:
        """Scroll the browser viewport."""
        if not self._using_playwright or not self._page:
            return False
        try:
            await self._page.mouse.wheel(delta_x, delta_y)
            return True
        except Exception as e:
            logger.debug("scroll_failed", delta_x=delta_x, delta_y=delta_y, error=str(e))
            return False

    async def wait_for_element(self, selector: str, timeout_ms: int = 15000) -> bool:
        """Wait until an element matching the selector is visible.

        A human waits for a "Download" or "Next" button to render before
        clicking it; without this the agent clicks stale coordinates on a
        page that is still loading.
        """
        if not self._using_playwright or not self._page:
            last_html = getattr(self, "_last_html", "") or ""
            clean_sel = selector.strip("#.")
            return bool(clean_sel and clean_sel in last_html)
        try:
            await self._page.wait_for_selector(selector, state="visible", timeout=timeout_ms)
            return True
        except Exception as e:
            logger.debug("wait_for_element_failed", selector=selector, error=str(e))
            return False

    async def download(self, selector: str, save_path: str, timeout_ms: int = 60000) -> bool:
        """Click a download link/button and save the file to save_path.

        Models the human step: click "Download" on the website, wait for the
        file to arrive, then it is on disk ready to run. accept_downloads is
        enabled at context creation so the download stream is captured rather
        than triggering the browser's download UI. Returns False if Playwright is
        unavailable or the download does not complete.
        """
        if not self._using_playwright or not self._page:
            return False
        try:
            async with self._page.expect_download(timeout=timeout_ms) as dl_info:
                await self._page.click(selector, timeout=5000)
            dl = await dl_info.value
            dl.save_as(save_path)
            return True
        except Exception as e:
            logger.debug("download_failed", selector=selector, save_path=save_path, error=str(e))
            return False

    async def find_interactive_elements(self) -> list[DOMElement]:
        """Find all clickable/input elements on the page."""
        if not self._using_playwright or not self._page:
            return []

        elements: list[DOMElement] = []
        for tag in ["a", "button", "input", "select", "textarea", "form"]:
            try:
                locators = self._page.locator(tag)
                count = await locators.count()
                for i in range(min(count, 50)):
                    el = locators.nth(i)
                    text = (await el.inner_text())[:100] if tag != "input" else ""
                    visible = await el.is_visible()
                    attrs = {}
                    for attr in ["href", "action", "name", "id", "type", "value"]:
                        val = await el.get_attribute(attr)
                        if val:
                            attrs[attr] = val
                    bbox = await el.bounding_box() if visible else None

                    if attrs.get("id"):
                        selector = f"#{attrs['id']}"
                    elif attrs.get("name"):
                        selector = f'{tag}[name="{attrs["name"]}"]'
                    else:
                        selector = f"{tag} >> nth={i}"

                    elements.append(DOMElement(
                        tag=tag,
                        text=text.strip(),
                        selector=selector,
                        attributes=attrs,
                        is_visible=visible,
                        bbox=(
                            int(bbox["x"]),
                            int(bbox["y"]),
                            int(bbox["width"]),
                            int(bbox["height"]),
                        )
                        if bbox
                        else None,
                    ))
            except Exception:
                pass

        return elements

    async def extract_api_endpoints(self) -> list[str]:
        """Extract API endpoint URLs from page JavaScript and network traffic."""
        endpoints: set[str] = []
        if not self._using_playwright or not self._page:
            return list(endpoints)

        # From page content
        html = await self._page.content()
        import re
        api_patterns = re.findall(r'["\'](/(?:api|v[0-9]+|graphql)[^"\']*)["\']', html)
        for match in api_patterns:
            endpoints.add(match if match.startswith("/") else f"/{match}")

        # From fetch/XHR URLs
        urls = await self._page.evaluate("""
            () => {
                const entries = performance.getEntriesByType('resource');
                return entries
                    .filter(e => e.initiatorType === 'fetch' || e.initiatorType === 'xmlhttprequest')
                    .map(e => e.name)
                    .slice(0, 50);
            }
        """)
        if isinstance(urls, list):
            endpoints.update(urls)

        return list(endpoints)

    async def inject_xss_payload(self, selector: str, payload: str) -> dict[str, Any]:
        """Submit input into a form field and observe whether a browser dialog is triggered."""
        if not self._using_playwright or not self._page:
            return {"injected": False, "triggered": False}

        alert_triggered = False

        async def handle_dialog(dialog):
            nonlocal alert_triggered
            alert_triggered = True
            await dialog.dismiss()

        self._page.on("dialog", handle_dialog)

        await self.type_text(selector, payload)

        # Try submitting nearby form
        try:
            form = self._page.locator(f"{selector}").locator("xpath=ancestor::form")
            if await form.count() > 0:
                submit = form.locator('button[type="submit"], input[type="submit"]')
                if await submit.count() > 0:
                    await submit.first.click()
        except Exception:
            pass

        await self._page.wait_for_timeout(2000)

        # Capture screenshot as proof
        screenshot = await self._page.screenshot(full_page=True, type="png")
        screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")

        return {
            "injected": True,
            "triggered": alert_triggered,
            "payload": payload,
            "screenshot_proof": screenshot_b64,
        }

    async def screenshot(self) -> PageSnapshot:
        """Capture screenshot and current page state without navigating or reloading."""
        if self._using_playwright and self._page:
            try:
                title = await self._page.title()
                html = await self._page.content()
                screenshot = await self._page.screenshot(full_page=True, type="png")
                screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")
                url = self._page.url or ""
                cookies = await self._context.cookies() if self._context else []
                return PageSnapshot(
                    url=url,
                    title=title,
                    status_code=int(getattr(self, "_last_status_code", 200) or 200),
                    html_content=html[:50000],
                    screenshot_b64=screenshot_b64,
                    cookies=[{"name": c["name"], "value": c["value"], "domain": c["domain"]} for c in cookies],
                )
            except Exception as e:
                logger.debug("browser_screenshot_failed", error=str(e))
        return PageSnapshot(
            url="",
            title="",
            status_code=0,
            html_content="",
            screenshot_b64="",
        )

    async def close(self) -> None:
        """Close browser and cleanup. Safe to call even if launch failed."""
        await self.save_state_to_vault()
        # Close the page/context before stopping playwright so the browser
        # process can shut down cleanly. When attached over CDP to the shared
        # :99 desktop browser, do NOT call self._browser.close() (which kills
        # the user's shared :99 desktop browser window); only disconnect and
        # clean up Playwright handles.
        if not self._attached_cdp:
            if self._page is not None:
                try:
                    await self._page.close()
                except Exception as e:
                    logger.debug("browser_page_close_error", error=str(e))
            if self._context is not None:
                try:
                    await self._context.close()
                except Exception as e:
                    logger.debug("browser_context_close_error", error=str(e))
            if self._browser is not None:
                try:
                    await self._browser.close()
                except Exception as e:
                    logger.debug("browser_close_error", error=str(e))
        self._page = None
        self._context = None
        self._browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.debug("playwright_stop_error", error=str(e))
            self._playwright = None
        self._attached_cdp = False
        self._using_playwright = False
        logger.info("browser_agent_closed")

    async def __aenter__(self) -> BrowserAgent:
        await self.launch()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def current_page_state(self) -> tuple[str, str]:
        """Return the live (url, title) of the current page, or a blank default.

        Used by the unified computer-use loop so the observation reflects the
        page state AFTER interactions (click/type), not just the last navigate.
        """
        if self._using_playwright and self._page:
            try:
                url = self._page.url or "about:blank"
                title = await self._page.title() if self._page else ""
                return url, title
            except Exception:
                return "about:blank", ""
        return "about:blank", ""

    async def current_page_observation(self) -> dict[str, Any]:
        """Return bounded, live browser evidence for the reasoning loop.

        URL/title alone is not enough for target-first analysis. The agent
        reasons from the loaded page while excluding cookies and local storage.
        """
        url, title = await self.current_page_state()
        observation: dict[str, Any] = {
            "url": url,
            "title": title,
            "text": "",
            "api_endpoints": [],
            "interactive_elements": [],
            "window_offset": (0, 0),
        }
        if not self._using_playwright or not self._page:
            return observation
        try:
            win_metrics = await self._page.evaluate(
                "() => [window.screenX || 0, window.screenY || 0, "
                "(window.outerWidth || 0) - (window.innerWidth || 0), "
                "(window.outerHeight || 0) - (window.innerHeight || 0)]"
            )
            if isinstance(win_metrics, (list, tuple)) and len(win_metrics) == 4:
                sx, sy, dw, dh = (int(v or 0) for v in win_metrics)
                offset_x = sx + max(0, dw // 2)
                offset_y = sy + max(0, dh)
                observation["window_offset"] = (offset_x, offset_y)
        except Exception as exc:
            logger.debug("browser_window_offset_failed", error=str(exc))
        try:
            text = await self._page.locator("body").inner_text(timeout=3000)
            observation["text"] = (text or "")[:12000]
        except Exception as exc:
            logger.debug("browser_body_text_failed", error=str(exc))
        try:
            observation["api_endpoints"] = (await self.extract_api_endpoints())[:100]
        except Exception as exc:
            logger.debug("browser_api_endpoint_extraction_failed", error=str(exc))
        try:
            observation["interactive_elements"] = [
                {
                    "tag": e.tag,
                    "text": e.text,
                    "selector": e.selector,
                    "visible": e.is_visible,
                    "bbox": e.bbox,
                    "attributes": e.attributes,
                }
                for e in (await self.find_interactive_elements())[:100]
            ]
        except Exception as exc:
            logger.debug("browser_interactive_observation_failed", error=str(exc))
        return observation
