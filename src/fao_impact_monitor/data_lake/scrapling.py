import asyncio
import base64
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from curl_cffi.curl import CurlError
from scrapling.fetchers import AsyncFetcher

logger = logging.getLogger(__name__)

PDF_MAGIC_BYTES = b"%PDF"
HTML_MAGIC_BYTES = b"<!doctype"

# HTTP bodies smaller than this (or empty) trigger a browser re-fetch. Catches
# Chrome PDF-viewer shells and other near-empty redirect placeholders.
_MIN_USEFUL_HTTP_BODY_BYTES = 512

BodyCaptureMode = Literal["raw", "rendered"]

# browserforge header data often lags Playwright's bundled Chrome version.
_MAX_BROWSERFORGE_CHROME_VERSION = 143

_SCRIPT_OR_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_LINK_HREF_RE = re.compile(r"""<a\b[^>]*\bhref\s*=""", re.IGNORECASE)
_SCRIPT_TAG_RE = re.compile(r"<script\b", re.IGNORECASE)
_EMPTY_MOUNT_RE = re.compile(
    r"""<div[^>]*\bid=["'](root|app|__next|___gatsby|_r_)["'][^>]*>\s*</div>""",
    re.IGNORECASE,
)
_EMPTY_CUSTOM_ELEMENT_RE = re.compile(
    r"<([a-z][a-z0-9]*-[a-z0-9-]*)\b[^>]*>\s*</\1>",
    re.IGNORECASE,
)
_SPA_SUBSTRING_MARKERS = (
    "data-reactroot",
    "ng-version=",
    "<app-root",
    "__next_f",
    "data-v-app",
    "__sveltekit",
    "q:container",
    "astro-island",
)
_DOWNLOAD_URL_RE = re.compile(
    r"(?:/download(?:\?|$))|(?:/bitstreams/)|(?:\.pdf(?:\?|$))",
    re.IGNORECASE,
)

_FETCH_FINAL_URL_JS = """
async (url) => {
    const res = await fetch(url, { credentials: 'include' });
    const bytes = new Uint8Array(await res.arrayBuffer());
    let binary = '';
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    }
    return btoa(binary);
}
"""


def _as_bytes(body: bytes | str | bytearray) -> bytes:
    if isinstance(body, bytes):
        return body
    if isinstance(body, str):
        return body.encode()
    return bytes(body)


def _looks_like_html(body: bytes) -> bool:
    head = body.lstrip()[:512].lower()
    return head.startswith((HTML_MAGIC_BYTES, b"<html")) or b"<html" in head


def looks_like_spa_shell(body: bytes) -> bool:
    """Return True when ``body`` looks like a client-rendered SPA shell.

    Detects near-empty HTML with framework mount points / custom elements and
    script bundles, where useful crawl links only appear after JS execution.
    """
    if not body or body.startswith(PDF_MAGIC_BYTES) or not _looks_like_html(body):
        return False

    html = body.decode("utf-8", errors="replace")
    lower = html.lower()
    without_assets = _SCRIPT_OR_STYLE_RE.sub(" ", lower)
    visible = _WHITESPACE_RE.sub(" ", _TAG_RE.sub(" ", without_assets)).strip()
    visible_len = len(visible)
    link_count = len(_LINK_HREF_RE.findall(lower))
    script_count = len(_SCRIPT_TAG_RE.findall(lower))
    has_marker = bool(_EMPTY_MOUNT_RE.search(lower)) or bool(
        _EMPTY_CUSTOM_ELEMENT_RE.search(lower)
    )
    if not has_marker:
        has_marker = any(marker in lower for marker in _SPA_SUBSTRING_MARKERS)

    # Substantial server-rendered content: keep the HTTP body.
    if visible_len >= 500 and link_count >= 3:
        return False
    if has_marker and visible_len < 300:
        return True
    # Script-heavy page with no crawlable links and almost no text.
    return script_count >= 3 and link_count == 0 and visible_len < 300


def _url_looks_like_binary_download(url: str) -> bool:
    """True for bitstream/PDF download URLs that should yield raw file bytes."""
    return bool(_DOWNLOAD_URL_RE.search(url))


def looks_like_chrome_pdf_viewer_shell(body: bytes) -> bool:
    """True for Chrome's built-in PDF viewer HTML (empty body + embedder CSS)."""
    if not body or body.startswith(PDF_MAGIC_BYTES) or not _looks_like_html(body):
        return False
    head = body.lstrip()[:2048].lower()
    return b"pdf_embedder.css" in head or b"chrome-extension://" in head


def looks_like_useless_http_body(body: bytes) -> bool:
    """Return True when the HTTP body is empty, tiny, or a PDF-viewer shell.

    Near-empty bodies and viewer shells need a browser re-fetch. FAO bitstream
    URLs often return an SPA shell over HTTP (handled separately); the Chrome
    PDF viewer only appears when Chromium navigates to ``application/pdf``.
    """
    if body.startswith(PDF_MAGIC_BYTES):
        return False
    if not body or len(body) < _MIN_USEFUL_HTTP_BODY_BYTES:
        return True
    return looks_like_chrome_pdf_viewer_shell(body)


def _is_acceptable_raw_capture(body: bytes) -> bool:
    """Whether a browser raw capture looks like real content (not a viewer shell)."""
    if not body:
        return False
    if body.startswith(PDF_MAGIC_BYTES):
        return True
    if looks_like_chrome_pdf_viewer_shell(body):
        return False
    # FAO bitstream API URLs under /bitstreams/ often return JSON metadata.
    head = body.lstrip()[:1]
    if head in (b"{", b"["):
        return False
    return not (_looks_like_html(body) and len(body) < _MIN_USEFUL_HTTP_BODY_BYTES)


def chromium_installed() -> bool:
    """Return whether patchright's Chromium browser binary is available."""
    try:
        from patchright.sync_api import sync_playwright
    except ImportError:
        return False

    with sync_playwright() as playwright:
        return Path(playwright.chromium.executable_path).exists()


def ensure_chromium(*, force: bool = False) -> None:
    """Install Chromium for Scrapling's StealthyFetcher if missing.

    uv cannot run post-install hooks, so call this after ``uv sync`` (or rely on
    ``browser_fetch``, which calls it automatically).
    """
    if not force and chromium_installed():
        logger.debug("Chromium already installed")
        return
    logger.info("Installing Chromium for Scrapling browser fetcher")
    subprocess.run(
        [sys.executable, "-m", "patchright", "install", "chromium"],
        check=True,
    )


def ensure_chromium_main() -> None:
    """CLI entry point: ``uv run install-browsers``."""
    force = "--force" in sys.argv[1:]
    ensure_chromium(force=force)


def _stealthy_fetcher() -> Any:
    """Import StealthyFetcher, working around browserforge/Chrome version mismatches."""
    try:
        from scrapling.fetchers import StealthyFetcher

        return StealthyFetcher
    except ValueError:
        from scrapling.engines.toolbelt import fingerprints

        fingerprints.chromium_version = min(
            fingerprints.chromium_version, _MAX_BROWSERFORGE_CHROME_VERSION
        )
        fingerprints.chrome_version = min(
            fingerprints.chrome_version, _MAX_BROWSERFORGE_CHROME_VERSION
        )
        for name in list(sys.modules):
            if name.startswith(
                ("scrapling.engines._browsers", "scrapling.fetchers.stealth")
            ):
                del sys.modules[name]
        from scrapling.fetchers import StealthyFetcher

        return StealthyFetcher


async def fetch(
    *,
    url: str,
    stealthy_headers: bool = True,
    follow_redirects: bool | Literal["safe"] = True,
    timeout: int = 30,
    retries: int = 3,
    retry_delay: int = 2,
) -> bytes:
    """Fetch ``url`` with Scrapling ``AsyncFetcher`` and return the response body."""
    logger.info("HTTP fetch %s (timeout=%ss, retries=%s)", url, timeout, retries)
    try:
        response = await AsyncFetcher.get(
            url,
            stealthy_headers=stealthy_headers,
            follow_redirects=follow_redirects,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
        )
    except CurlError:
        logger.warning("HTTP fetch failed for %s", url, exc_info=True)
        raise
    body = _as_bytes(response.body)
    logger.debug("HTTP fetch succeeded for %s (%s bytes)", url, len(body))
    return body


def _browser_errors() -> tuple[type[BaseException], ...]:
    try:
        from patchright._impl._errors import Error as PatchrightError
    except ImportError:
        return (RuntimeError,)
    return (PatchrightError, RuntimeError)


_BROWSER_ERRORS = _browser_errors()


async def _wait_for_settled_url(page: Any, *, settle_ms: int = 500) -> str:
    """Wait until ``page.url`` stops changing between short pauses."""
    url = str(page.url)
    wait_for_timeout = getattr(page, "wait_for_timeout", None)
    if wait_for_timeout is None:
        return url
    for _ in range(10):
        await wait_for_timeout(settle_ms)
        current = str(page.url)
        if current == url:
            return url
        url = current
    return str(page.url)


def _response_may_be_pdf(response: Any) -> bool:
    """True when a response is worth reading for PDF bytes (cheap pre-filter)."""
    headers = getattr(response, "headers", None) or {}
    content_type = str(headers.get("content-type", "")).lower()
    if "application/pdf" in content_type or "octet-stream" in content_type:
        return True
    resp_url = str(getattr(response, "url", "")).lower()
    return (
        resp_url.endswith(".pdf")
        or ".pdf?" in resp_url
        # FAO serves the file at .../bitstreams/<uuid>/content
        or ("/bitstreams/" in resp_url and resp_url.rstrip("/").endswith("/content"))
    )


def _prefer_network_capture(current: bytes | None, candidate: bytes) -> bytes:
    """Keep the larger PDF body when several PDF responses are seen."""
    if current is None:
        return candidate
    return candidate if len(candidate) >= len(current) else current


async def _request_bytes(page: Any, url: str) -> bytes | None:
    """Fetch ``url`` via Playwright's request API (bypasses Chrome PDF viewer)."""
    context = getattr(page, "context", None)
    request = getattr(context, "request", None) if context is not None else None
    if request is None:
        request = getattr(page, "request", None)
    if request is None:
        return None
    response = await request.get(url)
    body = await response.body()
    return _as_bytes(body)


async def _capture_final_body(page: Any) -> bytes:
    """Fetch raw bytes for the final URL after SPA redirects settle.

    Prefer Playwright's ``request`` API so ``application/pdf`` responses are not
    replaced by Chrome's built-in PDF viewer HTML shell. Fall back to an in-page
    ``fetch`` when the request API is unavailable.

    Prefer capturing the network response during navigation (see
    ``browser_fetch``) so this path is only a fallback and does not
    re-download large PDFs on the happy path.
    """
    last_error: BaseException | None = None
    for _ in range(3):
        try:
            url = await _wait_for_settled_url(page, settle_ms=250)
            body = await _request_bytes(page, url)
            if body is not None and _is_acceptable_raw_capture(body):
                return body
            payload = await page.evaluate(_FETCH_FINAL_URL_JS, url)
            body = base64.b64decode(payload)
            if _is_acceptable_raw_capture(body):
                return body
            wait_for_timeout = getattr(page, "wait_for_timeout", None)
            if wait_for_timeout is not None:
                await wait_for_timeout(250)
        except _BROWSER_ERRORS as exc:
            last_error = exc
            wait_for_timeout = getattr(page, "wait_for_timeout", None)
            if wait_for_timeout is not None:
                await wait_for_timeout(250)
    if last_error is not None:
        raise last_error
    url = str(page.url)
    body = await _request_bytes(page, url)
    if body is not None and _is_acceptable_raw_capture(body):
        return body
    payload = await page.evaluate(_FETCH_FINAL_URL_JS, url)
    body = base64.b64decode(payload)
    if _is_acceptable_raw_capture(body):
        return body
    if body is not None:
        return body
    raise RuntimeError(f"Failed to capture acceptable raw body for {url}")


async def _capture_rendered_html(page: Any) -> bytes:
    """Return the browser-rendered DOM HTML after client-side content settles."""
    await _wait_for_settled_url(page)
    wait_for_selector = getattr(page, "wait_for_selector", None)
    wait_for_timeout = getattr(page, "wait_for_timeout", None)
    if wait_for_selector is not None:
        try:
            # Prefer waiting for crawlable links; fall back to a short settle.
            await wait_for_selector("a[href]", timeout=15_000)
        except _BROWSER_ERRORS:
            if wait_for_timeout is not None:
                await wait_for_timeout(2_000)
    elif wait_for_timeout is not None:
        await wait_for_timeout(3_000)
    html = await page.content()
    if isinstance(html, bytes):
        return html
    return str(html).encode("utf-8")


async def browser_fetch(
    *,
    url: str,
    headless: bool = True,
    disable_resources: bool = False,
    network_idle: bool = True,
    solve_cloudflare: bool = True,
    timeout: int = 30_000,
    retries: int = 3,
    body_mode: BodyCaptureMode = "raw",
) -> bytes:
    """Fetch ``url`` with Scrapling ``StealthyFetcher`` and return the response body.

    ``body_mode="raw"`` captures network response bodies during navigation (so
    ``application/pdf`` bytes are kept before Chrome replaces them with its PDF
    viewer shell). Falls back to Playwright's request API / in-page ``fetch`` of
    the final URL when no suitable network body was seen.

    ``body_mode="rendered"`` returns ``page.content()`` after the DOM settles —
    needed for SPA shells where useful markup only appears after JS runs.
    """
    logger.info(
        "Browser fetch %s (timeout=%sms, retries=%s, solve_cloudflare=%s, body_mode=%s)",
        url,
        timeout,
        retries,
        solve_cloudflare,
        body_mode,
    )
    # sync_playwright cannot run inside an active asyncio loop
    await asyncio.to_thread(ensure_chromium)
    stealthy_fetcher = _stealthy_fetcher()
    captured: dict[str, bytes] = {}
    network_tasks: list[asyncio.Task[None]] = []

    async def page_setup(page: Any) -> None:
        """Register listeners before navigation so PDF bytes are not re-downloaded."""
        if body_mode != "raw":
            return
        on = getattr(page, "on", None)
        if on is None:
            return

        def on_response(response: Any) -> None:
            if not _response_may_be_pdf(response):
                return

            async def _read() -> None:
                try:
                    body = _as_bytes(await response.body())
                except Exception:
                    logger.debug(
                        "Could not read browser response body for %s",
                        getattr(response, "url", "?"),
                        exc_info=True,
                    )
                    return
                # Only keep real PDF bytes — FAO also serves JSON under
                # /bitstreams/ URLs, which must not win the capture.
                if not body.startswith(PDF_MAGIC_BYTES):
                    return
                captured["network_body"] = _prefer_network_capture(
                    captured.get("network_body"),
                    body,
                )

            network_tasks.append(asyncio.create_task(_read()))

        on("response", on_response)

    async def page_action(page: Any) -> Any:
        if network_tasks:
            await asyncio.gather(*network_tasks, return_exceptions=True)
            network_tasks.clear()
        if body_mode == "rendered":
            captured["body"] = await _capture_rendered_html(page)
        else:
            network_body = captured.get("network_body")
            if network_body is not None and network_body.startswith(PDF_MAGIC_BYTES):
                logger.debug(
                    "Using network-captured PDF for %s (%s bytes)",
                    url,
                    len(network_body),
                )
                captured["body"] = network_body
            else:
                captured["body"] = await _capture_final_body(page)
        return page

    async def _fetch(*, with_cloudflare: bool) -> bytes:
        captured.pop("body", None)
        captured.pop("network_body", None)
        network_tasks.clear()
        try:
            await stealthy_fetcher.async_fetch(
                url,
                headless=headless,
                disable_resources=disable_resources,
                network_idle=network_idle,
                solve_cloudflare=with_cloudflare,
                timeout=timeout,
                retries=retries,
                page_setup=page_setup,
                page_action=page_action,
            )
        except _BROWSER_ERRORS:
            # Scrapling may still fail on page.content() during client-side
            # navigations even after we already captured the body.
            if "body" not in captured:
                raise
        return captured["body"]

    try:
        body = await _fetch(with_cloudflare=solve_cloudflare)
    except _BROWSER_ERRORS:
        # Cloudflare detection itself calls page.content() and races with SPAs
        # (e.g. YouTube). Retry once without the solver if we asked for it.
        if not solve_cloudflare:
            logger.warning("Browser fetch failed for %s", url, exc_info=True)
            raise
        logger.warning(
            "Browser fetch with Cloudflare solver failed for %s; retrying without it",
            url,
            exc_info=True,
        )
        body = await _fetch(with_cloudflare=False)

    logger.debug("Browser fetch succeeded for %s (%s bytes)", url, len(body))
    return body


async def reliable_fetch(
    *,
    url: str,
    timeout: int = 30,
    stealthy_headers: bool = True,
    follow_redirects: bool | Literal["safe"] = True,
    fetch_retries: int = 3,
    retry_delay: int = 2,
    headless: bool = True,
    disable_resources: bool = False,
    network_idle: bool = True,
    solve_cloudflare: bool = True,
    browser_retries: int = 3,
) -> bytes:
    """Fetch ``url`` with ``fetch``, falling back to ``browser_fetch`` when needed.

    Falls back to the browser when:

    - HTTP fetch raises ``CurlError`` (transport / client failures), or
    - the HTTP body looks like an empty SPA shell (no crawlable content without JS),
      or
    - the HTTP body is empty / tiny / a Chrome PDF-viewer shell (need raw PDF bytes).

    SPA shells are re-fetched with ``body_mode="rendered"`` so the hydrated DOM is
    returned — except for bitstream/PDF download URLs, which use ``body_mode="raw"``
    so Chrome's PDF viewer HTML is not returned. Transport failures and useless /
    PDF-viewer bodies keep ``body_mode="raw"``.

    ``timeout`` is in seconds (same as ``fetch``) and is converted to milliseconds
    for ``browser_fetch``.
    """
    logger.info("Reliable fetch %s (HTTP then browser fallback)", url)
    try:
        body = await fetch(
            url=url,
            stealthy_headers=stealthy_headers,
            follow_redirects=follow_redirects,
            timeout=timeout,
            retries=fetch_retries,
            retry_delay=retry_delay,
        )
    except CurlError as exc:
        logger.warning(
            "HTTP fetch failed for %s (%s); falling back to browser",
            url,
            exc,
        )
        return await browser_fetch(
            url=url,
            headless=headless,
            disable_resources=disable_resources,
            network_idle=network_idle,
            solve_cloudflare=solve_cloudflare,
            timeout=timeout * 1000,
            retries=browser_retries,
            body_mode="raw",
        )

    spa_shell = looks_like_spa_shell(body)
    useless = looks_like_useless_http_body(body)
    if spa_shell and _url_looks_like_binary_download(url):
        logger.info(
            "SPA shell on download URL %s (%s bytes); re-fetching raw bytes",
            url,
            len(body),
        )
        return await browser_fetch(
            url=url,
            headless=headless,
            disable_resources=disable_resources,
            network_idle=network_idle,
            solve_cloudflare=solve_cloudflare,
            timeout=timeout * 1000,
            retries=browser_retries,
            body_mode="raw",
        )
    if spa_shell:
        logger.info(
            "SPA shell detected for %s (%s bytes); re-fetching rendered HTML",
            url,
            len(body),
        )
        return await browser_fetch(
            url=url,
            headless=headless,
            disable_resources=disable_resources,
            network_idle=network_idle,
            solve_cloudflare=solve_cloudflare,
            timeout=timeout * 1000,
            retries=browser_retries,
            body_mode="rendered",
        )
    if useless:
        logger.info(
            "Useless/empty HTTP body for %s (%s bytes); "
            "re-fetching raw bytes via browser",
            url,
            len(body),
        )
        return await browser_fetch(
            url=url,
            headless=headless,
            disable_resources=disable_resources,
            network_idle=network_idle,
            solve_cloudflare=solve_cloudflare,
            timeout=timeout * 1000,
            retries=browser_retries,
            body_mode="raw",
        )
    return body
