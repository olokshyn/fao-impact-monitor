import asyncio
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest
from curl_cffi.curl import CurlError

from fao_impact_monitor.data_lake.scrapling import (
    HTML_MAGIC_BYTES,
    PDF_MAGIC_BYTES,
    browser_fetch,
    ensure_chromium,
    fetch,
    looks_like_spa_shell,
    looks_like_useless_http_body,
    reliable_fetch,
)

FAO_SPA_SHELL = b"""<!DOCTYPE html>
<html>
<head><title>FAO Knowledge Repository</title></head>
<body>
  <ds-app></ds-app>
  <script src="runtime.js" type="module"></script>
  <script src="polyfills.js" type="module"></script>
  <script src="main.js" type="module"></script>
</body>
</html>
"""

FAO_PDF_URL_1 = (
    "https://openknowledge.fao.org/bitstreams/"
    "b9c93694-9ef0-40e5-9321-9a0095b02316/download"
)
FAO_PDF_URL_2 = (
    "https://openknowledge.fao.org/bitstreams/"
    "c1caede2-ea98-46b0-b663-4cae429e05d3/download"
)
FAO_NEWS_URL = (
    "https://www.fao.org/newsroom/detail/"
    "el-nino-is-coming-here-is-where-the-risks-to-agriculture-are-highest/en"
)
YOUTUBE_URL = "https://www.youtube.com/watch?v=yezjIzb3OT8"

CHROME_PDF_VIEWER_SHELL = (
    b"<!DOCTYPE html><html><head>\n"
    b'    <link rel="stylesheet" href="chrome-extension://'
    b'mhjfbmdgcfjbbpaeojofohoefgiehjai/pdf_embedder.css">\n'
    b"  </head>\n  <body>\n    \n  \n\n</body></html>"
)


def test_fetch_returns_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    get = AsyncMock(return_value=SimpleNamespace(body=b"%PDF-1.4 content"))
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.AsyncFetcher.get",
        get,
    )

    body = asyncio.run(fetch(url="https://example.com/doc.pdf", timeout=10, retries=1))

    assert body == b"%PDF-1.4 content"
    get.assert_awaited_once_with(
        "https://example.com/doc.pdf",
        stealthy_headers=True,
        follow_redirects=True,
        timeout=10,
        retries=1,
        retry_delay=2,
    )


def test_fetch_encodes_string_body(monkeypatch: pytest.MonkeyPatch) -> None:
    get = AsyncMock(return_value=SimpleNamespace(body="<html/>"))
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.AsyncFetcher.get",
        get,
    )

    body = asyncio.run(fetch(url="https://example.com/"))

    assert body == b"<html/>"


def test_browser_fetch_returns_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_page_action(_url: str, **kwargs: object) -> SimpleNamespace:
        page_setup = kwargs.get("page_setup")
        page_action = kwargs["page_action"]
        assert callable(page_action)
        page = SimpleNamespace(url="https://example.com/protected")

        async def evaluate(_script: str, url: str) -> str:
            assert url == "https://example.com/protected"
            # base64 for b"browser-body"
            return "YnJvd3Nlci1ib2R5"

        page.evaluate = evaluate
        page.on = lambda _event, _handler: None
        if callable(page_setup):
            result = page_setup(page)
            if asyncio.iscoroutine(result):
                await result
        await page_action(page)
        return SimpleNamespace(body=b"unused")

    async_fetch = AsyncMock(side_effect=run_page_action)
    stealthy_fetcher = SimpleNamespace(async_fetch=async_fetch)
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.ensure_chromium",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling._stealthy_fetcher",
        lambda: stealthy_fetcher,
    )

    body = asyncio.run(
        browser_fetch(
            url="https://example.com/protected",
            timeout=60_000,
            retries=2,
        )
    )

    assert body == b"browser-body"
    async_fetch.assert_awaited_once_with(
        "https://example.com/protected",
        headless=True,
        disable_resources=False,
        network_idle=True,
        solve_cloudflare=True,
        timeout=60_000,
        retries=2,
        page_setup=ANY,
        page_action=ANY,
    )


def test_browser_fetch_uses_network_captured_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_bytes = b"%PDF-1.4 network-captured"

    async def run_with_network(_url: str, **kwargs: object) -> SimpleNamespace:
        page_setup = kwargs["page_setup"]
        page_action = kwargs["page_action"]
        assert callable(page_setup)
        assert callable(page_action)

        handlers: list[object] = []

        def on(event: str, handler: object) -> None:
            assert event == "response"
            handlers.append(handler)

        page = SimpleNamespace(url="https://example.com/doc.pdf", on=on)
        result = page_setup(page)
        if asyncio.iscoroutine(result):
            await result

        async def response_body() -> bytes:
            return pdf_bytes

        fake_response = SimpleNamespace(
            url="https://cdn.example.com/file.pdf",
            headers={"content-type": "application/pdf"},
            body=response_body,
        )

        assert handlers
        handler = handlers[0]
        assert callable(handler)
        handler(fake_response)
        await page_action(page)
        return SimpleNamespace(body=b"unused")

    async_fetch = AsyncMock(side_effect=run_with_network)
    stealthy_fetcher = SimpleNamespace(async_fetch=async_fetch)
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.ensure_chromium",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling._stealthy_fetcher",
        lambda: stealthy_fetcher,
    )

    body = asyncio.run(
        browser_fetch(
            url="https://example.com/doc.pdf",
            solve_cloudflare=False,
            retries=1,
        )
    )

    assert body == pdf_bytes


def test_looks_like_spa_shell_detects_fao_app_shell() -> None:
    assert looks_like_spa_shell(FAO_SPA_SHELL)
    assert not looks_like_spa_shell(b"%PDF-1.4")
    assert not looks_like_spa_shell(b"not html")
    content_html = (
        b"<!DOCTYPE html><html><body>"
        b"<h1>Report</h1><p>El Nino impacts agriculture in Kenya.</p>"
        b'<a href="/doc1.pdf">PDF</a><a href="/doc2">More</a>'
        b'<a href="/doc3">Extra</a></body></html>'
    )
    assert not looks_like_spa_shell(content_html)


def test_looks_like_useless_http_body() -> None:
    assert looks_like_useless_http_body(b"")
    assert looks_like_useless_http_body(b"<html></html>")
    assert looks_like_useless_http_body(CHROME_PDF_VIEWER_SHELL)
    assert not looks_like_useless_http_body(b"%PDF-1.4" + b"x" * 600)
    assert not looks_like_useless_http_body(
        b"<!DOCTYPE html><html><body>" + (b"<p>content</p>" * 80) + b"</body></html>"
    )


def test_reliable_fetch_uses_browser_raw_for_pdf_viewer_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_mock = AsyncMock(return_value=CHROME_PDF_VIEWER_SHELL)
    browser_mock = AsyncMock(return_value=b"%PDF-1.4 real-bytes")
    monkeypatch.setattr("fao_impact_monitor.data_lake.scrapling.fetch", fetch_mock)
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.browser_fetch",
        browser_mock,
    )

    body = asyncio.run(
        reliable_fetch(
            url=FAO_PDF_URL_2,
            timeout=15,
            fetch_retries=1,
            solve_cloudflare=False,
            browser_retries=2,
        )
    )

    assert body.startswith(PDF_MAGIC_BYTES)
    browser_mock.assert_awaited_once_with(
        url=FAO_PDF_URL_2,
        headless=True,
        disable_resources=False,
        network_idle=True,
        solve_cloudflare=False,
        timeout=15_000,
        retries=2,
        body_mode="raw",
    )


def test_browser_fetch_rendered_returns_page_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_page_action(_url: str, **kwargs: object) -> SimpleNamespace:
        page_setup = kwargs.get("page_setup")
        page_action = kwargs["page_action"]
        assert callable(page_action)
        page = SimpleNamespace(url="https://example.com/spa")

        async def wait_for_timeout(_ms: int) -> None:
            return

        async def wait_for_selector(_sel: str, timeout: int = 0) -> None:
            del timeout

        async def content() -> str:
            return "<html><body><a href='/x'>link</a></body></html>"

        page.wait_for_timeout = wait_for_timeout
        page.wait_for_selector = wait_for_selector
        page.content = content
        page.on = lambda _event, _handler: None
        if callable(page_setup):
            result = page_setup(page)
            if asyncio.iscoroutine(result):
                await result
        await page_action(page)
        return SimpleNamespace(body=b"unused")

    async_fetch = AsyncMock(side_effect=run_page_action)
    stealthy_fetcher = SimpleNamespace(async_fetch=async_fetch)
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.ensure_chromium",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling._stealthy_fetcher",
        lambda: stealthy_fetcher,
    )

    body = asyncio.run(
        browser_fetch(
            url="https://example.com/spa",
            body_mode="rendered",
            solve_cloudflare=False,
            retries=1,
        )
    )
    assert b'href="/x"' in body or b"href='/x'" in body


def test_reliable_fetch_uses_fetch_when_successful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_mock = AsyncMock(return_value=b"http-body" + b"x" * 600)
    browser_mock = AsyncMock(return_value=b"browser-body")
    monkeypatch.setattr("fao_impact_monitor.data_lake.scrapling.fetch", fetch_mock)
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.browser_fetch",
        browser_mock,
    )

    body = asyncio.run(
        reliable_fetch(
            url="https://example.com/",
            timeout=15,
            fetch_retries=1,
            retry_delay=1,
            stealthy_headers=False,
            follow_redirects="safe",
            headless=False,
            disable_resources=True,
            network_idle=False,
            solve_cloudflare=False,
            browser_retries=2,
        )
    )

    assert body.startswith(b"http-body")
    fetch_mock.assert_awaited_once_with(
        url="https://example.com/",
        stealthy_headers=False,
        follow_redirects="safe",
        timeout=15,
        retries=1,
        retry_delay=1,
    )
    browser_mock.assert_not_awaited()


def test_reliable_fetch_uses_browser_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_mock = AsyncMock(side_effect=CurlError("network down"))
    browser_mock = AsyncMock(return_value=b"browser-body")
    monkeypatch.setattr("fao_impact_monitor.data_lake.scrapling.fetch", fetch_mock)
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.browser_fetch",
        browser_mock,
    )

    body = asyncio.run(
        reliable_fetch(
            url="https://example.com/protected",
            timeout=15,
            fetch_retries=1,
            retry_delay=1,
            stealthy_headers=False,
            follow_redirects=False,
            headless=False,
            disable_resources=True,
            network_idle=False,
            solve_cloudflare=False,
            browser_retries=2,
        )
    )

    assert body == b"browser-body"
    fetch_mock.assert_awaited_once()
    browser_mock.assert_awaited_once_with(
        url="https://example.com/protected",
        headless=False,
        disable_resources=True,
        network_idle=False,
        solve_cloudflare=False,
        timeout=15_000,
        retries=2,
        body_mode="raw",
    )


def test_reliable_fetch_uses_browser_for_spa_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_mock = AsyncMock(return_value=FAO_SPA_SHELL)
    browser_mock = AsyncMock(
        return_value=b"<html><body><a href='/pub.pdf'>PDF</a></body></html>"
    )
    monkeypatch.setattr("fao_impact_monitor.data_lake.scrapling.fetch", fetch_mock)
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.browser_fetch",
        browser_mock,
    )

    body = asyncio.run(
        reliable_fetch(
            url="https://openknowledge.fao.org/search",
            timeout=15,
            fetch_retries=1,
            solve_cloudflare=False,
            browser_retries=2,
        )
    )

    assert b"/pub.pdf" in body
    browser_mock.assert_awaited_once_with(
        url="https://openknowledge.fao.org/search",
        headless=True,
        disable_resources=False,
        network_idle=True,
        solve_cloudflare=False,
        timeout=15_000,
        retries=2,
        body_mode="rendered",
    )


def test_reliable_fetch_spa_shell_on_download_url_uses_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_mock = AsyncMock(return_value=FAO_SPA_SHELL)
    browser_mock = AsyncMock(return_value=b"%PDF-1.7 raw-bytes")
    monkeypatch.setattr("fao_impact_monitor.data_lake.scrapling.fetch", fetch_mock)
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.browser_fetch",
        browser_mock,
    )

    body = asyncio.run(
        reliable_fetch(
            url=FAO_PDF_URL_2,
            timeout=15,
            fetch_retries=1,
            solve_cloudflare=False,
            browser_retries=2,
        )
    )

    assert body.startswith(PDF_MAGIC_BYTES)
    browser_mock.assert_awaited_once_with(
        url=FAO_PDF_URL_2,
        headless=True,
        disable_resources=False,
        network_idle=True,
        solve_cloudflare=False,
        timeout=15_000,
        retries=2,
        body_mode="raw",
    )


def test_reliable_fetch_does_not_fallback_on_non_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_mock = AsyncMock(side_effect=ValueError("bad url"))
    browser_mock = AsyncMock(return_value=b"browser-body")
    monkeypatch.setattr("fao_impact_monitor.data_lake.scrapling.fetch", fetch_mock)
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.browser_fetch",
        browser_mock,
    )

    with pytest.raises(ValueError, match="bad url"):
        asyncio.run(reliable_fetch(url="https://example.com/"))

    browser_mock.assert_not_awaited()


def test_reliable_fetch_propagates_browser_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_mock = AsyncMock(side_effect=CurlError("network down"))
    browser_mock = AsyncMock(side_effect=RuntimeError("browser failed"))
    monkeypatch.setattr("fao_impact_monitor.data_lake.scrapling.fetch", fetch_mock)
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.browser_fetch",
        browser_mock,
    )

    with pytest.raises(RuntimeError, match="browser failed"):
        asyncio.run(reliable_fetch(url="https://example.com/"))


def test_ensure_chromium_skips_when_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.chromium_installed",
        lambda: True,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.subprocess.run",
        lambda cmd, check: calls.append(list(cmd)),
    )

    ensure_chromium()

    assert calls == []


def test_ensure_chromium_installs_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.chromium_installed",
        lambda: False,
    )
    monkeypatch.setattr(
        "fao_impact_monitor.data_lake.scrapling.subprocess.run",
        lambda cmd, check: calls.append(list(cmd)),
    )

    ensure_chromium()

    assert len(calls) == 1
    assert calls[0][-3:] == ["patchright", "install", "chromium"]


@pytest.mark.integration
def test_browser_fetch_fao_pdf_follows_redirect() -> None:
    body = asyncio.run(
        browser_fetch(
            url=FAO_PDF_URL_1,
            timeout=90_000,
            solve_cloudflare=False,
        )
    )

    assert body.startswith(PDF_MAGIC_BYTES)
    assert len(body) > 1_000


@pytest.mark.integration
def test_reliable_fetch_fao_bitstream_download_returns_pdf() -> None:
    """Bitstream /download URLs must yield PDF bytes, not a PDF-viewer HTML shell."""
    body = asyncio.run(
        reliable_fetch(
            url=FAO_PDF_URL_2,
            timeout=90,
            fetch_retries=2,
            browser_retries=2,
            solve_cloudflare=False,
        )
    )

    assert body.startswith(PDF_MAGIC_BYTES)
    assert len(body) > 1_000


@pytest.mark.integration
def test_fetch_fao_news_html_page() -> None:
    body = asyncio.run(fetch(url=FAO_NEWS_URL, timeout=60))

    assert body
    assert body.lstrip().lower().startswith(HTML_MAGIC_BYTES)
    text = body.decode("utf-8", errors="replace").lower()
    assert "el niño" in text or "el nino" in text
    assert "agriculture" in text


@pytest.mark.integration
def test_browser_fetch_youtube_protected_page() -> None:
    body = asyncio.run(
        browser_fetch(
            url=YOUTUBE_URL,
            timeout=90_000,
            # YouTube isn't Cloudflare-protected; the solver's page.content()
            # races with client-side navigations (themeRefresh, etc.).
            solve_cloudflare=False,
        )
    )

    assert body
    assert body.lstrip().lower().startswith(HTML_MAGIC_BYTES)
    text = body.decode("utf-8", errors="replace")
    assert "yezjIzb3OT8" in text
    assert "youtube" in text.lower()
    assert "el niño" in text.lower() or "el nino" in text.lower()


@pytest.mark.integration
@pytest.mark.skip(
    reason="plain fetch() currently succeeds for YouTube; revisit fallback coverage"
)
def test_reliable_fetch_youtube_falls_back_when_http_fails() -> None:
    with pytest.raises(CurlError):
        asyncio.run(fetch(url=YOUTUBE_URL, timeout=5, retries=1, retry_delay=1))

    body = asyncio.run(
        reliable_fetch(
            url=YOUTUBE_URL,
            timeout=30,
            fetch_retries=1,
            retry_delay=1,
            browser_retries=1,
            # YouTube isn't Cloudflare-protected; the solver's page.content()
            # races with client-side navigations (themeRefresh, etc.).
            solve_cloudflare=False,
        )
    )

    assert body
    assert body.lstrip().lower().startswith(HTML_MAGIC_BYTES)
    text = body.decode("utf-8", errors="replace")
    assert "yezjIzb3OT8" in text
    assert "youtube" in text.lower()
