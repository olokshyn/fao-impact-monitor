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
    reliable_fetch,
)

FAO_PDF_URL = (
    "https://openknowledge.fao.org/bitstreams/"
    "b9c93694-9ef0-40e5-9321-9a0095b02316/download"
)
FAO_NEWS_URL = (
    "https://www.fao.org/newsroom/detail/"
    "el-nino-is-coming-here-is-where-the-risks-to-agriculture-are-highest/en"
)
YOUTUBE_URL = "https://www.youtube.com/watch?v=yezjIzb3OT8"


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
        page_action = kwargs["page_action"]
        assert callable(page_action)
        page = SimpleNamespace(url="https://example.com/protected")

        async def evaluate(_script: str, url: str) -> str:
            assert url == "https://example.com/protected"
            # base64 for b"browser-body"
            return "YnJvd3Nlci1ib2R5"

        page.evaluate = evaluate
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
        page_action=ANY,
    )


def test_reliable_fetch_uses_fetch_when_successful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_mock = AsyncMock(return_value=b"http-body")
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

    assert body == b"http-body"
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
            url=FAO_PDF_URL,
            timeout=90_000,
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
