"""Reusable local HTTP server for serving mock HTML and PDF fixtures in tests."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any


@dataclass
class MockHttpServer:
    """Thread-backed HTTP server with a mutable route table."""

    host: str = "127.0.0.1"
    port: int = 0
    _routes: dict[str, tuple[bytes, str]] = field(default_factory=dict)
    _httpd: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _thread: Thread | None = field(default=None, init=False, repr=False)

    def add_html(self, path: str, html: str) -> None:
        self._routes[_normalize_path(path)] = (
            html.encode("utf-8"),
            "text/html; charset=utf-8",
        )

    def add_pdf(self, path: str, body: bytes | None = None) -> None:
        payload = body if body is not None else b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        self._routes[_normalize_path(path)] = (payload, "application/pdf")

    def add_bytes(self, path: str, body: bytes, content_type: str) -> None:
        self._routes[_normalize_path(path)] = (body, content_type)

    @property
    def base_url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("MockHttpServer is not started")
        host = str(self._httpd.server_address[0])
        port = int(self._httpd.server_address[1])
        return f"http://{host}:{port}"

    def url(self, path: str) -> str:
        return f"{self.base_url}{_normalize_path(path)}"

    def start(self) -> None:
        if self._httpd is not None:
            return
        routes = self._routes
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = _normalize_path(self.path.split("?", 1)[0])
                item = routes.get(path)
                if item is None:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b"not found")
                    return
                body, content_type = item
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        # Keep a reference so Handler closure stays clear to type checkers.
        _ = server

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


def _normalize_path(path: str) -> str:
    if not path.startswith("/"):
        return f"/{path}"
    return path


def mock_http_server() -> Iterator[MockHttpServer]:
    """Context-style generator used by pytest fixtures."""
    server = MockHttpServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()
