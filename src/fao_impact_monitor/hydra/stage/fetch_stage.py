"""FetchStage: download URL bodies via scrapling and persist via fsspec."""

from __future__ import annotations

import io
import logging
import zipfile
from enum import StrEnum, auto
from typing import Annotated, Any, Literal

from beanie import Document as BeanieDocument
from beanie import Indexed, PydanticObjectId
from fsspec.core import url_to_fs
from pydantic import BaseModel, Field

from fao_impact_monitor.hydra.config import FetchConfig
from fao_impact_monitor.hydra.document.document import Document
from fao_impact_monitor.hydra.scrapling import (
    HTML_MAGIC_BYTES,
    PDF_MAGIC_BYTES,
    AsyncFetchParams,
    BrowserFetchParams,
    ReliableFetchParams,
    ScraplingFetchResult,
    _looks_like_html,
    looks_like_useless_http_body,
    reliable_fetch_with_meta,
)
from fao_impact_monitor.hydra.stage.stage import Stage, StageResult
from fao_impact_monitor.hydra.status import Status
from fao_impact_monitor.hydra.task.task import Task

logger = logging.getLogger(__name__)

OLE_MAGIC_BYTES = b"\xd0\xcf\x11\xe0"
ZIP_MAGIC_BYTES = b"PK"


class ContentType(StrEnum):
    PDF = auto()
    HTML = auto()
    DOC = auto()
    DOCX = auto()
    PPT = auto()
    PPTX = auto()


class FetchRequest(BaseModel):
    fetcher: Literal["async", "stealthy"]
    fetcher_params: dict[str, Any] = Field(default_factory=dict)
    url: str
    headers: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class FetchResponse(BaseModel):
    status_code: int
    headers: dict[str, Any] = Field(default_factory=dict)
    body_header: bytes


class Fetch(BeanieDocument):
    """Persisted fetch attempt for a URL (collection ``fetches``)."""

    url: Annotated[str, Indexed(unique=True)]
    successful: bool
    request: FetchRequest | None = None
    response: FetchResponse | None = None
    content_type: ContentType | None = None
    body_path: str | None = None
    error: str | None = None

    class Settings:
        name = "fetches"


class FetchStageResult(StageResult):
    name: str = "fetch"
    status_code: int | None = None
    requested_url: str | None = None
    fetched_url: str | None = None
    response_headers: dict[str, Any] = Field(default_factory=dict)
    content_type: ContentType | None = None
    body_path: str | None = None


def extension_for_content_type(content_type: ContentType) -> str:
    return f".{content_type.value.lower()}"


def _header_content_type(headers: dict[str, Any]) -> str:
    for key, value in headers.items():
        if str(key).lower() == "content-type":
            return str(value).lower()
    return ""


def _url_suffix(url: str) -> str:
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    for ext in (".pdf", ".html", ".htm", ".docx", ".doc", ".pptx", ".ppt"):
        if path.endswith(ext):
            return ext
    return ""


def _zip_office_content_type(body: bytes) -> ContentType | None:
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile:
        return None
    if any(name.startswith("word/") for name in names):
        return ContentType.DOCX
    if any(name.startswith("ppt/") for name in names):
        return ContentType.PPTX
    return None


def infer_content_type(
    body: bytes,
    *,
    url: str,
    response_headers: dict[str, Any],
) -> ContentType | None:
    """Infer ContentType from magic bytes, headers, URL suffix, and zip internals."""
    content_type = _header_content_type(response_headers)
    suffix = _url_suffix(url)

    if body.startswith(PDF_MAGIC_BYTES) or "application/pdf" in content_type:
        return ContentType.PDF

    if body.startswith(OLE_MAGIC_BYTES):
        if (
            "presentation" in content_type
            or "ms-powerpoint" in content_type
            or suffix == ".ppt"
        ):
            return ContentType.PPT
        if "msword" in content_type or suffix == ".doc" or "word" in content_type:
            return ContentType.DOC
        # Prefer Content-Type when present; default DOC for generic OLE.
        if "powerpoint" in content_type:
            return ContentType.PPT
        return ContentType.DOC

    if body.startswith(ZIP_MAGIC_BYTES):
        if (
            "wordprocessingml" in content_type
            or "msword" in content_type
            or suffix == ".docx"
        ):
            return ContentType.DOCX
        if (
            "presentationml" in content_type
            or "ms-powerpoint" in content_type
            or suffix == ".pptx"
        ):
            return ContentType.PPTX
        office = _zip_office_content_type(body)
        if office is not None:
            return office

    head = body.lstrip()[:512].lower()
    if (
        head.startswith((HTML_MAGIC_BYTES, b"<html"))
        or "text/html" in content_type
        or suffix in {".html", ".htm"}
        or _looks_like_html(body)
    ):
        return ContentType.HTML

    return None


def _write_body_fsspec(body_path: str, body: bytes) -> None:
    """Write ``body`` to an fsspec path (local or remote URL).

    Do not use ``pathlib.Path`` here: it collapses ``s3://`` / ``file://`` to
    ``s3:/`` / ``file:/`` and breaks remote roots.
    """
    fs, path = url_to_fs(body_path)
    parent = fs._parent(path)
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(path, "wb") as handle:
        handle.write(body)


def _is_successful_fetch(meta: ScraplingFetchResult) -> bool:
    body = meta.body
    if not body:
        return False
    if looks_like_useless_http_body(body):
        return False
    if meta.fetcher == "stealthy":
        return True
    return 200 <= meta.status_code <= 299


async def _resolve_document(task: Task) -> Document | None:
    if task.document_id is not None:
        return await Document.get(task.document_id)
    if not task.url:
        return None
    return await Document.find_one(
        Document.url == task.url,
        Document.source == task.source,
    )


def _result_from_fetch(
    fetch: Fetch, *, status: Status, error: str | None = None
) -> FetchStageResult:
    response = fetch.response
    request = fetch.request
    return FetchStageResult(
        name="fetch",
        status=status,
        error=error,
        status_code=response.status_code if response else None,
        requested_url=fetch.url,
        fetched_url=request.url if request else None,
        response_headers=dict(response.headers) if response else {},
        content_type=fetch.content_type,
        body_path=fetch.body_path,
    )


class FetchStage(Stage):
    """Download ``task.url`` and persist the body under ``FetchConfig.body_save_dir``."""

    name = "fetch"

    def __init__(self, config: FetchConfig | None = None) -> None:
        self.config = config or FetchConfig()

    async def process(
        self,
        task: Task,
        params: dict[str, Any],
        workflow_name: str,
        workflow_node_name: str,
    ) -> tuple[StageResult, dict[str, Any] | None]:
        if not task.url:
            result = FetchStageResult(
                name=self.name,
                status=Status.FAILED,
                error="Task.url is required",
            )
            return result, None

        existing = await Fetch.find_one(Fetch.url == task.url)
        if existing is not None and existing.successful:
            result = _result_from_fetch(existing, status=Status.COMPLETED)
            document = await _resolve_document(task)
            if document is not None:
                await document.push_stage_result(
                    workflow_name, workflow_node_name, result
                )
            return result, None

        fetch_overrides = params.get("fetch_params") or {}
        browser_overrides = params.get("browser_fetch_params") or {}
        reliable_params = ReliableFetchParams(
            fetch_params=AsyncFetchParams.model_validate(fetch_overrides),
            browser_fetch_params=BrowserFetchParams.model_validate(browser_overrides),
        )

        meta: ScraplingFetchResult | None = None
        error: str | None = None
        try:
            meta = await reliable_fetch_with_meta(task.url, reliable_params)
        except Exception as exc:
            logger.warning("Fetch failed for %s: %s", task.url, exc, exc_info=True)
            error = str(exc)

        successful = meta is not None and _is_successful_fetch(meta)
        content_type: ContentType | None = None
        if successful and meta is not None:
            content_type = infer_content_type(
                meta.body,
                url=meta.request_url,
                response_headers=meta.response_headers,
            )
            if content_type is None:
                successful = False
                error = "Could not infer content type"

        fetch = existing or Fetch(url=task.url, successful=False)
        if fetch.id is None:
            # Assign id before writing the blob so the filename is stable.
            fetch.id = PydanticObjectId()

        if successful and meta is not None and content_type is not None:
            ext = extension_for_content_type(content_type)
            root = self.config.body_save_dir.rstrip("/")
            body_path = f"{root}/{fetch.id}{ext}"
            _write_body_fsspec(body_path, meta.body)
            fetch.successful = True
            fetch.request = FetchRequest(
                fetcher=meta.fetcher,
                fetcher_params=meta.fetcher_params,
                url=meta.request_url,
                headers=meta.request_headers,
                params=meta.request_params,
            )
            fetch.response = FetchResponse(
                status_code=meta.status_code,
                headers=meta.response_headers,
                body_header=meta.body[:512],
            )
            fetch.content_type = content_type
            fetch.body_path = body_path
            fetch.error = None
        else:
            fetch.successful = False
            fetch.request = None
            fetch.response = None
            fetch.content_type = None
            fetch.body_path = None
            if error is None:
                error = "Fetch unsuccessful"
            fetch.error = error

        if existing is None:
            await fetch.insert()
        else:
            await fetch.save()

        status = Status.COMPLETED if fetch.successful else Status.FAILED
        result = _result_from_fetch(fetch, status=status, error=fetch.error)
        document = await _resolve_document(task)
        if document is not None:
            await document.push_stage_result(workflow_name, workflow_node_name, result)
        return result, None
