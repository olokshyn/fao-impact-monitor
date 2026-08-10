"""Content-addressed source artifacts and local PDF text/rendering."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf as _pymupdf

from fao_impact_monitor.pdf_pipeline.models import ArtifactRef, BoundingBox

pymupdf: Any = _pymupdf


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RenderedPage:
    physical_page: int
    width_points: float
    height_points: float
    rotation_degrees: int
    extracted_text: str
    artifact: ArtifactRef


class PdfArtifacts:
    """Stores one immutable artifact set below ``<artifact_dir>/<pdf sha>``."""

    def __init__(self, root: Path, content_sha256: str, dpi: int) -> None:
        self.root = root
        self.content_sha256 = content_sha256
        self.dpi = dpi
        self.directory = root / content_sha256

    def prepare(self, source: Path) -> tuple[ArtifactRef, list[RenderedPage]]:
        self.directory.mkdir(parents=True, exist_ok=True)
        pdf_path = self.directory / "source.pdf"
        if not pdf_path.exists():
            shutil.copyfile(source, pdf_path)
        source_artifact = ArtifactRef(
            relative_path=str(pdf_path.relative_to(self.root)),
            sha256=self.content_sha256,
            media_type="application/pdf",
        )
        return source_artifact, self._render(pdf_path)

    def _render(self, pdf_path: Path) -> list[RenderedPage]:
        pages_dir = self.directory / "pages"
        pages_dir.mkdir(exist_ok=True)
        scale = self.dpi / 72.0
        document = pymupdf.open(pdf_path)
        try:
            result: list[RenderedPage] = []
            for index, page in enumerate(document):
                image_path = pages_dir / f"page-{index + 1:04d}.png"
                if not image_path.exists():
                    page.get_pixmap(
                        matrix=pymupdf.Matrix(scale, scale), alpha=False
                    ).save(image_path)
                pixmap = pymupdf.Pixmap(image_path)
                artifact = ArtifactRef(
                    relative_path=str(image_path.relative_to(self.root)),
                    sha256=sha256_file(image_path),
                    media_type="image/png",
                    width_px=pixmap.width,
                    height_px=pixmap.height,
                )
                result.append(
                    RenderedPage(
                        physical_page=index + 1,
                        width_points=float(page.rect.width),
                        height_points=float(page.rect.height),
                        rotation_degrees=int(page.rotation),
                        extracted_text=page.get_text("text").strip(),
                        artifact=artifact,
                    )
                )
            return result
        finally:
            document.close()

    def section_pdf(self, physical_pages: Sequence[int]) -> Path:
        """Create a compact PDF whose pages map to the requested source pages."""
        ordered_pages = list(dict.fromkeys(physical_pages))
        if not ordered_pages:
            raise ValueError("section PDF requires at least one physical page")
        key = ",".join(str(page) for page in ordered_pages)
        digest = hashlib.sha256(key.encode("ascii")).hexdigest()[:16]
        section_dir = self.directory / "section-pdfs"
        section_dir.mkdir(exist_ok=True)
        section_path = section_dir / (
            f"pages-{ordered_pages[0]:04d}-{ordered_pages[-1]:04d}-{digest}.pdf"
        )
        if section_path.exists():
            return section_path

        source = pymupdf.open(self.directory / "source.pdf")
        excerpt = pymupdf.open()
        temporary_path = section_path.with_name(
            f".{section_path.name}.{os.getpid()}.tmp"
        )
        try:
            page_count = len(source)
            for physical_page in ordered_pages:
                if not 1 <= physical_page <= page_count:
                    raise ValueError(
                        f"physical page {physical_page} outside 1-{page_count}"
                    )
                excerpt.insert_pdf(
                    source,
                    from_page=physical_page - 1,
                    to_page=physical_page - 1,
                )
            excerpt.save(temporary_path, garbage=4, deflate=True)
            temporary_path.replace(section_path)
        finally:
            excerpt.close()
            source.close()
            temporary_path.unlink(missing_ok=True)
        return section_path

    def crop(self, physical_page: int, bbox: BoundingBox) -> ArtifactRef | None:
        """Render a source-region crop; coordinates are normalized PDF points."""
        pdf_path = self.directory / "source.pdf"
        crops_dir = self.directory / "crops"
        crops_dir.mkdir(exist_ok=True)
        name = f"p{physical_page:04d}-{bbox.x0:.1f}-{bbox.y0:.1f}-{bbox.x1:.1f}-{bbox.y1:.1f}.png"
        path = crops_dir / name
        if not path.exists():
            document = pymupdf.open(pdf_path)
            try:
                page = document[physical_page - 1]
                clip = pymupdf.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1) & page.rect
                if clip.is_empty or clip.width <= 0 or clip.height <= 0:
                    return None
                page.get_pixmap(
                    matrix=pymupdf.Matrix(self.dpi / 72.0, self.dpi / 72.0),
                    clip=clip,
                    alpha=False,
                ).save(path)
            finally:
                document.close()
        pixmap = pymupdf.Pixmap(path)
        return ArtifactRef(
            relative_path=str(path.relative_to(self.root)),
            sha256=sha256_file(path),
            media_type="image/png",
            width_px=pixmap.width,
            height_px=pixmap.height,
        )
