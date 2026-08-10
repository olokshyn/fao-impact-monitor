"""Standalone, provenance-first ingestion and retrieval for rich FAO PDFs."""

from fao_impact_monitor.pdf_pipeline.retrieval import PdfEvidenceVectorStore

__all__ = ["PdfEvidenceVectorStore"]
