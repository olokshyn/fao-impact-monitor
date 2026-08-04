"""Thin async wrapper over the FAO Tellus search and document-chunk APIs."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from fao_impact_monitor.config import TellusConfig, get_config
from fao_impact_monitor.utils.country import iso3_to_country_name

logger = logging.getLogger(__name__)


class TellusAuthenticationError(EnvironmentError):
    """Tellus rejected or was not given an API bearer token."""


def _tellus_bearer_token(config: TellusConfig) -> str:
    token = config.bearer_token.get_secret_value().strip()
    if not token:
        raise TellusAuthenticationError("TELLUS_BEARER_TOKEN is not set")
    return token


def _raise_for_tellus_status(response: httpx.Response) -> None:
    if response.status_code in {401, 403}:
        raise TellusAuthenticationError(
            f"Tellus rejected TELLUS_BEARER_TOKEN (HTTP {response.status_code}); "
            "refresh the token before rerunning the pipeline"
        )
    response.raise_for_status()


async def tellus_search_chunks(
    query: str,
    countries_iso3: list[str] | None = None,
    *,
    config: TellusConfig | None = None,
) -> dict[str, Any]:
    """Search Tellus for chunks relevant to a semantic query and optional countries."""
    cfg = config or get_config().tellus
    headers = {"Authorization": f"Bearer {_tellus_bearer_token(cfg)}"}
    body: dict[str, Any] = {
        "semantic_query": query,
        "min_year": cfg.min_year,
        "max_results": cfg.max_results,
    }
    if countries_iso3:
        body["keyword_queries"] = [
            iso3_to_country_name(iso3) for iso3 in countries_iso3
        ]

    url = f"{cfg.api_base.rstrip('/')}/api/v1/search"
    logger.info(
        "Tellus search: query=%r max_results=%s min_year=%s countries=%s",
        query,
        cfg.max_results,
        cfg.min_year,
        countries_iso3 or [],
    )

    async with httpx.AsyncClient(timeout=cfg.timeout) as client:
        response = await client.post(url, headers=headers, json=body)
        _raise_for_tellus_status(response)
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError(f"Tellus search expected a JSON object, got {type(data)}")
        chunks = data.get("chunks") or []
        documents = data.get("documents") or []
        n_chunks = len(chunks) if isinstance(chunks, list) else 0
        n_docs = len(documents) if isinstance(documents, list) else 0
        logger.info(
            "Tellus search returned %s chunk(s) and %s document(s) for query=%r",
            n_chunks,
            n_docs,
            query,
        )
        return data


async def tellus_get_all_document_chunks(
    document_id: str,
    *,
    config: TellusConfig | None = None,
) -> dict[str, Any]:
    """Fetch all chunks (and document metadata) for one Tellus document."""
    cfg = config or get_config().tellus
    headers = {"Authorization": f"Bearer {_tellus_bearer_token(cfg)}"}
    url = f"{cfg.api_base.rstrip('/')}/api/v1/sources/document/{document_id}/chunks"
    logger.info("Tellus fetch all chunks for document_id=%s", document_id)

    async with httpx.AsyncClient(timeout=cfg.timeout) as client:
        response = await client.get(url, headers=headers)
        _raise_for_tellus_status(response)
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError(
                f"Tellus document chunks expected a JSON object, got {type(data)}"
            )
        chunks = data.get("chunks") or []
        n_chunks = len(chunks) if isinstance(chunks, list) else 0
        logger.info(
            "Tellus document %s returned %s chunk(s)",
            document_id,
            n_chunks,
        )
        return data
