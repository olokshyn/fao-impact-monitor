"""Load LinkExtractAgent train/eval examples from ``train_data/link_extract_agent``."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DATA_DIR = REPO_ROOT / "train_data" / "link_extract_agent"
EXAMPLES_PATH = TRAIN_DATA_DIR / "examples.json"


class LinkExtractExample(TypedDict):
    id: str
    page_url: str
    html_file: str
    urls: list[str]
    topic: str
    data_filter: str
    page_body: str


def load_examples_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load ``examples.json`` from the train_data directory."""
    manifest_path = path or EXAMPLES_PATH
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected object in {manifest_path}, got {type(data)}")
    return data


def load_link_extract_examples(
    train_dir: Path | None = None,
) -> list[LinkExtractExample]:
    """Return examples with HTML bodies loaded from disk."""
    root = train_dir or TRAIN_DATA_DIR
    manifest = load_examples_manifest(root / "examples.json")
    topic = str(manifest["topic"])
    data_filter = str(manifest["data_filter"])
    examples: list[LinkExtractExample] = []
    for item in manifest["examples"]:
        html_path = root / str(item["html_file"])
        page_body = html.unescape(
            html_path.read_text(encoding="utf-8", errors="replace")
        )
        urls = [str(url) for url in item["urls"]]
        examples.append(
            {
                "id": str(item["id"]),
                "page_url": str(item["page_url"]),
                "html_file": str(item["html_file"]),
                "urls": urls,
                "topic": topic,
                "data_filter": data_filter,
                "page_body": page_body,
            }
        )
    return examples
