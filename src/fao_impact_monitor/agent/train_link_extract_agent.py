"""Train LinkExtractAgent with DSPy BootstrapFewShot and save compiled state.

Usage:
    uv run train-link-extract-agent
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import dspy
import typer
from dspy.teleprompt import BootstrapFewShot

from fao_impact_monitor.agent.link_extract_agent import (
    ExtractLinks,
    LinkExtractAgent,
    configure_dspy_lm,
    save_link_extract_agent,
    validate_urls_in_body,
)
from fao_impact_monitor.agent.link_extract_train_data import load_link_extract_examples
from fao_impact_monitor.config import get_config

app = typer.Typer(add_completion=False, no_args_is_help=False)


class _ExtractProgram(dspy.Module):  # type: ignore[misc]
    """Thin program used only for BootstrapFewShot (no retry loop)."""

    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractLinks)

    def forward(
        self,
        page_url: str,
        page_body: str,
        topic: str,
        data_filter: str,
        max_urls: int,
        correction: str = "",
    ) -> dspy.Prediction:
        return self.extract(
            page_url=page_url,
            page_body=page_body,
            topic=topic,
            data_filter=data_filter,
            max_urls=max_urls,
            correction=correction,
        )


def _predicted_urls(prediction: Any) -> list[str]:
    predicted = list(getattr(prediction, "urls", None) or [])
    return [url for url in predicted if isinstance(url, str)]


def _exact_match_metric(
    example: dspy.Example,
    prediction: Any,
    trace: Any | None = None,
) -> bool:
    del trace
    expected = list(example.urls)
    return _predicted_urls(prediction) == expected


def build_trainset() -> list[dspy.Example]:
    """Build DSPy examples from ``train_data/link_extract_agent``."""
    config = get_config().link_extract
    trainset: list[dspy.Example] = []
    for item in load_link_extract_examples():
        missing = validate_urls_in_body(item["urls"], item["page_body"])
        if missing:
            raise ValueError(
                f"Gold URLs missing from HTML for {item['id']}: {missing[:3]}"
            )
        example = dspy.Example(
            example_id=item["id"],
            page_url=item["page_url"],
            page_body=item["page_body"],
            topic=item["topic"],
            data_filter=item["data_filter"],
            max_urls=max(len(item["urls"]), config.max_urls_per_page),
            correction="",
            urls=item["urls"],
        ).with_inputs(
            "page_url",
            "page_body",
            "topic",
            "data_filter",
            "max_urls",
            "correction",
        )
        trainset.append(example)
    return trainset


def _multiset_diff(left: list[str], right: list[str]) -> list[str]:
    """Return items in ``left`` not accounted for by ``right`` (multiset)."""
    remaining = Counter(right)
    diff: list[str] = []
    for url in left:
        if remaining[url] > 0:
            remaining[url] -= 1
        else:
            diff.append(url)
    return diff


def _report_url_mismatches(
    example: dspy.Example,
    predicted: list[str],
) -> None:
    """Print which URLs were wrong for a failed example."""
    expected = list(example.urls)
    example_id = getattr(example, "example_id", None) or example.page_url
    typer.echo(f"  Mismatch on example {example_id!r}:")

    missing = _multiset_diff(expected, predicted)
    unexpected = _multiset_diff(predicted, expected)
    if missing:
        typer.echo(f"    Missing ({len(missing)}):")
        for url in missing:
            typer.echo(f"      - {url}")
    if unexpected:
        typer.echo(f"    Unexpected ({len(unexpected)}):")
        for url in unexpected:
            typer.echo(f"      - {url}")
    if not missing and not unexpected and expected != predicted:
        typer.echo("    Wrong order (same URLs, different sequence):")
        for index, (want, got) in enumerate(zip(expected, predicted, strict=True)):
            if want != got:
                typer.echo(f"      [{index}] expected={want}")
                typer.echo(f"           predicted={got}")


def evaluate_on_trainset(
    agent: LinkExtractAgent,
    trainset: list[dspy.Example],
    *,
    max_retries: int,
) -> tuple[int, int]:
    """Return ``(correct, total)`` exact-match counts on ``trainset``."""
    correct = 0
    for example in trainset:
        prediction = agent(
            page_url=example.page_url,
            page_body=example.page_body,
            topic=example.topic,
            data_filter=example.data_filter,
            max_urls=int(example.max_urls),
            max_retries=max_retries,
        )
        if _exact_match_metric(example, prediction):
            correct += 1
        else:
            _report_url_mismatches(example, _predicted_urls(prediction))
    return correct, len(trainset)


def train_and_save(state_path: Path | None = None) -> Path:
    """Optimize LinkExtractAgent and write compiled state."""
    config = get_config().link_extract
    out = Path(state_path) if state_path is not None else Path(config.dspy_state_path)
    configure_dspy_lm()
    trainset = build_trainset()
    teleprompter = BootstrapFewShot(
        metric=_exact_match_metric,
        max_bootstrapped_demos=min(2, len(trainset)),
        max_labeled_demos=len(trainset),
    )
    compiled = teleprompter.compile(_ExtractProgram(), trainset=trainset)
    agent = LinkExtractAgent()
    agent.extract = compiled.extract
    correct, total = evaluate_on_trainset(
        agent,
        trainset,
        max_retries=config.max_agent_retries,
    )
    accuracy = (correct / total) if total else 0.0
    typer.echo(f"Train exact-match: {correct}/{total} ({accuracy:.1%})")
    save_link_extract_agent(agent, out)
    typer.echo(f"Saved trained LinkExtractAgent to {out}")
    return out


@app.command()
def main(
    state_path: Annotated[
        Path | None,
        typer.Option(
            "--state-path",
            help=(
                "Where to write compiled DSPy state "
                "(default: LinkExtractConfig.dspy_state_path)"
            ),
        ),
    ] = None,
) -> None:
    """Train LinkExtractAgent from train_data/link_extract_agent examples."""
    train_and_save(state_path)


if __name__ == "__main__":
    app()
