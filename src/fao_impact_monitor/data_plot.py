from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

_FAO_BLUE = "#116AAB"
_FAO_COLORS = (_FAO_BLUE, "#F2A900", "#5B8C3A", "#7A5195", "#D95F59", "#2A9D8F")
_POINTS_PER_PANEL = 20
_MAX_PANELS = 12


def plot_time_series(
    data: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    series_columns: tuple[str, ...] = (),
    default_unit: str = "",
) -> Path | None:
    """Write directly labelled time-series panels, or return None if too dense."""
    if not {"year", "value"}.issubset(data.columns):
        return None

    frame = data.dropna(subset=["year", "value"]).copy()
    if frame.empty:
        return None
    frame["year"] = frame["year"].astype(int)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"])
    if frame.empty:
        return None

    groups = _series_groups(frame, series_columns, default_unit)
    panels: list[tuple[str, str, pd.DataFrame]] = []
    for label, unit, series in groups:
        ordered = series.sort_values("year")
        for start in range(0, len(ordered), _POINTS_PER_PANEL):
            panels.append(
                (label, unit, ordered.iloc[start : start + _POINTS_PER_PANEL])
            )
    if not panels or len(panels) > _MAX_PANELS:
        return None

    figure_width = 10.5
    figure_height = 3.25 * len(panels) + 0.7
    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(figure_width, figure_height),
        squeeze=False,
        layout="constrained",
    )
    figure.patch.set_facecolor("white")
    figure.suptitle(title, fontsize=16, fontweight="bold", color="#253238")

    for index, ((label, unit, panel), axis) in enumerate(
        zip(panels, axes.flat, strict=True)
    ):
        _draw_panel(axis, panel, label=label, unit=unit, color=_FAO_COLORS[index % 6])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_comparison_time_series(
    data: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    series_column: str = "series",
    default_unit: str = "",
) -> Path | None:
    """Overlay multiple named series on shared axes with a legend."""
    if not {"year", "value", series_column}.issubset(data.columns):
        return None

    frame = data.dropna(subset=["year", "value", series_column]).copy()
    if frame.empty:
        return None
    frame["year"] = frame["year"].astype(int)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"])
    if frame.empty:
        return None

    series_names = list(dict.fromkeys(frame[series_column].astype(str).tolist()))
    if not series_names:
        return None

    units = (
        frame["unit"].dropna().astype(str).unique().tolist()
        if "unit" in frame.columns
        else []
    )
    unit = units[0] if len(units) == 1 else default_unit
    if len(units) > 1:
        # Different units: one panel per unit with overlaid series.
        panels: list[tuple[str, pd.DataFrame]] = []
        for unit_name, unit_frame in frame.groupby("unit", dropna=False, sort=True):
            panels.append((_label_value(unit_name) or default_unit, unit_frame))
    else:
        panels = [(unit, frame)]

    if len(panels) > _MAX_PANELS:
        return None

    figure_width = 10.5
    figure_height = 3.5 * len(panels) + 0.7
    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(figure_width, figure_height),
        squeeze=False,
        layout="constrained",
    )
    figure.patch.set_facecolor("white")
    figure.suptitle(title, fontsize=16, fontweight="bold", color="#253238")

    for panel_index, ((panel_unit, panel_frame), axis) in enumerate(
        zip(panels, axes.flat, strict=True)
    ):
        years_all: list[int] = []
        for series_index, series_name in enumerate(series_names):
            series = panel_frame[panel_frame[series_column].astype(str) == series_name]
            if series.empty:
                continue
            ordered = series.sort_values("year")
            years = ordered["year"].astype(int).tolist()
            values = ordered["value"].astype(float).tolist()
            years_all.extend(years)
            color = _FAO_COLORS[series_index % len(_FAO_COLORS)]
            axis.plot(
                years,
                values,
                color=color,
                linewidth=2.4,
                marker="o",
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1.2,
                label=series_name,
            )
        if not years_all:
            continue
        unique_years = sorted(set(years_all))
        axis.grid(axis="y", color="#DCE3E7", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#AAB7BE")
        axis.tick_params(colors="#4A5A61", labelsize=9)
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: _format_value(value))
        )
        axis.set_xticks(unique_years)
        axis.set_xticklabels(
            [str(year) for year in unique_years], rotation=45, ha="right"
        )
        axis.set_xlabel("Year", color="#4A5A61", fontweight="semibold")
        axis.set_ylabel(panel_unit or "Value", color="#4A5A61", fontweight="semibold")
        if len(panels) > 1 and panel_unit:
            axis.set_title(panel_unit, loc="left", fontsize=11, fontweight="semibold")
        axis.legend(loc="best", fontsize=9, frameon=False)
        axis.margins(x=0.035, y=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output_path


def _series_groups(
    frame: pd.DataFrame,
    series_columns: tuple[str, ...],
    default_unit: str,
) -> list[tuple[str, str, pd.DataFrame]]:
    usable_columns = tuple(column for column in series_columns if column in frame)
    if not usable_columns:
        return [("", default_unit, frame)]

    groups: list[tuple[str, str, pd.DataFrame]] = []
    grouper: str | list[str]
    grouper = usable_columns[0] if len(usable_columns) == 1 else list(usable_columns)
    for key, series in frame.groupby(grouper, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        unit = ""
        if "unit" in usable_columns:
            unit_index = usable_columns.index("unit")
            unit = _label_value(values[unit_index])
        labelled = [
            _label_value(value)
            for column, value in zip(usable_columns, values, strict=True)
            if column != "unit" and _label_value(value)
        ]
        groups.append((" — ".join(labelled), unit or default_unit, series))
    return groups


def _draw_panel(
    axis: Axes,
    data: pd.DataFrame,
    *,
    label: str,
    unit: str,
    color: str,
) -> None:
    years = data["year"].astype(int).tolist()
    values = data["value"].astype(float).tolist()
    axis.plot(
        years,
        values,
        color=color,
        linewidth=2.4,
        marker="o",
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=1.2,
    )
    axis.fill_between(
        years, values, [min(values)] * len(values), color=color, alpha=0.08
    )
    axis.grid(axis="y", color="#DCE3E7", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#AAB7BE")
    axis.tick_params(colors="#4A5A61", labelsize=9)
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: _format_value(value))
    )
    axis.set_xticks(years)
    axis.set_xticklabels([str(year) for year in years], rotation=45, ha="right")
    axis.set_xlabel("Year", color="#4A5A61", fontweight="semibold")
    axis.set_ylabel(unit or "Value", color="#4A5A61", fontweight="semibold")
    if label:
        axis.set_title(label, loc="left", fontsize=11, fontweight="semibold")

    for point_index, (year, value) in enumerate(zip(years, values, strict=True)):
        direction = 1 if point_index % 2 == 0 else -1
        axis.annotate(
            _format_value(value),
            xy=(year, value),
            xytext=(0, 8 if direction > 0 else -12),
            textcoords="offset points",
            ha="center",
            va="bottom" if direction > 0 else "top",
            fontsize=8,
            fontweight="semibold",
            color="#253238",
        )
    axis.margins(x=0.035, y=0.2)


def _format_value(value: float) -> str:
    if not math.isfinite(value):
        return ""
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3g}B"
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if magnitude >= 10_000:
        return f"{value / 1_000:.3g}K"
    if value.is_integer():
        return f"{value:,.0f}"
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def _label_value(value: object) -> str:
    label = str(value)
    return "" if label in {"", "<NA>", "nan", "None"} else label
