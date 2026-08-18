from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from threading import Lock
from typing import Any, Final

import pandas as pd
from pydantic import ConfigDict, Field

from fao_impact_monitor.metric.metric import Metric

from .data_source import DataResult, DataSource
from .data_source_config import DataSourceConfig

logger = logging.getLogger(__name__)

DESINVENTAR_URL: Final = "https://www.desinventar.net/DesInventar/index.jsp"
UNSPECIFIED_REGION: Final = "Unspecified"

DEFAULT_EVENT_KEYWORDS: Final[tuple[str, ...]] = (
    "flood",
    "rain",
    "storm",
    "drought",
    "cyclone",
    "wind",
    "landslide",
    "land slide",
    "mudslide",
    "hail",
    "thunder",
    "lightning",
    "lightening",
    "heat wave",
    "cold wave",
    "frost",
    "dry spell",
    "forest fire",
    "wild fire",
    "bush fire",
    "napolo",
)

# Local-language synonyms for the English climate-hazard keywords above.
# Observed DesInventar `evento` languages in this repo: Spanish (GTM/HND/NIC/SLV),
# Portuguese (MOZ), French (DJI), Tetum (TLS). Used only when English matching
# returns no rows.
EVENT_KEYWORD_LOCAL_SYNONYMS: Final[dict[str, tuple[str, ...]]] = {
    "flood": (
        "inundación",
        "inundacao",
        "inundação",
        "cheias",
        "enxurrada",
        "inondation",
        "inundasaun",
        "spate",
    ),
    "rain": (
        "lluvias",
        "chuvas",
        "pluies",
        "udan",
    ),
    "storm": (
        "tempestad",
        "tempestade",
        "tempête",
        "tempete",
        "tormenta",
        "marejada",
        "laloran",
    ),
    "drought": (
        "sequía",
        "sequia",
        "seca",
        "sécheresse",
        "secheresse",
    ),
    "cyclone": (
        "huracán",
        "huracan",
        "ciclone",
        "depressao",
        "depressão",
    ),
    "wind": (
        "vendaval",
        "anin bot",
    ),
    "landslide": (
        "deslizamiento",
        "deslizamento",
        "glissement",
        "alud",
        "aluvión",
        "aluvion",
        "avenida torrencial",
        "flujo de escombros",
        "lahar",
        "rai monu",
    ),
    "land slide": (
        "deslizamiento",
        "deslizamento",
        "glissement",
        "alud",
        "aluvión",
        "aluvion",
        "avenida torrencial",
        "flujo de escombros",
        "lahar",
        "rai monu",
    ),
    "mudslide": (
        "aluvión",
        "aluvion",
        "alud",
        "flujo de escombros",
    ),
    "hail": (
        "granizada",
        "granizo",
        "grele",
        "grêle",
    ),
    "thunder": (
        "tormenta eléctrica",
        "tormenta electrica",
        "trovoada",
        "descarga electrica",
        "descarga elétrica",
    ),
    "lightning": (
        "tormenta eléctrica",
        "tormenta electrica",
        "foudre",
        "descarga electrica",
        "descarga elétrica",
    ),
    "lightening": (
        "tormenta eléctrica",
        "tormenta electrica",
        "foudre",
        "descarga electrica",
        "descarga elétrica",
    ),
    "heat wave": (
        "ola de calor",
        "onda calor",
        "vague de chaleur",
    ),
    "cold wave": (
        "frente frío",
        "frente frio",
        "vague de froid",
    ),
    "frost": ("helada",),
    "forest fire": (
        "incendio forestal",
        "queimadas",
        "ahi han",
    ),
    "wild fire": (
        "incendio forestal",
        "queimadas",
        "ahi han",
    ),
    "bush fire": (
        "incendio forestal",
        "queimadas",
        "ahi han",
    ),
}

# Numeric count fields vs sector flags that use -1 for "yes".
_FLAG_INDICATORS: Final[frozenset[str]] = frozenset(
    {
        "transporte",
        "energia",
        "acueducto",
        "alcantarillado",
        "comunicaciones",
        "salud",
        "educacion",
    }
)

KNOWN_INDICATORS: Final[frozenset[str]] = frozenset(
    {
        "muertos",
        "desaparece",
        "heridos",
        "afectados",
        "damnificados",
        "evacuados",
        "reubicados",
        "vivdest",
        "vivafec",
        "nescuelas",
        "nhospitales",
        "kmvias",
        "transporte",
        "energia",
        "acueducto",
        "alcantarillado",
        "comunicaciones",
    }
)

_FICHA_FIELDS: Final[tuple[str, ...]] = (
    "clave",
    "evento",
    "fechano",
    "fechames",
    "fechadia",
    "lugar",
    "level0",
    "name0",
    "name1",
    "name2",
    "muertos",
    "desaparece",
    "heridos",
    "afectados",
    "damnificados",
    "evacuados",
    "reubicados",
    "vivdest",
    "vivafec",
    "nescuelas",
    "nhospitales",
    "kmvias",
    "transporte",
    "energia",
    "acueducto",
    "alcantarillado",
    "comunicaciones",
    "salud",
    "educacion",
)


class DesInventarDataResult(DataResult):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame


class DesInventarDataSourceConfig(DataSourceConfig):
    indicator: str
    data_dir: Path = Path("datasets/DesInventar")
    data_path: Path | None = None
    event_keywords: list[str] = Field(
        default_factory=lambda: list(DEFAULT_EVENT_KEYWORDS)
    )


class DesInventar(DataSource):
    source: str = "DesInventar"
    DROP_ZERO_VALUES: bool = True

    def __init__(self) -> None:
        self._xml_cache: dict[tuple[str, int], pd.DataFrame] = {}
        self._cache_lock = Lock()

    async def get_data(
        self,
        metric: Metric,
        data_source_config: DataSourceConfig,
        country_iso3: str,
        *,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> list[DataResult]:
        del metric
        config = DesInventarDataSourceConfig.model_validate(
            data_source_config.model_dump()
        )
        return await asyncio.to_thread(
            self._get_data_sync,
            config,
            country_iso3,
            year_start,
            year_end,
        )

    def _get_data_sync(
        self,
        config: DesInventarDataSourceConfig,
        country_iso3: str,
        year_start: int | None,
        year_end: int | None,
    ) -> list[DataResult]:
        if year_start is not None and year_end is not None and year_start > year_end:
            raise ValueError(
                f"year_start ({year_start}) must be <= year_end ({year_end})"
            )
        if config.indicator not in KNOWN_INDICATORS:
            raise ValueError(
                f"Unknown DesInventar indicator {config.indicator!r}; "
                f"known indicators: {', '.join(sorted(KNOWN_INDICATORS))}"
            )

        data_path = resolve_country_xml_path(
            country_iso3,
            data_dir=config.data_dir,
            data_path=config.data_path,
        )
        if not data_path.is_file():
            logger.warning("DesInventar XML not found: %s", data_path)
            return []

        fichas = self._load_fichas(data_path)
        if fichas.empty:
            return []

        results = self._build_results(
            fichas,
            config,
            country_iso3,
            year_start=year_start,
            year_end=year_end,
            keywords=list(config.event_keywords),
            data_path=data_path,
        )
        if results or not config.event_keywords:
            return results

        local_keywords = local_event_keyword_synonyms(config.event_keywords)
        if not local_keywords:
            return []
        return self._build_results(
            fichas,
            config,
            country_iso3,
            year_start=year_start,
            year_end=year_end,
            keywords=local_keywords,
            data_path=data_path,
            requested_keywords=list(config.event_keywords),
        )

    def _build_results(
        self,
        fichas: pd.DataFrame,
        config: DesInventarDataSourceConfig,
        country_iso3: str,
        *,
        year_start: int | None,
        year_end: int | None,
        keywords: list[str],
        data_path: Path,
        requested_keywords: list[str] | None = None,
    ) -> list[DataResult]:
        selected = filter_fichas_by_event_keywords(fichas, keywords)
        selected = _filter_years(selected, year_start, year_end)
        if selected.empty:
            return []

        values = selected[config.indicator].map(
            lambda raw: parse_indicator_value(raw, config.indicator)
        )
        selected = selected.assign(value=values)
        selected = selected[selected["value"].notna()].copy()
        if self.DROP_ZERO_VALUES:
            selected = selected[selected["value"] != 0].copy()
        if selected.empty:
            return []

        data = pd.DataFrame(
            {
                "country_iso3": country_iso3.upper(),
                "event_id": selected["clave"].astype(str),
                "disaster_type": selected["evento"].astype(str),
                "location": selected["lugar"].map(_clean_text),
                "regions": selected.apply(regions_from_ficha, axis=1),
                "start_year": pd.to_numeric(selected["fechano"], errors="coerce"),
                "start_month": pd.to_numeric(selected["fechames"], errors="coerce"),
                "start_day": pd.to_numeric(selected["fechadia"], errors="coerce"),
                "value": selected["value"].astype(float),
            }
        )
        data = data.dropna(subset=["start_year"]).reset_index(drop=True)
        if data.empty:
            return []
        data["start_year"] = data["start_year"].astype(int)
        data = data.sort_values(
            by=["start_year", "start_month", "start_day", "event_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)

        return [
            DesInventarDataResult(
                source=self.source,
                title=config.indicator,
                url=DESINVENTAR_URL,
                citation=(
                    f'DesInventar. "{config.indicator}". UNDRR. {DESINVENTAR_URL}'
                ),
                metadata={
                    "indicator": config.indicator,
                    "country_iso3": country_iso3.upper(),
                    "year_start": year_start,
                    "year_end": year_end,
                    "unit": config.unit,
                    "event_keywords": list(
                        requested_keywords
                        if requested_keywords is not None
                        else config.event_keywords
                    ),
                    "event_keywords_applied": list(keywords),
                    "data_path": str(data_path),
                    "row_count": len(data),
                },
                data=data,
            )
        ]

    def _load_fichas(self, data_path: Path) -> pd.DataFrame:
        resolved = data_path.resolve()
        cache_key = (str(resolved), resolved.stat().st_mtime_ns)
        with self._cache_lock:
            cached = self._xml_cache.get(cache_key)
            if cached is not None:
                return cached
            data = parse_fichas_xml(resolved)
            self._xml_cache[cache_key] = data
            return data


def resolve_country_xml_path(
    country_iso3: str,
    *,
    data_dir: Path = Path("datasets/DesInventar"),
    data_path: Path | None = None,
) -> Path:
    """Resolve the per-country DesInventar XML path."""
    if data_path is not None:
        return data_path
    code = country_iso3.strip().lower()
    return data_dir / f"DI_export_{code}" / f"DI_export_{code}.xml"


def parse_fichas_xml(path: Path) -> pd.DataFrame:
    """Parse only ``<fichas><TR>`` datacards from a DesInventar export XML."""
    rows: list[dict[str, str]] = []
    in_fichas = False
    in_tr = False
    current: dict[str, str] = {}

    for event, elem in ET.iterparse(path, events=("start", "end")):
        tag = elem.tag
        if event == "start":
            if tag == "fichas":
                in_fichas = True
            elif in_fichas and tag == "TR":
                in_tr = True
                current = {field: "" for field in _FICHA_FIELDS}
            continue

        if tag == "fichas":
            in_fichas = False
            elem.clear()
            continue

        if not in_fichas:
            elem.clear()
            continue

        if in_tr and tag != "TR" and tag in current:
            current[tag] = (elem.text or "").strip()
            elem.clear()
            continue

        if tag == "TR" and in_tr:
            rows.append(current)
            in_tr = False
            current = {}
            elem.clear()
            continue

        elem.clear()

    if not rows:
        return pd.DataFrame(columns=list(_FICHA_FIELDS))
    return pd.DataFrame(rows)


def matches_event_keywords(evento: Any, keywords: list[str]) -> bool:
    """Return True when the event name matches any climate-hazard keyword.

    Single-token keywords match a whole token, a token prefix (``flood`` →
    ``floods``), or a token suffix of length >= 5 (``storm`` →
    ``thunderstorm``). That avoids false positives such as ``rain`` inside
    ``terrain``. Multi-word keywords match as phrases with word boundaries.
    """
    text = _clean_text(evento).casefold()
    if not text:
        return False
    tokens = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    for keyword in keywords:
        kw = keyword.casefold().strip()
        if not kw:
            continue
        if " " in kw:
            pattern = r"(?<!\w)" + re.escape(kw).replace(r"\ ", r"\s+") + r"(?!\w)"
            if re.search(pattern, text):
                return True
            continue
        for token in tokens:
            if token == kw or token.startswith(kw):
                return True
            if len(kw) >= 5 and token.endswith(kw):
                return True
    return False


def local_event_keyword_synonyms(english_keywords: list[str]) -> list[str]:
    """Collect local-language synonyms for the given English event keywords."""
    synonyms: list[str] = []
    seen: set[str] = set()
    for keyword in english_keywords:
        key = keyword.casefold().strip()
        if not key:
            continue
        for synonym in EVENT_KEYWORD_LOCAL_SYNONYMS.get(key, ()):
            folded = synonym.casefold()
            if folded in seen:
                continue
            seen.add(folded)
            synonyms.append(synonym)
    return synonyms


def filter_fichas_by_event_keywords(
    fichas: pd.DataFrame,
    keywords: list[str],
) -> pd.DataFrame:
    """Filter fichas to rows whose ``evento`` matches any of ``keywords``."""
    if not keywords:
        return fichas
    return fichas[
        fichas["evento"].map(lambda event: matches_event_keywords(event, keywords))
    ].copy()


def parse_indicator_value(raw: Any, indicator: str) -> float | None:
    """Parse a DesInventar field; keep 0; drop only missing/unparseable.

    Flag indicators map ``-1`` (yes) to ``1`` and keep ``0`` as ``0``.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip().replace(",", "")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if indicator in _FLAG_INDICATORS:
        if value == 0:
            return 0.0
        return 1.0
    return value


def regions_from_ficha(row: pd.Series) -> list[str]:
    """Prefer name0, then name1, then name2, then lugar."""
    for key in ("name0", "name1", "name2", "lugar"):
        cleaned = _clean_text(row.get(key))
        if cleaned:
            return [cleaned]
    return [UNSPECIFIED_REGION]


def national_totals_by_year(data: pd.DataFrame) -> pd.DataFrame:
    """Sum event values by start year."""
    if data.empty:
        return pd.DataFrame(columns=["year", "value"])
    totals = (
        data.groupby("start_year", dropna=False, sort=False)["value"]
        .sum()
        .rename_axis("year")
        .reset_index(name="value")
    )
    totals["year"] = totals["year"].astype("Int64")
    return totals.sort_values("year", ascending=False, kind="mergesort").reset_index(
        drop=True
    )


def subnational_totals_by_year(data: pd.DataFrame) -> pd.DataFrame:
    """Sum values by year and region set without splitting one datacard."""
    if data.empty:
        return pd.DataFrame(columns=["year", "region", "value"])

    rows: list[dict[str, Any]] = []
    for row in data.itertuples(index=False):
        year = getattr(row, "start_year", None)
        value = getattr(row, "value", None)
        if year is None or pd.isna(year) or value is None or pd.isna(value):
            continue
        regions = getattr(row, "regions", None)
        if not isinstance(regions, list) or not regions:
            regions = [UNSPECIFIED_REGION]
        rows.append(
            {
                "year": int(year),
                "region": format_region_set(regions),
                "value": float(value),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["year", "region", "value"])

    totals = (
        pd.DataFrame(rows)
        .groupby(["year", "region"], as_index=False, sort=False)
        .agg(value=("value", "sum"))
    )
    sorted_totals = totals.sort_values(
        by=["year", "region"],
        ascending=[False, True],
        kind="mergesort",
    )
    return sorted_totals.reset_index(drop=True)


def format_region_set(regions: list[str]) -> str:
    """Stable display key for one or more admin units belonging to the same total."""
    unique = list(dict.fromkeys(regions))
    if not unique:
        return UNSPECIFIED_REGION
    return "; ".join(unique)


def _filter_years(
    data: pd.DataFrame,
    year_start: int | None,
    year_end: int | None,
) -> pd.DataFrame:
    years = pd.to_numeric(data["fechano"], errors="coerce")
    selected = data
    if year_start is not None:
        selected = selected[years >= year_start]
        years = pd.to_numeric(selected["fechano"], errors="coerce")
    if year_end is not None:
        selected = selected[years <= year_end]
    return selected


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).split())
