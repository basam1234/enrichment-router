from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import quote

import requests

from ..models import EnrichmentRequest, FieldName, ResolvedField
from .heuristic import TLD_TO_COUNTRY

WIKIPEDIA_REST_BASE: str = "https://en.wikipedia.org/api/rest_v1/page/summary/"
TIER1_DECLARED_COST_USD: float = 0.0
TIER1_DECLARED_LATENCY_MS: float = 600.0

# Wikipedia API blocks default python-requests User-Agents to prevent
# scrapers. A custom User-Agent is required for production traffic.
HEADERS: dict[str, str] = {
    "User-Agent": "EnrichmentRouter/1.0 (https://github.com/your-repo/enrichment-router)"
}

# Confidence values for tier-1 resolutions. short_description is the
# most reliable because it comes from Wikipedia's curated page summary
# field. Country extraction from the article extract is less precise
# due to ambiguous mentions, hence 0.55. is_public has the lowest
# confidence (0.5) because keyword matching against phrases like
# "publicly traded" can miss reworded descriptions or match stale text.
CONF_SHORT_DESCRIPTION: float = 0.85
CONF_COUNTRY: float = 0.55
CONF_IS_PUBLIC: float = 0.5

PUBLIC_INDICATORS: tuple[str, ...] = (
    "publicly traded",
    "public company",
    "nyse",
    "nasdaq",
)
PRIVATE_INDICATORS: tuple[str, ...] = (
    "privately held",
    "private company",
)

# Deduplicated country-name list derived from the tier-0 TLD table.
# Sorted for deterministic iteration order across Python versions; the
# sort order does not affect _match_country's behavior (it checks all
# matches and filters by substring containment, not by priority).
COUNTRY_NAMES: list[str] = sorted(set(TLD_TO_COUNTRY.values()), key=lambda n: (-len(n), n))


@dataclass
class WikipediaResult:
    resolved: dict[FieldName, ResolvedField] = field(default_factory=dict)
    cost_usd: float = 0.0
    measured_latency_ms: float = 0.0
    not_found: bool = False


def _default_fetcher(name: str) -> dict | None:
    url = WIKIPEDIA_REST_BASE + quote(name)
    resp = requests.get(url, timeout=10, headers=HEADERS)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _match_country(extract: str) -> str | None:
    """Return the unique country name found in extract, or None.

    When one matched country name is a substring of another (e.g.,
    "Guinea" is a substring of "Equatorial Guinea"), only the longest
    match is counted. This prevents false ambiguity from nested country
    names — without it, a mention of "Equatorial Guinea" would also
    match "Guinea," pushing the count to 2 and reporting unresolved.

    Case-sensitive matching is used because country names are proper
    nouns — they appear capitalized in Wikipedia extracts, and
    case-folding would risk false positives against lowercase common
    words (e.g., "china" the material vs "China" the country).
    """
    matches = [name for name in COUNTRY_NAMES if name in extract]
    if not matches:
        return None
    filtered = [m for m in matches if not any(m != other and m in other for other in matches)]
    if len(filtered) == 1:
        return filtered[0]
    return None


def _match_is_public(extract: str) -> bool | None:
    lowered = extract.lower()
    for phrase in PUBLIC_INDICATORS:
        if phrase in lowered:
            return True
    for phrase in PRIVATE_INDICATORS:
        if phrase in lowered:
            return False
    return None


def enrich(
    request: EnrichmentRequest,
    fetcher: Optional[Callable[[str], Optional[dict]]] = None,
) -> WikipediaResult:
    """Run the tier-1 Wikipedia lookup against request.fields_needed.

    fetcher lets tests inject a canned response without monkeypatching
    requests globally. Production callers leave it None.
    """
    fetch = fetcher or _default_fetcher
    start = time.perf_counter()
    data = fetch(request.name)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if data is None:
        return WikipediaResult(
            resolved={},
            cost_usd=0.0,
            measured_latency_ms=elapsed_ms,
            not_found=True,
        )

    resolved: dict[FieldName, ResolvedField] = {}
    extract: str = data.get("extract", "") or ""
    description: str = data.get("description", "") or ""

    if "short_description" in request.fields_needed and description:
        resolved["short_description"] = ResolvedField(
            name="short_description",
            value=description,
            tier=1,
            confidence=CONF_SHORT_DESCRIPTION,
        )

    if "country" in request.fields_needed and extract:
        country = _match_country(extract)
        if country is not None:
            resolved["country"] = ResolvedField(
                name="country",
                value=country,
                tier=1,
                confidence=CONF_COUNTRY,
            )

    if "is_public" in request.fields_needed and extract:
        is_pub = _match_is_public(extract)
        if is_pub is not None:
            resolved["is_public"] = ResolvedField(
                name="is_public",
                value=is_pub,
                tier=1,
                confidence=CONF_IS_PUBLIC,
            )

    return WikipediaResult(
        resolved=resolved,
        cost_usd=0.0,
        measured_latency_ms=elapsed_ms,
        not_found=False,
    )
