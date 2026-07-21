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
CONF_SHORT_DESCRIPTION: float = 0.85
CONF_COUNTRY: float = 0.65
CONF_IS_PUBLIC: float = 0.65
CONF_INDUSTRY: float = 0.65
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
# Strict keyword map to ensure high certainty extraction.
# If these keywords are found in the description or extract, we resolve the industry.
# If none are found, we return None, allowing escalation to Tier 2.
INDUSTRY_MAP: dict[str, str] = {
    "financial services": "Finance",
    "banking": "Finance",
    "venture capital": "Finance",
    "private equity": "Finance",
    "software": "Technology",
    "technology": "Technology",
    "artificial intelligence": "Technology",
    "app": "Technology",
    "application": "Technology",
    "platform": "Technology",
    "tool": "Technology",
    "saas": "Technology",
    "internet": "Technology",
    "research": "Technology",
    "pharmaceutical": "Healthcare",
    "biotechnology": "Healthcare",
    "healthcare": "Healthcare",
    "automotive": "Industrial/Robotics",
    "robotics": "Industrial/Robotics",
    "food": "Food & Beverage",
    "beverage": "Food & Beverage",
    "retail": "Retail",
    "e-commerce": "Retail",
    "media": "Media",
    "entertainment": "Media",
    "energy": "Energy",
}
# Deduplicated country-name list derived from the tier-0 TLD table.
COUNTRY_NAMES: list[str] = sorted(set(TLD_TO_COUNTRY.values()), key=lambda n: (-len(n), n))

# Custom User-Agent to prevent Wikipedia from blocking requests with a 403.
HEADERS: dict[str, str] = {
    "User-Agent": "EnrichmentRouter/1.0 (https://github.com/your-repo/enrichment-router)"
}


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


def _is_company_page(data: dict) -> bool:
    """Check if the Wikipedia page is actually about a company.

    Wikipedia will return pages for concepts, animals, or disambiguation
    pointers if a company name is also a common word (e.g., "Meta", "Meow").
    We reject these to prevent non-company data from polluting the enrichment.
    """
    desc = (data.get("description") or "").lower()
    extract = (data.get("extract") or "").lower()
    text = f"{desc} {extract}"

    # Reject disambiguation or generic term pages
    if "disambiguation" in text or "referred to by" in text or "may refer to" in text:
        return False

    # Must contain at least one organizational keyword to be treated as a company
    company_keywords = (
        "company",
        "corporation",
        "firm",
        "inc.",
        "ltd.",
        "llc",
        "organization",
        "subsidiary",
        "startup",
        "platform",
        "service",
        "provider",
        "agency",
        "chain",
        "business",
        "enterprise",
        "brand",
        "retailer",
        "bank",
        "software",
        "app",
        "application",
        "tool",
        "website",
        "technology",
        "system",
        "network",
        "developer",
        "manufacturer",
    )
    return any(kw in text for kw in company_keywords)


def _match_industry(extract: str, description: str) -> str | None:
    """Extract an industry label strictly from known keywords.
    Returns None if no known industry is found, allowing escalation.
    """
    text = f"{description} {extract}".lower()
    for key, label in INDUSTRY_MAP.items():
        if key in text:
            return label
    return None


def _match_country(extract: str) -> str | None:
    """Return the unique country name found in extract, or None.

    When one matched country name is a substring of another (e.g.,
    "Guinea" is a substring of "Equatorial Guinea"), only the longest
    match is counted. This prevents false ambiguity from nested country
    names.
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

    # Validate the page is actually about a company. If not, treat as not_found
    # so the system cleanly escalates to Tier 2 instead of using bad data.
    if not _is_company_page(data):
        return WikipediaResult(
            resolved={},
            cost_usd=0.0,
            measured_latency_ms=elapsed_ms,
            not_found=True,
        )

    resolved: dict[FieldName, ResolvedField] = {}
    extract: str = data.get("extract", "") or ""
    description: str = data.get("description", "") or ""

    if "industry" in request.fields_needed and (extract or description):
        industry = _match_industry(extract, description)
        if industry is not None:
            resolved["industry"] = ResolvedField(
                name="industry",
                value=industry,
                tier=1,
                confidence=CONF_INDUSTRY,
            )

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
