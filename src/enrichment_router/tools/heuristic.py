from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..models import EnrichmentRequest, FieldName, ResolvedField

# Tier 0 is the cheapest enrichment tier. Confidence is intentionally
# low (0.35) to reflect that heuristic rules are educated guesses from
# name/keyword matching and ccTLD lookups — useful as a fallback but
# never authoritative. Cost is zero because these are pure-Python rules
# with no network calls, no external APIs, and no token usage.
TIER0_CONFIDENCE: float = 0.35
TIER0_DECLARED_COST_USD: float = 0.0
TIER0_DECLARED_LATENCY_MS: float = 10.0

TLD_TO_COUNTRY: dict[str, str] = {
    "us": "United States",
    "uk": "United Kingdom",
    "gb": "United Kingdom",
    "de": "Germany",
    "fr": "France",
    "jp": "Japan",
    "cn": "China",
    "in": "India",
    "br": "Brazil",
    "au": "Australia",
    "ca": "Canada",
    "mx": "Mexico",
    "es": "Spain",
    "it": "Italy",
    "nl": "Netherlands",
    "se": "Sweden",
    "no": "Norway",
    "dk": "Denmark",
    "fi": "Finland",
    "pl": "Poland",
    "ru": "Russia",
    "kr": "South Korea",
    "sg": "Singapore",
    "ae": "United Arab Emirates",
    "za": "South Africa",
    "ng": "Nigeria",
    "ie": "Ireland",
    "ch": "Switzerland",
    "at": "Austria",
    "be": "Belgium",
    "pt": "Portugal",
    "gr": "Greece",
    "nz": "New Zealand",
    "th": "Thailand",
    "vn": "Vietnam",
    "id": "Indonesia",
    "my": "Malaysia",
    "ph": "Philippines",
    "pk": "Pakistan",
    "bd": "Bangladesh",
    "lk": "Sri Lanka",
    "ua": "Ukraine",
    "cz": "Czech Republic",
    "hu": "Hungary",
    "ro": "Romania",
    "bg": "Bulgaria",
    "hr": "Croatia",
    "rs": "Serbia",
    "sk": "Slovakia",
    "si": "Slovenia",
    "lt": "Lithuania",
    "lv": "Latvia",
    "ee": "Estonia",
    "is": "Iceland",
    "lu": "Luxembourg",
    "mt": "Malta",
    "cy": "Cyprus",
    "sa": "Saudi Arabia",
    "qa": "Qatar",
    "kw": "Kuwait",
    "om": "Oman",
    "bh": "Bahrain",
    "jo": "Jordan",
    "lb": "Lebanon",
    "iq": "Iraq",
    "af": "Afghanistan",
    "np": "Nepal",
    "mm": "Myanmar",
    "kh": "Cambodia",
    "la": "Laos",
    "mn": "Mongolia",
    "kz": "Kazakhstan",
    "uz": "Uzbekistan",
    "az": "Azerbaijan",
    "ge": "Georgia",
    "am": "Armenia",
    "by": "Belarus",
    "md": "Moldova",
    "al": "Albania",
    "mk": "North Macedonia",
    "ba": "Bosnia and Herzegovina",
    "me": "Montenegro",
    "cl": "Chile",
    "ar": "Argentina",
    "pe": "Peru",
    "co": "Colombia",
    "ve": "Venezuela",
    "ec": "Ecuador",
    "bo": "Bolivia",
    "py": "Paraguay",
    "uy": "Uruguay",
    "cr": "Costa Rica",
    "pa": "Panama",
    "gt": "Guatemala",
    "hn": "Honduras",
    "sv": "El Salvador",
    "ni": "Nicaragua",
    "do": "Dominican Republic",
    "cu": "Cuba",
    "ma": "Morocco",
    "dz": "Algeria",
    "tn": "Tunisia",
    "ly": "Libya",
    "eg": "Egypt",
    "et": "Ethiopia",
    "ke": "Kenya",
    "tz": "Tanzania",
    "ug": "Uganda",
    "gh": "Ghana",
    "ci": "Ivory Coast",
    "sn": "Senegal",
    "cm": "Cameroon",
    "zw": "Zimbabwe",
    "zm": "Zambia",
    "mz": "Mozambique",
    "ao": "Angola",
    "na": "Namibia",
    "bw": "Botswana",
    "mg": "Madagascar",
    "rw": "Rwanda",
    "sd": "Sudan",
    "cg": "Republic of the Congo",
    "cd": "Democratic Republic of the Congo",
    "gn": "Guinea",
    "gq": "Equatorial Guinea",
    "gw": "Guinea-Bissau",
    "ml": "Mali",
    "bf": "Burkina Faso",
    "ne": "Niger",
    "td": "Chad",
    "cf": "Central African Republic",
    "ss": "South Sudan",
    "tr": "Turkey",
    "ir": "Iran",
    "sy": "Syria",
    "ye": "Yemen",
}

INDUSTRY_KEYWORDS: list[tuple[str, str]] = [
    ("health", "Healthcare"),
    ("med", "Healthcare"),
    ("bio", "Healthcare"),
    ("pharma", "Healthcare"),
    ("robot", "Industrial/Robotics"),
    ("auto", "Industrial/Robotics"),
    ("motors", "Industrial/Robotics"),
    ("foods", "Food & Beverage"),
    ("kitchen", "Food & Beverage"),
    ("bakery", "Food & Beverage"),
    ("brew", "Food & Beverage"),
    ("capital", "Finance"),
    ("partners", "Finance"),
    ("ventures", "Finance"),
    ("fund", "Finance"),
    ("labs", "Technology"),
    ("ai", "Technology"),
    ("tech", "Technology"),
    ("cloud", "Technology"),
    ("software", "Technology"),
]


@dataclass
class HeuristicResult:
    resolved: dict[FieldName, ResolvedField] = field(default_factory=dict)
    cost_usd: float = 0.0
    measured_latency_ms: float = 0.0


def _match_industry(name: str) -> str | None:
    lowered = name.lower()
    for keyword, label in INDUSTRY_KEYWORDS:
        if keyword in lowered:
            return label
    return None


def _match_country_from_domain(domain: str | None) -> str | None:
    """Extract the ccTLD from a domain and map it to a country name.

    Takes the last label of the domain (split on '.'), lowercases it,
    and looks it up in TLD_TO_COUNTRY. This works for both simple
    domains (foo.de -> "de" -> "Germany") and compound domains
    (foo.co.uk -> "uk" -> "United Kingdom") because the final label
    is the ccTLD in both cases. Generic TLDs like .com, .org, .net are
    not in TLD_TO_COUNTRY and return None.

    Returns None for unrecognized TLDs (no exception, no guess).
    """
    if not domain:
        return None
    cleaned = domain.strip().lower().lstrip(".")
    if not cleaned:
        return None
    parts = cleaned.split(".")
    last = parts[-1]
    return TLD_TO_COUNTRY.get(last)


def enrich(request: EnrichmentRequest) -> HeuristicResult:
    """Run tier-0 heuristics against the request's fields_needed.

    Only industry and country are ever resolved by this tier.
    is_public and short_description are always left unresolved.
    """
    start = time.perf_counter()
    resolved: dict[FieldName, ResolvedField] = {}

    if "industry" in request.fields_needed:
        industry = _match_industry(request.name)
        if industry is not None:
            resolved["industry"] = ResolvedField(
                name="industry",
                value=industry,
                tier=0,
                confidence=TIER0_CONFIDENCE,
            )

    if "country" in request.fields_needed and request.domain:
        country = _match_country_from_domain(request.domain)
        if country is not None:
            resolved["country"] = ResolvedField(
                name="country",
                value=country,
                tier=0,
                confidence=TIER0_CONFIDENCE,
            )

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return HeuristicResult(
        resolved=resolved,
        cost_usd=0.0,
        measured_latency_ms=elapsed_ms,
    )
