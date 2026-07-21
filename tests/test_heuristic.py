from __future__ import annotations

from enrichment_router.models import EnrichmentRequest
from enrichment_router.tools.heuristic import (
    TLD_TO_COUNTRY,
    _match_country_from_domain,
    enrich,
)


def test_tld_spot_checks():
    expected = {
        "us": "United States",
        "uk": "United Kingdom",
        "jp": "Japan",
        "br": "Brazil",
        "ng": "Nigeria",
        "ae": "United Arab Emirates",
        "kr": "South Korea",
        "nz": "New Zealand",
        "ar": "Argentina",
        "eg": "Egypt",
        "pt": "Portugal",
        "ua": "Ukraine",
        "vn": "Vietnam",
        "sa": "Saudi Arabia",
        "ke": "Kenya",
        "cl": "Chile",
        "ie": "Ireland",
        "ro": "Romania",
        "bd": "Bangladesh",
        "kz": "Kazakhstan",
    }
    for tld, country in expected.items():
        assert TLD_TO_COUNTRY.get(tld) == country


def test_fake_tld_returns_none():
    assert _match_country_from_domain("example.xyz") is None
    assert _match_country_from_domain("example.fake") is None


def test_domain_country_resolution():
    request = EnrichmentRequest(name="Anything", domain="foo.de", fields_needed={"country"})
    result = enrich(request)
    assert result.resolved["country"].value == "Germany"
    assert result.resolved["country"].confidence == 0.35
    assert result.resolved["country"].tier == 0


def test_industry_keyword_technology():
    request = EnrichmentRequest(name="OpenAI Labs", fields_needed={"industry"})
    result = enrich(request)
    assert result.resolved["industry"].value == "Technology"


def test_industry_keyword_finance():
    request = EnrichmentRequest(name="Sequoia Capital", fields_needed={"industry"})
    result = enrich(request)
    assert result.resolved["industry"].value == "Finance"


def test_industry_keyword_healthcare():
    request = EnrichmentRequest(name="MedBio Health", fields_needed={"industry"})
    result = enrich(request)
    assert result.resolved["industry"].value == "Healthcare"


def test_industry_keyword_food_beverage():
    request = EnrichmentRequest(name="Sweet Bakery", fields_needed={"industry"})
    result = enrich(request)
    assert result.resolved["industry"].value == "Food & Beverage"


def test_industry_keyword_industrial_robotics():
    request = EnrichmentRequest(name="Boston Robotics", fields_needed={"industry"})
    result = enrich(request)
    assert result.resolved["industry"].value == "Industrial/Robotics"


def test_industry_case_insensitivity():
    request = EnrichmentRequest(name="OPENAI", fields_needed={"industry"})
    result = enrich(request)
    assert result.resolved["industry"].value == "Technology"


def test_no_industry_match():
    request = EnrichmentRequest(name="Acme", fields_needed={"industry"})
    result = enrich(request)
    assert result.resolved == {}


def test_tier0_never_resolves_is_public_or_short_description():
    request = EnrichmentRequest(
        name="Anything",
        fields_needed={"is_public", "short_description"},
    )
    result = enrich(request)
    assert result.resolved == {}


def test_empty_fields_needed():
    request = EnrichmentRequest(name="Anything", fields_needed=set())
    result = enrich(request)
    assert result.resolved == {}
    assert result.cost_usd == 0.0


def test_multiple_fields():
    request = EnrichmentRequest(
        name="OpenAI Labs",
        domain="foo.fr",
        fields_needed={"industry", "country"},
    )
    result = enrich(request)
    assert result.resolved["industry"].value == "Technology"
    assert result.resolved["country"].value == "France"
