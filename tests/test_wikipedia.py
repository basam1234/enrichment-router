from __future__ import annotations

from enrichment_router.models import EnrichmentRequest
from enrichment_router.tools.wikipedia import enrich

STRIPE_FIXTURE: dict = {
    "description": "American financial services company",
    "extract": (
        "Stripe, Inc. is an American financial services and software "
        "as a service (SaaS) company dual-headquartered in South San "
        "Francisco, California, United States and Dublin, Ireland. "
        "It is privately held."
    ),
}

SINGLE_COUNTRY_FIXTURE: dict = {
    "description": "A German company",
    "extract": "Acme is based in Germany. It was founded in 2010.",
}

NESTED_COUNTRY_FIXTURE: dict = {
    "description": "An African company",
    "extract": "A company based in Equatorial Guinea.",
}

AMB_COUNTRY_FIXTURE: dict = {
    "description": "A global company",
    "extract": "The company operates in the United States and Germany.",
}

ZERO_COUNTRY_FIXTURE: dict = {
    "description": "A small company",
    "extract": "A small startup based somewhere.",
}

PUBLIC_FIXTURE: dict = {
    "description": "A public company",
    "extract": "The company is publicly traded on the NYSE.",
}


def NOT_FOUND_FETCHER(name: str) -> None:
    """Fake fetcher simulating a 404 / not-found response, for tests."""
    return None


def _make_fetcher(data: dict):
    return lambda name: data


def test_short_description():
    request = EnrichmentRequest(
        name="Stripe",
        fields_needed={"short_description"},
    )
    result = enrich(request, fetcher=_make_fetcher(STRIPE_FIXTURE))
    assert result.resolved["short_description"].value == "American financial services company"
    assert result.resolved["short_description"].confidence == 0.85


def test_country_single_match():
    request = EnrichmentRequest(
        name="Acme",
        fields_needed={"country"},
    )
    result = enrich(request, fetcher=_make_fetcher(SINGLE_COUNTRY_FIXTURE))
    assert result.resolved["country"].value == "Germany"
    assert result.resolved["country"].confidence == 0.55


def test_nested_country_name():
    request = EnrichmentRequest(
        name="Acme",
        fields_needed={"country"},
    )
    result = enrich(request, fetcher=_make_fetcher(NESTED_COUNTRY_FIXTURE))
    assert result.resolved["country"].value == "Equatorial Guinea"


def test_ambiguous_country():
    request = EnrichmentRequest(
        name="Acme",
        fields_needed={"country"},
    )
    result = enrich(request, fetcher=_make_fetcher(AMB_COUNTRY_FIXTURE))
    assert "country" not in result.resolved


def test_zero_country_matches():
    request = EnrichmentRequest(
        name="Acme",
        fields_needed={"country"},
    )
    result = enrich(request, fetcher=_make_fetcher(ZERO_COUNTRY_FIXTURE))
    assert "country" not in result.resolved


def test_public_company_true():
    request = EnrichmentRequest(
        name="Acme",
        fields_needed={"is_public"},
    )
    result = enrich(request, fetcher=_make_fetcher(PUBLIC_FIXTURE))
    assert result.resolved["is_public"].value is True
    assert result.resolved["is_public"].confidence == 0.5


def test_private_company_false():
    request = EnrichmentRequest(
        name="Stripe",
        fields_needed={"is_public"},
    )
    result = enrich(request, fetcher=_make_fetcher(STRIPE_FIXTURE))
    assert result.resolved["is_public"].value is False


def test_404_clean_not_found():
    request = EnrichmentRequest(
        name="NoSuchPage",
        fields_needed={"short_description"},
    )
    result = enrich(request, fetcher=NOT_FOUND_FETCHER)
    assert result.not_found is True
    assert result.resolved == {}


def test_industry_never_resolved():
    request = EnrichmentRequest(
        name="Stripe",
        fields_needed={"industry"},
    )
    result = enrich(request, fetcher=_make_fetcher(STRIPE_FIXTURE))
    assert "industry" not in result.resolved


def test_fields_needed_subset_respected():
    request = EnrichmentRequest(
        name="Stripe",
        fields_needed={"short_description"},
    )
    result = enrich(request, fetcher=_make_fetcher(STRIPE_FIXTURE))
    assert set(result.resolved.keys()) == {"short_description"}


def test_measured_latency_ms_non_negative():
    request = EnrichmentRequest(
        name="Stripe",
        fields_needed={"short_description"},
    )
    result = enrich(request, fetcher=_make_fetcher(STRIPE_FIXTURE))
    assert result.measured_latency_ms >= 0.0
