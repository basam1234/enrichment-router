from enrichment_router.models import EnrichmentRequest
from enrichment_router.tools.wikipedia import enrich

STRIPE_FIXTURE = {
    "description": "American financial services company",
    "extract": "Stripe, Inc. is an American financial services and software as a service (SaaS) company dual-headquartered in South San Francisco, California, United States and Dublin, Ireland. It is privately held.",
}

SINGLE_COUNTRY_FIXTURE = {
    "description": "A German company",
    "extract": "Acme is based in Germany. It was founded in 2010.",
}

NESTED_COUNTRY_FIXTURE = {
    "description": "An African company",
    "extract": "A company based in Equatorial Guinea.",
}

AMB_COUNTRY_FIXTURE = {
    "description": "An international company",
    "extract": "A company based in both the United States and Germany.",
}

ZERO_COUNTRY_FIXTURE = {
    "description": "A small startup company",
    "extract": "A small startup company based somewhere.",
}

PUBLIC_FIXTURE = {
    "description": "A public company",
    "extract": "A public company publicly traded on the NYSE.",
}

DISAMBIGUATION_FIXTURE = {
    "description": "Topics referred to by the same term",
    "extract": "Meta may refer to different topics.",
}

NON_COMPANY_FIXTURE = {
    "description": "Vocalization used by cats",
    "extract": "A meow is a vocalization used by cats.",
}


def NOT_FOUND_FETCHER(name: str) -> None:
    return None


def test_short_description():
    req = EnrichmentRequest(name="Stripe", fields_needed={"short_description"})
    res = enrich(req, fetcher=lambda n: STRIPE_FIXTURE)
    assert "short_description" in res.resolved
    assert res.resolved["short_description"].value == "American financial services company"
    assert res.resolved["short_description"].confidence == 0.85


def test_country_single_match():
    req = EnrichmentRequest(name="Acme", fields_needed={"country"})
    res = enrich(req, fetcher=lambda n: SINGLE_COUNTRY_FIXTURE)
    assert "country" in res.resolved
    assert res.resolved["country"].value == "Germany"
    assert res.resolved["country"].confidence == 0.65


def test_nested_country_name():
    req = EnrichmentRequest(name="Acme", fields_needed={"country"})
    res = enrich(req, fetcher=lambda n: NESTED_COUNTRY_FIXTURE)
    assert "country" in res.resolved
    assert res.resolved["country"].value == "Equatorial Guinea"


def test_ambiguous_country():
    req = EnrichmentRequest(name="Acme", fields_needed={"country"})
    res = enrich(req, fetcher=lambda n: AMB_COUNTRY_FIXTURE)
    assert "country" not in res.resolved


def test_zero_country_matches():
    req = EnrichmentRequest(name="Acme", fields_needed={"country"})
    res = enrich(req, fetcher=lambda n: ZERO_COUNTRY_FIXTURE)
    assert "country" not in res.resolved


def test_public_company_true():
    req = EnrichmentRequest(name="Acme", fields_needed={"is_public"})
    res = enrich(req, fetcher=lambda n: PUBLIC_FIXTURE)
    assert "is_public" in res.resolved
    assert res.resolved["is_public"].value
    assert res.resolved["is_public"].confidence == 0.65


def test_private_company_false():
    req = EnrichmentRequest(name="Stripe", fields_needed={"is_public"})
    res = enrich(req, fetcher=lambda n: STRIPE_FIXTURE)
    assert "is_public" in res.resolved
    assert not res.resolved["is_public"].value


def test_404_clean_not_found():
    req = EnrichmentRequest(name="Nonexistent", fields_needed={"industry"})
    res = enrich(req, fetcher=NOT_FOUND_FETCHER)
    assert res.not_found
    assert res.resolved == {}


def test_industry_extracted_from_description():
    req = EnrichmentRequest(name="Stripe", fields_needed={"industry"})
    # "financial services" is in the STRIPE_FIXTURE description
    res = enrich(req, fetcher=lambda n: STRIPE_FIXTURE)
    assert "industry" in res.resolved
    assert res.resolved["industry"].value == "Finance"
    assert res.resolved["industry"].confidence == 0.65


def test_industry_not_extracted_when_no_keyword():
    req = EnrichmentRequest(name="Acme", fields_needed={"industry"})
    # SINGLE_COUNTRY_FIXTURE has no industry keyword
    res = enrich(req, fetcher=lambda n: SINGLE_COUNTRY_FIXTURE)
    assert "industry" not in res.resolved


def test_fields_needed_subset_respected():
    req = EnrichmentRequest(name="Stripe", fields_needed={"short_description"})
    res = enrich(req, fetcher=lambda n: STRIPE_FIXTURE)
    assert len(res.resolved) == 1
    assert "short_description" in res.resolved


def test_measured_latency_ms_non_negative():
    req = EnrichmentRequest(name="Stripe", fields_needed={"short_description"})
    res = enrich(req, fetcher=lambda n: STRIPE_FIXTURE)
    assert res.measured_latency_ms >= 0.0


def test_disambiguation_page_treated_as_not_found():
    req = EnrichmentRequest(name="Meta", fields_needed={"short_description"})
    res = enrich(req, fetcher=lambda n: DISAMBIGUATION_FIXTURE)
    assert res.not_found
    assert res.resolved == {}


def test_non_company_page_treated_as_not_found():
    req = EnrichmentRequest(name="Meow", fields_needed={"short_description"})
    res = enrich(req, fetcher=lambda n: NON_COMPANY_FIXTURE)
    assert res.not_found
    assert res.resolved == {}
