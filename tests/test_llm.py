from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from enrichment_router.models import EnrichmentRequest
from enrichment_router.pricing import modeled_cost_usd
from enrichment_router.tools.llm import (
    FakeLLMClient,
    GroqLLMClient,
    LLMProviderConfig,
    LLMResponse,
    enrich,
)


def _make_response(text: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> LLMResponse:
    return LLMResponse(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def test_modeled_cost_usd():
    assert modeled_cost_usd(1_000_000, 0) == 1.0
    assert modeled_cost_usd(0, 1_000_000) == 5.0
    assert modeled_cost_usd(500, 100) == pytest.approx(0.001, rel=0.1)


def test_fake_llm_client_matches_by_substring():
    client = FakeLLMClient({"Stripe": _make_response("ok")})
    response = client.complete("sys", "Tell me about Stripe the company")
    assert response.text == "ok"


def test_fake_llm_client_raises_on_no_match():
    client = FakeLLMClient({"Stripe": _make_response("ok")})
    with pytest.raises(KeyError):
        client.complete("sys", "Tell me about Acme")


def test_enrich_parses_plain_json():
    request = EnrichmentRequest(name="Acme", fields_needed={"industry"})
    client = FakeLLMClient({"Acme": _make_response('{"industry": "Technology"}')})
    result = enrich(request, client)
    assert result.resolved["industry"].value == "Technology"
    assert result.resolved["industry"].tier == 2
    assert result.resolved["industry"].confidence == 0.9


def test_enrich_strips_markdown_fences():
    request = EnrichmentRequest(name="Acme", fields_needed={"industry"})
    response_text = '```json\n{"industry": "Technology"}\n```'
    client = FakeLLMClient({"Acme": _make_response(response_text)})
    result = enrich(request, client)
    assert result.resolved["industry"].value == "Technology"


def test_enrich_excludes_null_fields():
    request = EnrichmentRequest(name="Acme", fields_needed={"industry", "country"})
    response_text = '{"industry": "Technology", "country": null}'
    client = FakeLLMClient({"Acme": _make_response(response_text)})
    result = enrich(request, client)
    assert set(result.resolved.keys()) == {"industry"}


def test_enrich_tolerates_malformed_json():
    request = EnrichmentRequest(name="Acme", fields_needed={"industry"})
    client = FakeLLMClient(
        {"Acme": _make_response("not json", prompt_tokens=10, completion_tokens=5)}
    )
    result = enrich(request, client)
    assert result.resolved == {}


def test_modeled_cost_from_token_counts():
    request = EnrichmentRequest(name="Acme", fields_needed={"industry"})
    client = FakeLLMClient(
        {
            "Acme": _make_response(
                '{"industry": "Tech"}', prompt_tokens=1000, completion_tokens=200
            ),
        }
    )
    result = enrich(request, client)
    assert result.prompt_tokens == 1000
    assert result.completion_tokens == 200
    assert result.cost_usd == modeled_cost_usd(1000, 200)


def test_fields_needed_subset_respected():
    request = EnrichmentRequest(name="Acme", fields_needed={"industry"})
    response_text = '{"industry": "Technology", "country": "US"}'
    client = FakeLLMClient({"Acme": _make_response(response_text)})
    result = enrich(request, client)
    assert set(result.resolved.keys()) == {"industry"}


def test_empty_fields_needed():
    request = EnrichmentRequest(name="Acme", fields_needed=set())

    class NeverCalledClient:
        def complete(self, system: str, prompt: str) -> LLMResponse:
            raise AssertionError("complete should not be called for empty fields_needed")

    result = enrich(request, NeverCalledClient())
    assert result.resolved == {}
    assert result.cost_usd == 0.0


def test_llm_provider_config_is_frozen():
    config = LLMProviderConfig(base_url="https://x", api_key_env="K", model="m")
    with pytest.raises(FrozenInstanceError):
        config.base_url = "https://y"  # type: ignore[misc]


def test_groq_llm_client_raises_on_missing_env_var(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqLLMClient()
