from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from enrichment_router.budget import Budget
from enrichment_router.graph import run_enrichment
from enrichment_router.models import EnrichmentRequest
from enrichment_router.pricing import modeled_cost_usd
from enrichment_router.tools import heuristic, llm as llm_mod, wikipedia
from enrichment_router.tools.llm import FakeLLMClient, LLMResponse


@contextmanager
def spy_tools():
    with (
        patch.object(heuristic, "enrich", wraps=heuristic.enrich) as h,
        patch.object(wikipedia, "enrich", wraps=wikipedia.enrich) as w,
        patch.object(llm_mod, "enrich", wraps=llm_mod.enrich) as llm,
    ):
        yield h, w, llm


def stub_wiki_404(name: str) -> None:
    """Stub fetcher that simulates a Wikipedia 404 for any name."""
    return None


def test_empty_fields_needed_zero_tool_calls():
    request = EnrichmentRequest(name="Acme", fields_needed=set())
    with spy_tools() as (h_spy, w_spy, l_spy):
        result, trace = run_enrichment(
            request,
            budget=Budget(max_cost_usd=0.01, max_latency_ms=5000.0),
            llm_client=FakeLLMClient({}),
        )
    assert h_spy.call_count == 0
    assert w_spy.call_count == 0
    assert l_spy.call_count == 0
    assert result.status == "done_all_resolved"
    assert result.total_cost_usd == 0.0
    assert result.total_latency_ms == 0.0
    assert trace == []


def test_budget_exhausted_after_tier0():
    request = EnrichmentRequest(name="Acme", fields_needed={"industry"})
    with spy_tools() as (h_spy, w_spy, l_spy):
        result, _ = run_enrichment(
            request,
            budget=Budget(max_cost_usd=0.0, max_latency_ms=10.0),
            llm_client=FakeLLMClient({}),
        )
    assert w_spy.call_count == 0
    assert l_spy.call_count == 0
    assert result.status == "partial_budget"
    assert result.unresolved_fields == ["industry"]


def test_full_escalation_0_1_2():

    llm_response = LLMResponse(
        text='{"industry":"Tech","country":"US","is_public":true,"short_description":"A company"}',
        prompt_tokens=500,
        completion_tokens=100,
    )
    request = EnrichmentRequest(
        name="Acme",
        fields_needed={"industry", "country", "is_public", "short_description"},
    )
    with spy_tools() as (h_spy, w_spy, l_spy):
        result, _ = run_enrichment(
            request,
            budget=Budget(max_cost_usd=0.1, max_latency_ms=10000.0),
            llm_client=FakeLLMClient({"Acme": llm_response}),
            wiki_fetcher=stub_wiki_404,
        )
    assert h_spy.call_count == 1
    assert w_spy.call_count == 1
    assert l_spy.call_count == 1
    assert result.status == "done_all_resolved"
    for field_name in ("industry", "country", "is_public", "short_description"):
        assert field_name in result.resolved
        assert result.resolved[field_name].tier == 2
        assert result.resolved[field_name].confidence == 0.9
    assert result.total_cost_usd == modeled_cost_usd(500, 100)


def test_tier1_fully_resolves_no_tier2_call():
    wiki_fixture = {
        "description": "A payment company",
        "extract": "Stripe is a payment company.",
    }
    request = EnrichmentRequest(name="Acme", fields_needed={"short_description"})
    with spy_tools() as (h_spy, w_spy, l_spy):
        result, _ = run_enrichment(
            request,
            budget=Budget(max_cost_usd=0.1, max_latency_ms=10000.0),
            llm_client=FakeLLMClient({}),
            wiki_fetcher=lambda name: wiki_fixture,
        )
    assert result.status == "done_all_resolved"
    assert result.resolved["short_description"].tier == 1
    assert result.resolved["short_description"].confidence == 0.85
    assert h_spy.call_count == 1
    assert w_spy.call_count == 1
    assert l_spy.call_count == 0


def test_stop_at_tier2_with_unresolved():
    llm_response = LLMResponse(
        text='{"industry":null,"country":"US","is_public":null,"short_description":"A co."}',
        prompt_tokens=400,
        completion_tokens=80,
    )
    request = EnrichmentRequest(
        name="Acme",
        fields_needed={"industry", "country", "is_public", "short_description"},
    )
    result, _ = run_enrichment(
        request,
        budget=Budget(max_cost_usd=0.1, max_latency_ms=10000.0),
        llm_client=FakeLLMClient({"Acme": llm_response}),
        wiki_fetcher=stub_wiki_404,
    )
    assert result.status == "partial_no_more_tiers"
    assert set(result.resolved.keys()) & {"country", "short_description"} == {
        "country",
        "short_description",
    }
    assert sorted(result.unresolved_fields) == ["industry", "is_public"]


def test_budget_exhaustion_mid_escalation():
    wiki_fixture = {
        "description": "A payment company",
        "extract": "Stripe is a payment company.",
    }
    request = EnrichmentRequest(
        name="Acme",
        fields_needed={"industry", "short_description"},
    )
    with spy_tools() as (h_spy, w_spy, l_spy):
        result, _ = run_enrichment(
            request,
            budget=Budget(max_cost_usd=0.0, max_latency_ms=700.0),
            llm_client=FakeLLMClient({}),
            wiki_fetcher=lambda name: wiki_fixture,
        )
    assert result.status == "partial_budget"
    assert result.resolved.get("short_description") is not None
    assert result.resolved["short_description"].tier == 1
    assert result.unresolved_fields == ["industry"]
    assert l_spy.call_count == 0


def test_trace_is_append_only_across_cycle():
    llm_response = LLMResponse(
        text='{"industry":"Tech","country":"US","is_public":true,"short_description":"A co."}',
        prompt_tokens=500,
        completion_tokens=100,
    )
    request = EnrichmentRequest(
        name="Acme",
        fields_needed={"industry", "country", "is_public", "short_description"},
    )
    _, trace = run_enrichment(
        request,
        budget=Budget(max_cost_usd=0.1, max_latency_ms=10000.0),
        llm_client=FakeLLMClient({"Acme": llm_response}),
        wiki_fetcher=stub_wiki_404,
    )
    assert len(trace) >= 7
