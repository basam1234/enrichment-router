from __future__ import annotations

from unittest.mock import patch


from enrichment_router.tools.llm import (
    FakeLLMClient,
    LLMResponse,
)
from enrichment_router.pricing import modeled_cost_usd
from eval.baseline import load_dataset, run_baseline


def test_baseline_llm_for_all_needed_fields():
    prompt_tokens = 120
    completion_tokens = 60
    expected_cost = modeled_cost_usd(prompt_tokens, completion_tokens)
    client = FakeLLMClient(
        {
            "TestCo": LLMResponse(
                text='{"industry":"Technology","country":"US","is_public":true,"short_description":"A test company."}',
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
        }
    )
    raw = {"name": "TestCo", "max_cost_usd": 0.05, "max_latency_ms": 10000}
    result = run_baseline(raw, client)
    assert result.status == "done_all_resolved"
    assert len(result.resolved) == 4
    assert result.total_cost_usd == expected_cost
    assert result.unresolved_fields == []


def test_baseline_partial_when_llm_returns_nulls():
    client = FakeLLMClient(
        {
            "TestCo": LLMResponse(
                text='{"industry":"Technology","country":null,"is_public":null,"short_description":null}',
                prompt_tokens=100,
                completion_tokens=50,
            ),
        }
    )
    raw = {"name": "TestCo", "max_cost_usd": 0.05, "max_latency_ms": 10000}
    result = run_baseline(raw, client)
    assert result.status == "partial_no_more_tiers"
    assert sorted(result.unresolved_fields) == ["country", "is_public", "short_description"]


def test_baseline_skips_llm_when_fields_needed_empty():
    client = FakeLLMClient({})
    raw = {
        "name": "AllSet Corp",
        "industry": "Fintech",
        "country": "US",
        "is_public": False,
        "short_description": "All provided.",
        "max_cost_usd": 0.05,
        "max_latency_ms": 10000,
    }
    result = run_baseline(raw, client)
    assert result.status == "done_all_resolved"
    assert result.total_cost_usd == 0.0
    assert result.resolved == {}


def test_baseline_does_not_call_heuristic_or_wikipedia():
    with (
        patch("enrichment_router.tools.heuristic.enrich") as mock_heuristic,
        patch("enrichment_router.tools.wikipedia.enrich") as mock_wiki,
    ):
        client = FakeLLMClient(
            {
                "TestCo": LLMResponse(
                    text='{"industry":"Technology","country":"US","is_public":true,"short_description":"desc"}',
                    prompt_tokens=100,
                    completion_tokens=50,
                ),
            }
        )
        raw = {"name": "TestCo", "max_cost_usd": 0.05, "max_latency_ms": 10000}
        run_baseline(raw, client)
        mock_heuristic.assert_not_called()
        mock_wiki.assert_not_called()


def test_baseline_handles_missing_name():
    client = FakeLLMClient({})
    result = run_baseline({}, client)
    assert result.status == "partial_no_more_tiers"
    assert result.total_cost_usd == 0.0


def test_load_dataset_returns_18():
    assert len(load_dataset()) == 18
