from __future__ import annotations

from unittest.mock import MagicMock

from enrichment_router.tools.llm import FakeLLMClient, LLMResponse
from eval.baseline import run_baseline
from eval.run_eval import EvalReport, render_chart, run_eval


def stub_wiki_fetcher(name: str) -> None:
    """Stub fetcher that simulates a Wikipedia 404 for all names."""
    return None


def _build_client() -> FakeLLMClient:
    """Provides a FakeLLMClient with deterministic responses for
    every company in the 18-record eval dataset."""
    scripts: dict[str, LLMResponse] = {
        # Real companies (7) - return full, valid data
        "Stripe": LLMResponse(
            text='{"industry":"Financial Services","country":"United States","is_public":false,"short_description":"Payment processing API"}',
            prompt_tokens=50,
            completion_tokens=25,
        ),
        "Notion": LLMResponse(
            text='{"industry":"Software","country":"United States","is_public":false,"short_description":"Note-taking app"}',
            prompt_tokens=50,
            completion_tokens=20,
        ),
        "Airtable": LLMResponse(
            text='{"industry":"Software","country":"United States","is_public":false,"short_description":"Spreadsheet-database hybrid"}',
            prompt_tokens=50,
            completion_tokens=20,
        ),
        "Figma": LLMResponse(
            text='{"industry":"Software","country":"United States","is_public":true,"short_description":"Collaborative design tool"}',
            prompt_tokens=50,
            completion_tokens=20,
        ),
        "Linear": LLMResponse(
            text='{"industry":"Software","country":"United States","is_public":false,"short_description":"Issue tracking tool"}',
            prompt_tokens=50,
            completion_tokens=20,
        ),
        "GitHub": LLMResponse(
            text='{"industry":"Software","country":"United States","is_public":true,"short_description":"Code hosting platform"}',
            prompt_tokens=50,
            completion_tokens=20,
        ),
        "Vercel": LLMResponse(
            text='{"industry":"Software","country":"United States","is_public":false,"short_description":"Frontend cloud platform"}',
            prompt_tokens=50,
            completion_tokens=20,
        ),
        # Fictional companies (6) - LLM doesn't know them, returns nulls
        "Quantum Pie Holdings": LLMResponse(
            text='{"industry":null,"country":null,"is_public":null,"short_description":null}',
            prompt_tokens=50,
            completion_tokens=10,
        ),
        "Zxy Corp": LLMResponse(
            text='{"industry":null,"country":null,"is_public":null,"short_description":null}',
            prompt_tokens=50,
            completion_tokens=10,
        ),
        "Bogus Industries": LLMResponse(
            text='{"industry":null,"country":null,"is_public":null,"short_description":null}',
            prompt_tokens=50,
            completion_tokens=10,
        ),
        "FakeCo": LLMResponse(
            text='{"industry":null,"country":null,"is_public":null,"short_description":null}',
            prompt_tokens=50,
            completion_tokens=10,
        ),
        "Nonexistent LLC": LLMResponse(
            text='{"industry":null,"country":null,"is_public":null,"short_description":null}',
            prompt_tokens=50,
            completion_tokens=10,
        ),
        "Imaginary Systems": LLMResponse(
            text='{"industry":null,"country":null,"is_public":null,"short_description":null}',
            prompt_tokens=50,
            completion_tokens=10,
        ),
        # Keyword-bearing companies (4) - LLM confirms tier 0 guesses
        "OpenAI Labs": LLMResponse(
            text='{"industry":"Technology","country":"United States","is_public":false,"short_description":"AI research lab"}',
            prompt_tokens=50,
            completion_tokens=20,
        ),
        "Sequoia Capital": LLMResponse(
            text='{"industry":"Finance","country":"United States","is_public":false,"short_description":"Venture capital firm"}',
            prompt_tokens=50,
            completion_tokens=20,
        ),
        "Sweet Bakery": LLMResponse(
            text='{"industry":"Food & Beverage","country":"United States","is_public":false,"short_description":"Local bakery"}',
            prompt_tokens=50,
            completion_tokens=20,
        ),
        "Boston Robotics": LLMResponse(
            text='{"industry":"Industrial/Robotics","country":"Germany","is_public":false,"short_description":"Robotics manufacturer"}',
            prompt_tokens=50,
            completion_tokens=20,
        ),
        # All-fields-supplied (1) - LLM won't be called due to empty fields_needed,
        # but we provide a script just in case.
        "Pre-Filled Corp": LLMResponse(
            text='{"industry":"Technology","country":"United States","is_public":true,"short_description":"Pre-filled company"}',
            prompt_tokens=50,
            completion_tokens=20,
        ),
    }
    return FakeLLMClient(scripts)


def test_run_eval_returns_eval_report():
    client = _build_client()
    report = run_eval(client, stub_wiki_fetcher)
    assert isinstance(report, EvalReport)
    assert report.router.records_processed == 18
    assert report.baseline.records_processed == 18


def test_router_cost_lte_baseline_cost():
    client = _build_client()
    report = run_eval(client, stub_wiki_fetcher)
    assert report.router.total_cost_usd <= report.baseline.total_cost_usd


def test_cost_savings_pct_ge_zero():
    client = _build_client()
    report = run_eval(client, stub_wiki_fetcher)
    assert report.cost_savings_pct >= 0.0


def test_completion_rates_between_0_and_1():
    client = _build_client()
    report = run_eval(client, stub_wiki_fetcher)
    assert 0.0 <= report.router.completion_rate <= 1.0
    assert 0.0 <= report.baseline.completion_rate <= 1.0


def test_render_chart_writes_non_empty_png(tmp_path):
    fake_report = EvalReport(
        router=MagicMock(
            total_cost_usd=0.01,
            total_latency_ms=500.0,
            total_fields_needed=0,
            total_fields_resolved=0,
            completion_rate=1.0,
            records_processed=18,
            records_fully_resolved=18,
            name="router",
        ),
        baseline=MagicMock(
            total_cost_usd=0.05,
            total_latency_ms=3000.0,
            total_fields_needed=0,
            total_fields_resolved=0,
            completion_rate=1.0,
            records_processed=18,
            records_fully_resolved=18,
            name="baseline",
        ),
        cost_savings_pct=80.0,
        latency_savings_pct=83.3,
    )
    path = tmp_path / "test_chart.png"
    render_chart(fake_report, path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_all_fields_supplied_zero_cost():
    client = _build_client()
    raw = {
        "name": "Pre-Filled Corp",
        "industry": "Fintech",
        "country": "US",
        "is_public": False,
        "short_description": "All set.",
        "max_cost_usd": 0.05,
        "max_latency_ms": 10000,
    }
    # Router
    from enrichment_router.budget import Budget
    from enrichment_router.graph import run_enrichment
    from enrichment_router.validation import validate_request

    req = validate_request(raw)
    budget = Budget(max_cost_usd=0.05, max_latency_ms=10000.0)
    result, _trace = run_enrichment(
        request=req,
        budget=budget,
        llm_client=client,
        wiki_fetcher=stub_wiki_fetcher,
    )
    assert result.total_cost_usd == 0.0
    # Baseline
    baseline = run_baseline(raw, client)
    assert baseline.total_cost_usd == 0.0


def test_keyword_bearing_record_resolves_industry():
    client = _build_client()
    raw = {
        "name": "OpenAI Labs",
        "domain": "openai.com",
        "max_cost_usd": 0.05,
        "max_latency_ms": 10000,
    }

    from enrichment_router.budget import Budget
    from enrichment_router.graph import run_enrichment
    from enrichment_router.validation import validate_request

    req = validate_request(raw)
    budget = Budget(max_cost_usd=0.05, max_latency_ms=10000.0)
    result, _trace = run_enrichment(
        request=req,
        budget=budget,
        llm_client=client,
        wiki_fetcher=stub_wiki_fetcher,
    )
    assert "industry" in result.resolved
