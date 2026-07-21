from __future__ import annotations

import json
import sys
from pathlib import Path

from enrichment_router.models import (
    EnrichmentRequest,
    EnrichmentResult,
    RunStatus,
)
from enrichment_router.tools.llm import LLMClient, enrich as llm_enrich
from enrichment_router.validation import MissingNameError, validate_request

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def run_baseline(raw_request: dict, llm_client: LLMClient) -> EnrichmentResult:
    """Run the always-LLM baseline strategy.

    Skips tiers 0 and 1 entirely. Calls tier 2 (LLM) directly for every
    field in fields_needed. Does NOT enforce the budget — the baseline's
    purpose is to measure what "always use the LLM" costs in full.

    No trace events are recorded because the baseline is an offline
    evaluation tool, not the production router. Tracing would require a
    database connection and adds overhead irrelevant to cost measurement.
    """
    try:
        request = validate_request(raw_request)
    except MissingNameError:
        # Return an empty result rather than propagating the exception so
        # callers can batch-process datasets without individual failures
        # aborting the run.
        return EnrichmentResult(
            request=EnrichmentRequest(name=""),
            resolved={},
            status="partial_no_more_tiers",
            total_cost_usd=0.0,
            total_latency_ms=0.0,
            unresolved_fields=[],
        )

    if not request.fields_needed:
        return EnrichmentResult(
            request=request,
            resolved={},
            status="done_all_resolved",
            total_cost_usd=0.0,
            total_latency_ms=0.0,
            unresolved_fields=[],
        )

    llm_result = llm_enrich(request, llm_client)
    unresolved = sorted(request.fields_needed - set(llm_result.resolved.keys()))
    status: RunStatus = "done_all_resolved" if not unresolved else "partial_no_more_tiers"
    return EnrichmentResult(
        request=request,
        resolved=llm_result.resolved,
        status=status,
        total_cost_usd=llm_result.cost_usd,
        total_latency_ms=llm_result.measured_latency_ms,
        unresolved_fields=unresolved,
    )


def load_dataset() -> list[dict]:
    path = Path(__file__).resolve().parent / "companies.json"
    return json.loads(path.read_text())


if __name__ == "__main__":
    data = load_dataset()
    print(f"Loaded {len(data)} records from eval/companies.json")
    for i, rec in enumerate(data):
        try:
            req = validate_request(rec)
            print(f"  [{i}] {req.name!r} — fields_needed={sorted(req.fields_needed)}")
        except MissingNameError:
            print(f"  [{i}] INVALID (missing name)")
