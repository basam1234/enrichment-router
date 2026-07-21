from __future__ import annotations

from operator import add
from typing import Annotated, Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .budget import Budget, BudgetTracker
from .models import (
    EnrichmentRequest,
    EnrichmentResult,
    FieldName,
    ResolvedField,
    RunStatus,
)
from .policy import PolicyDecision, decide, ACCEPTANCE_THRESHOLD
from .tools import heuristic, llm as llm_mod, wikipedia
from .tools.llm import LLMClient
from .trace import TraceEvent, make_event

TIER_DECLARED: dict[int, tuple[float, float]] = {
    0: (heuristic.TIER0_DECLARED_COST_USD, heuristic.TIER0_DECLARED_LATENCY_MS),
    1: (wikipedia.TIER1_DECLARED_COST_USD, wikipedia.TIER1_DECLARED_LATENCY_MS),
    2: (llm_mod.TIER2_DECLARED_COST_USD, llm_mod.TIER2_DECLARED_LATENCY_MS),
}
MAX_TIER: int = 2


class GraphState(TypedDict):
    request: EnrichmentRequest
    resolved: dict[FieldName, ResolvedField]
    current_tier: int
    budget_tracker: BudgetTracker
    trace: Annotated[list[TraceEvent], add]
    status: Optional[RunStatus]
    unresolved_fields: list[FieldName]
    last_decision: Optional[str]
    llm_client: LLMClient
    wiki_fetcher: Optional[Callable[[str], Optional[dict]]]


def _run_tier_tool(
    state: GraphState,
    tier: int,
) -> tuple[dict[FieldName, ResolvedField], float, float]:
    if tier == 0:
        result = heuristic.enrich(state["request"])
        return result.resolved, result.cost_usd, result.measured_latency_ms
    if tier == 1:
        result = wikipedia.enrich(state["request"], fetcher=state["wiki_fetcher"])
        return result.resolved, result.cost_usd, result.measured_latency_ms
    if tier == 2:
        # Calculate which fields are STILL unresolved or below the acceptance threshold.
        # We must ask the LLM to re-evaluate any field that a lower tier
        # guessed at but didn't have enough confidence for.
        currently_unresolved = {
            f
            for f in state["request"].fields_needed
            if f not in state["resolved"] or state["resolved"][f].confidence < ACCEPTANCE_THRESHOLD
        }

        # Create a copy of the request with ONLY the currently unresolved fields
        llm_request = state["request"].model_copy(
            update={
                "fields_needed": currently_unresolved,
            }
        )

        # If there is nothing left to resolve, return empty
        if not currently_unresolved:
            return {}, 0.0, 0.0

        result = llm_mod.enrich(llm_request, client=state["llm_client"])
        return result.resolved, result.cost_usd, result.measured_latency_ms
    raise ValueError(f"Unknown tier {tier}")


def try_tier(state: GraphState) -> dict:
    tier = state["current_tier"]
    declared_cost, declared_latency = TIER_DECLARED[tier]
    events: list[TraceEvent] = [make_event("try_tier", tier=tier, action="invoke")]

    resolved_now, cost, latency = _run_tier_tool(state, tier)

    new_resolved = dict(state["resolved"])
    new_resolved.update(resolved_now)

    drift_event = state["budget_tracker"].record_actual(
        actual_cost_usd=cost,
        actual_latency_ms=latency,
        estimated_latency_ms=declared_latency,
    )
    if drift_event is not None:
        events.append(drift_event)

    events.append(
        make_event(
            "try_tier",
            tier=tier,
            newly_resolved=sorted(resolved_now.keys()),
            cost_usd=cost,
            latency_ms=latency,
        )
    )
    return {"resolved": new_resolved, "trace": events}


def check_sufficiency(state: GraphState) -> dict:
    """Pure policy check. Sets last_decision so route_after_check can
    dispatch without re-deriving. Also advances current_tier when
    escalating.

    last_decision is stored in state because route_after_check runs
    AFTER this node's updates are merged — if it re-derived from
    current_tier, the already-incremented value would produce the wrong
    next_tier calculation.
    """
    current_tier = state["current_tier"]
    next_tier = current_tier + 1
    has_headroom = True
    if next_tier <= MAX_TIER:
        nc, nl = TIER_DECLARED[next_tier]
        has_headroom = state["budget_tracker"].has_headroom_for(nc, nl)

    outcome = decide(
        fields_needed=state["request"].fields_needed,
        resolved=state["resolved"],
        current_tier=current_tier,
        has_budget_headroom_for_next_tier=has_headroom,
    )

    events = [
        make_event(
            "check_sufficiency",
            decision=outcome.decision.value,
            unresolved_fields=outcome.unresolved_fields,
            reason=outcome.reason,
        )
    ]

    update: dict = {
        "trace": events,
        "unresolved_fields": outcome.unresolved_fields,
        "last_decision": outcome.decision.value,
    }
    if outcome.decision == PolicyDecision.ESCALATE_TO_NEXT_TIER:
        update["current_tier"] = outcome.next_tier
    return update


def route_after_check(state: GraphState) -> str:
    """Reads last_decision directly — does NOT re-derive via decide()."""
    decision = state["last_decision"]
    return {
        PolicyDecision.ESCALATE_TO_NEXT_TIER.value: "try_tier",
        PolicyDecision.DONE_ALL_RESOLVED.value: "finalize_done",
        PolicyDecision.STOP_BUDGET_EXHAUSTED.value: "finalize_partial_budget",
        PolicyDecision.STOP_NO_MORE_TIERS.value: "finalize_partial_no_more_tiers",
    }[decision]


def finalize_done(state: GraphState) -> dict:
    return {
        "status": "done_all_resolved",
        "trace": [make_event("finalize_done", status="done_all_resolved")],
    }


def finalize_partial_budget(state: GraphState) -> dict:
    return {
        "status": "partial_budget",
        "trace": [
            make_event(
                "finalize_partial_budget",
                status="partial_budget",
                unresolved_fields=state.get("unresolved_fields", []),
            )
        ],
    }


def finalize_partial_no_more_tiers(state: GraphState) -> dict:
    return {
        "status": "partial_no_more_tiers",
        "trace": [
            make_event(
                "finalize_partial_no_more_tiers",
                status="partial_no_more_tiers",
                unresolved_fields=state.get("unresolved_fields", []),
            )
        ],
    }


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("try_tier", try_tier)
    g.add_node("check_sufficiency", check_sufficiency)
    g.add_node("finalize_done", finalize_done)
    g.add_node("finalize_partial_budget", finalize_partial_budget)
    g.add_node("finalize_partial_no_more_tiers", finalize_partial_no_more_tiers)
    g.set_entry_point("try_tier")
    g.add_edge("try_tier", "check_sufficiency")
    g.add_conditional_edges(
        "check_sufficiency",
        route_after_check,
        {
            "try_tier": "try_tier",
            "finalize_done": "finalize_done",
            "finalize_partial_budget": "finalize_partial_budget",
            "finalize_partial_no_more_tiers": "finalize_partial_no_more_tiers",
        },
    )
    g.add_edge("finalize_done", END)
    g.add_edge("finalize_partial_budget", END)
    g.add_edge("finalize_partial_no_more_tiers", END)
    return g.compile()


def _caller_supplied_resolved(
    request: EnrichmentRequest,
) -> dict[FieldName, ResolvedField]:
    """Build ResolvedField entries for caller-supplied fields.

    Tagged with caller_supplied=True so the frontend displays
    "Caller-supplied" instead of a tier label. Confidence 1.0; tier=0
    is a placeholder never read when caller_supplied=True.
    """
    return {
        fname: ResolvedField(
            name=fname,
            value=value,
            tier=0,
            confidence=1.0,
            caller_supplied=True,
        )
        for fname, value in request.known_fields.items()
    }


def run_enrichment(
    request: EnrichmentRequest,
    budget: Budget,
    llm_client: LLMClient,
    wiki_fetcher: Optional[Callable[[str], Optional[dict]]] = None,
) -> tuple[EnrichmentResult, list[TraceEvent]]:
    """Run the full enrichment graph. Returns (result, trace_events).

    Caller-supplied known_fields are NOT seeded into the graph's
    resolved dict — the policy only checks fields_needed, which
    excludes known_fields by construction. After the graph completes,
    known_fields are merged into the result for display.

    The graph uses a cycle (try_tier → check_sufficiency → try_tier)
    rather than a linear chain so that each tier decision can be made
    based on the latest resolved state and budget. last_decision is
    stored in GraphState instead of being re-derived in
    route_after_check because the conditional router runs after state
    updates are merged — re-deriving from the already-incremented
    current_tier would produce the wrong dispatch target.

    When multiple tiers resolve the same field (e.g., tier 0 industry
    at 0.35, then tier 2 industry at 0.9), the last tier to write wins
    because resolved is a plain dict updated via ``.update()`` on each
    cycle. This is intentional: higher tiers have higher confidence.

    The trace reducer uses ``operator.add`` (list concatenation), so
    each node appends events without stomping previous entries.

    Empty fields_needed is handled as a fast-path short-circuit — no
    graph is built or invoked, and the returned trace is ``[]``.
    """
    if not request.fields_needed:
        resolved = _caller_supplied_resolved(request)
        result = EnrichmentResult(
            request=request,
            resolved=resolved,
            status="done_all_resolved",
            total_cost_usd=0.0,
            total_latency_ms=0.0,
            unresolved_fields=[],
        )
        return result, []

    compiled = build_graph()
    initial_state: GraphState = {
        "request": request,
        "resolved": {},
        "current_tier": 0,
        "budget_tracker": BudgetTracker(budget=budget),
        "trace": [],
        "status": None,
        "unresolved_fields": [],
        "last_decision": None,
        "llm_client": llm_client,
        "wiki_fetcher": wiki_fetcher,
    }

    final_state = compiled.invoke(initial_state, config={"recursion_limit": 20})

    final_resolved = dict(final_state["resolved"])
    final_resolved.update(_caller_supplied_resolved(request))

    result = EnrichmentResult(
        request=request,
        resolved=final_resolved,
        status=final_state["status"],
        total_cost_usd=final_state["budget_tracker"].spent_cost_usd,
        total_latency_ms=final_state["budget_tracker"].spent_latency_ms,
        unresolved_fields=final_state.get("unresolved_fields", []),
    )
    return result, list(final_state["trace"])
