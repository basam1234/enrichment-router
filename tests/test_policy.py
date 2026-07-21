from __future__ import annotations

from enrichment_router.models import ResolvedField
from enrichment_router.policy import (
    ACCEPTANCE_THRESHOLD,
    PolicyDecision,
    decide,
)


def _rf(name, value, confidence, tier=1):
    return ResolvedField(name=name, value=value, tier=tier, confidence=confidence)


def test_done_all_resolved():
    resolved = {
        "industry": _rf("industry", "Tech", 0.9),
        "country": _rf("country", "US", 0.9),
    }
    outcome = decide(
        {"industry", "country"}, resolved, current_tier=1, has_budget_headroom_for_next_tier=True
    )
    assert outcome.decision == PolicyDecision.DONE_ALL_RESOLVED
    assert outcome.unresolved_fields == []
    assert outcome.next_tier is None


def test_escalate_to_next_tier():
    outcome = decide({"industry"}, {}, current_tier=0, has_budget_headroom_for_next_tier=True)
    assert outcome.decision == PolicyDecision.ESCALATE_TO_NEXT_TIER
    assert outcome.next_tier == 1


def test_stop_budget_exhausted():
    outcome = decide({"industry"}, {}, current_tier=0, has_budget_headroom_for_next_tier=False)
    assert outcome.decision == PolicyDecision.STOP_BUDGET_EXHAUSTED


def test_stop_no_more_tiers():
    outcome = decide({"industry"}, {}, current_tier=2, has_budget_headroom_for_next_tier=True)
    assert outcome.decision == PolicyDecision.STOP_NO_MORE_TIERS


def test_tier0_confidence_below_threshold():
    resolved = {"industry": _rf("industry", "Tech", 0.35, tier=0)}
    outcome = decide({"industry"}, resolved, current_tier=0, has_budget_headroom_for_next_tier=True)
    assert outcome.decision == PolicyDecision.ESCALATE_TO_NEXT_TIER
    assert outcome.unresolved_fields == ["industry"]


def test_tier1_country_confidence_above_threshold():
    # fields_needed={"country"}, resolved with country at confidence 0.65
    # current_tier=1, has_budget_headroom=True -> DONE_ALL_RESOLVED
    resolved = {"country": _rf("country", "Germany", 0.65, tier=1)}
    outcome = decide({"country"}, resolved, current_tier=1, has_budget_headroom_for_next_tier=True)
    assert outcome.decision == PolicyDecision.DONE_ALL_RESOLVED
    assert outcome.unresolved_fields == []


def test_tier1_short_description_above_threshold():
    resolved = {"short_description": _rf("short_description", "A company", 0.85, tier=1)}
    outcome = decide(
        {"short_description"}, resolved, current_tier=1, has_budget_headroom_for_next_tier=True
    )
    assert outcome.decision == PolicyDecision.DONE_ALL_RESOLVED


def test_mixed_partial_escalate():
    resolved = {
        "industry": _rf("industry", "Tech", 0.9),
    }
    outcome = decide(
        {"industry", "country"}, resolved, current_tier=1, has_budget_headroom_for_next_tier=True
    )
    assert outcome.decision == PolicyDecision.ESCALATE_TO_NEXT_TIER
    assert outcome.unresolved_fields == ["country"]


def test_threshold_boundary_counts_as_resolved():
    resolved = {"industry": _rf("industry", "Tech", ACCEPTANCE_THRESHOLD)}
    outcome = decide({"industry"}, resolved, current_tier=1, has_budget_headroom_for_next_tier=True)
    assert outcome.decision == PolicyDecision.DONE_ALL_RESOLVED


def test_tier2_all_resolved_is_done_not_stop():
    resolved = {"industry": _rf("industry", "Tech", 0.9, tier=2)}
    outcome = decide({"industry"}, resolved, current_tier=2, has_budget_headroom_for_next_tier=True)
    assert outcome.decision == PolicyDecision.DONE_ALL_RESOLVED


def test_fields_not_in_needed_are_ignored():
    resolved = {"country": _rf("country", "US", 0.9)}
    outcome = decide({"industry"}, resolved, current_tier=1, has_budget_headroom_for_next_tier=True)
    assert outcome.decision == PolicyDecision.ESCALATE_TO_NEXT_TIER
    assert outcome.unresolved_fields == ["industry"]
