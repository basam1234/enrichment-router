from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from enrichment_router.budget import Budget, BudgetTracker


def test_has_headroom_for_true_when_both_fit():
    budget = Budget(max_cost_usd=0.01, max_latency_ms=5000.0)
    tracker = BudgetTracker(budget=budget)
    assert tracker.has_headroom_for(0.001, 1000.0) is True


def test_cost_exhaustion_blocks_even_with_latency_headroom():
    budget = Budget(max_cost_usd=0.001, max_latency_ms=5000.0)
    tracker = BudgetTracker(budget=budget, spent_cost_usd=0.001)
    assert tracker.has_headroom_for(0.001, 1000.0) is False


def test_latency_exhaustion_blocks_even_with_cost_headroom():
    budget = Budget(max_cost_usd=0.01, max_latency_ms=1000.0)
    tracker = BudgetTracker(budget=budget, spent_latency_ms=1000.0)
    assert tracker.has_headroom_for(0.0, 500.0) is False


def test_both_exhausted_returns_false():
    budget = Budget(max_cost_usd=0.001, max_latency_ms=500.0)
    tracker = BudgetTracker(budget=budget, spent_cost_usd=0.001, spent_latency_ms=500.0)
    assert tracker.has_headroom_for(0.001, 500.0) is False


def test_exact_boundary_allowed():
    budget = Budget(max_cost_usd=0.01, max_latency_ms=5000.0)
    tracker = BudgetTracker(budget=budget)
    assert tracker.has_headroom_for(0.01, 5000.0) is True


def test_record_actual_updates_totals():
    budget = Budget(max_cost_usd=0.01, max_latency_ms=5000.0)
    tracker = BudgetTracker(budget=budget)
    tracker.record_actual(0.0005, 200.0, 100.0)
    assert tracker.spent_cost_usd == 0.0005
    assert tracker.spent_latency_ms == 200.0


def test_record_actual_emits_drift_event_above_threshold():
    budget = Budget(max_cost_usd=0.01, max_latency_ms=5000.0)
    tracker = BudgetTracker(budget=budget)
    event = tracker.record_actual(0.0, 250.0, 100.0)
    assert event is not None
    assert event.detail["kind"] == "estimate_drift"
    assert event.detail["delta_ms"] == 150.0


def test_record_actual_returns_none_below_threshold():
    budget = Budget(max_cost_usd=0.01, max_latency_ms=5000.0)
    tracker = BudgetTracker(budget=budget)
    event = tracker.record_actual(0.0, 100.4, 100.0)
    assert event is None


def test_record_actual_accumulates_across_multiple_calls():
    budget = Budget(max_cost_usd=0.01, max_latency_ms=5000.0)
    tracker = BudgetTracker(budget=budget)
    tracker.record_actual(0.001, 500.0, 500.0)
    tracker.record_actual(0.002, 300.0, 300.0)
    assert tracker.spent_cost_usd == 0.003
    assert tracker.spent_latency_ms == 800.0


def test_budget_is_frozen():
    budget = Budget(max_cost_usd=0.01, max_latency_ms=5000.0)
    with pytest.raises(FrozenInstanceError):
        budget.max_cost_usd = 0.02  # type: ignore[misc]


def test_drift_event_has_node_budget():
    budget = Budget(max_cost_usd=0.01, max_latency_ms=5000.0)
    tracker = BudgetTracker(budget=budget)
    event = tracker.record_actual(0.0, 250.0, 100.0)
    assert event is not None
    assert event.node == "budget"
