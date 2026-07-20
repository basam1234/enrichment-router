from __future__ import annotations

from dataclasses import dataclass

from .trace import TraceEvent, make_event

# Drift events are only emitted when the absolute difference between
# expected and actual latency exceeds 1.0 ms. Sub-millisecond variance
# is normal noise from the OS scheduler and doesn't warrant a trace
# event. Cost drift is intentionally not checked — latency is the
# better proxy for "did something unexpected happen" because cost for
# tiers 0 and 1 is zero, and LLM cost is computed from exact token
# counts which don't drift. Latency can spike due to network jitter,
# cold starts, or CPU contention, so monitoring it catches real issues.
DRIFT_THRESHOLD_MS: float = 1.0


@dataclass(frozen=True)
class Budget:
    """Immutable budget for a single enrichment run.

    Frozen so that the tracker can rely on its budget not changing
    mid-run — prevents bugs where the caller mutates the budget after
    the tracker has already made decisions about headroom.
    """

    max_cost_usd: float
    max_latency_ms: float


@dataclass
class BudgetTracker:
    """Tracks spend against a Budget across multiple tier executions.

    `has_headroom_for` checks the joint constraint (both cost and
    latency must pass) against declared tier estimates before a tier
    runs. This prevents running a tier that would exceed the budget
    even if the other dimension has slack — the router must respect
    both limits simultaneously.
    """

    budget: Budget
    spent_cost_usd: float = 0.0
    spent_latency_ms: float = 0.0

    def has_headroom_for(
        self,
        declared_cost_usd: float,
        declared_latency_estimate_ms: float,
    ) -> bool:
        """Check whether there's budget left for a tier that declares
        given cost and latency estimates."""
        cost_ok = self.spent_cost_usd + declared_cost_usd <= self.budget.max_cost_usd + 1e-12
        latency_ok = (
            self.spent_latency_ms + declared_latency_estimate_ms
            <= self.budget.max_latency_ms + 1e-12
        )
        return cost_ok and latency_ok

    def record_actual(
        self,
        actual_cost_usd: float,
        actual_latency_ms: float,
        estimated_latency_ms: float,
    ) -> TraceEvent | None:
        """Update running totals with measured values. Returns a
        TraceEvent describing estimate-vs-actual drift, or None if
        below DRIFT_THRESHOLD_MS.

        Deviation from spec: returns TraceEvent | None instead of None
        so the caller can append the event to the run's trace list
        without the tracker needing a reference to that list.
        """
        self.spent_cost_usd += actual_cost_usd
        self.spent_latency_ms += actual_latency_ms

        delta = actual_latency_ms - estimated_latency_ms
        if abs(delta) >= DRIFT_THRESHOLD_MS:
            return make_event(
                "budget",
                kind="estimate_drift",
                estimated_latency_ms=estimated_latency_ms,
                actual_latency_ms=actual_latency_ms,
                delta_ms=delta,
            )
        return None
