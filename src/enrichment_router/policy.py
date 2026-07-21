from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import FieldName, ResolvedField

# Per-field acceptance threshold. Lowered to 0.5 to accept Tier 1
# (Wikipedia) country and is_public resolutions, prioritizing cost
# savings over strict LLM verification.
ACCEPTANCE_THRESHOLD: float = 0.5


class PolicyDecision(str, Enum):
    ESCALATE_TO_NEXT_TIER = "escalate_to_next_tier"
    DONE_ALL_RESOLVED = "done_all_resolved"
    STOP_BUDGET_EXHAUSTED = "stop_budget_exhausted"
    STOP_NO_MORE_TIERS = "stop_no_more_tiers"


@dataclass
class PolicyOutcome:
    decision: PolicyDecision
    unresolved_fields: list[FieldName]
    next_tier: int | None
    reason: str


def _accepted_fields(
    fields_needed: set[FieldName],
    resolved: dict[FieldName, ResolvedField],
) -> set[FieldName]:
    return {
        f for f in fields_needed if f in resolved and resolved[f].confidence >= ACCEPTANCE_THRESHOLD
    }


def decide(
    fields_needed: set[FieldName],
    resolved: dict[FieldName, ResolvedField],
    current_tier: int,
    has_budget_headroom_for_next_tier: bool,
) -> PolicyOutcome:
    """Pure-Python escalation decision.

    This function is intentionally pure — it receives facts (field
    state, budget headroom) as arguments instead of reaching into a
    tracker or resolver. Pure functions are trivial to test, reason
    about, and replay from a trace log.

    Four cases checked in order:
    1. DONE_ALL_RESOLVED — all needed fields accepted.
    2. STOP_BUDGET_EXHAUSTED — not done, but no budget for the next tier.
    3. STOP_NO_MORE_TIERS — not done, budget exists, but tier 2 is max.
    4. ESCALATE_TO_NEXT_TIER — not done, budget exists, next tier exists.
    """
    accepted = _accepted_fields(fields_needed, resolved)
    unresolved = sorted(fields_needed - accepted)

    if not unresolved:
        return PolicyOutcome(
            decision=PolicyDecision.DONE_ALL_RESOLVED,
            unresolved_fields=[],
            next_tier=None,
            reason="all needed fields resolved at or above threshold",
        )

    if not has_budget_headroom_for_next_tier:
        return PolicyOutcome(
            decision=PolicyDecision.STOP_BUDGET_EXHAUSTED,
            unresolved_fields=unresolved,
            next_tier=None,
            reason="budget exhausted before next tier",
        )

    next_tier = current_tier + 1
    if next_tier > 2:
        return PolicyOutcome(
            decision=PolicyDecision.STOP_NO_MORE_TIERS,
            unresolved_fields=unresolved,
            next_tier=None,
            reason="no more tiers available",
        )

    return PolicyOutcome(
        decision=PolicyDecision.ESCALATE_TO_NEXT_TIER,
        unresolved_fields=unresolved,
        next_tier=next_tier,
        reason=f"escalating to tier {next_tier}",
    )
