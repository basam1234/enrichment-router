from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

FieldName = Literal["industry", "country", "is_public", "short_description"]

Tier = Literal[0, 1, 2]

RunStatus = Literal[
    "done_all_resolved",
    "partial_budget",
    "partial_no_more_tiers",
]

TARGET_FIELDS: tuple[FieldName, ...] = (
    "industry",
    "country",
    "is_public",
    "short_description",
)


class EnrichmentRequest(BaseModel):
    """Validated, post-validation view of an incoming enrichment request.

    ``known_fields`` holds only the four target fields the caller actually
    supplied (others omitted, not None).  ``fields_needed`` is the complement
    against the four target fields — empty if the caller already supplied
    all four, in which case the router must do zero work.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    domain: Optional[str] = None
    known_fields: dict[FieldName, object] = Field(default_factory=dict)
    fields_needed: set[FieldName] = Field(default_factory=set)


class ResolvedField(BaseModel):
    """A single field resolved by some tier, with provenance.

    ``caller_supplied=True`` marks fields the caller already provided
    (not enriched by any tool).  The frontend displays "Caller-supplied"
    instead of a tier label for these.  The ``tier`` value for caller-
    supplied fields is a placeholder (``0``) and should not be displayed.
    Without this flag, caller-supplied fields would be indistinguishable
    from tier-0 heuristic results in the UI's "which tier resolved
    each field" column.
    """

    model_config = ConfigDict(extra="forbid")
    name: FieldName
    value: object
    tier: Tier
    confidence: float
    caller_supplied: bool = False


class EnrichmentResult(BaseModel):
    """Final output of one enrichment run."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    request: EnrichmentRequest
    resolved: dict[FieldName, ResolvedField] = Field(default_factory=dict)
    status: RunStatus
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    unresolved_fields: list[FieldName] = Field(default_factory=list)
