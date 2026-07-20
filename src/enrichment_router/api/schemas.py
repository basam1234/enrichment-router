from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CreateRecordRequest(BaseModel):
    name: str
    domain: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    is_public: Optional[bool] = None
    short_description: Optional[str] = None
    max_cost_usd: float = Field(default=0.01, gt=0)
    max_latency_ms: float = Field(default=5000.0, gt=0)


class ResolvedFieldOut(BaseModel):
    name: str
    value: object
    tier: int
    confidence: float
    caller_supplied: bool = False


class CreateRecordResponse(BaseModel):
    run_id: int
    record_id: int
    name: str
    domain: Optional[str]
    status: str
    total_cost_usd: float
    total_latency_ms: float
    resolved: list[ResolvedFieldOut]
    unresolved_fields: list[str]


class RecordSummaryOut(BaseModel):
    record_id: int
    name: str
    status: str
    total_cost_usd: float
    total_latency_ms: float
    resolved_field_count: int
    created_at: str


class RecordDetailOut(BaseModel):
    record_id: int
    name: str
    domain: Optional[str]
    run_id: int
    status: str
    total_cost_usd: float
    total_latency_ms: float
    resolved: dict[str, dict]
    created_at: str


class TraceEventOut(BaseModel):
    node: str
    detail: dict
    timestamp: str
    id: str


class HealthOut(BaseModel):
    status: str
