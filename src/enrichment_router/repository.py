from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import create_engine, desc, select
from sqlalchemy.orm import Session

from .db import (
    Base,
    EnrichmentRunORM,
    RecordORM,
    TraceEventORM,
    get_database_url,
)
from .models import EnrichmentRequest, EnrichmentResult, FieldName, ResolvedField
from .trace import TraceEvent

# Module-level engine: shared across all repository operations within
# a process. Lazy-initialised on first call to get_engine() so
# tests can call configure_engine("sqlite:///:memory:") before any
# repository function touches the engine. configure_engine is
# idempotent — callers (including test setup) can call it multiple
# times without side effects.
_engine: Optional[Any] = None


def configure_engine(database_url: str | None = None) -> Any:
    global _engine
    if _engine is not None:
        return _engine
    url = database_url or get_database_url()
    _engine = create_engine(url, future=True)
    Base.metadata.create_all(_engine)
    return _engine


def get_engine() -> Any:
    if _engine is None:
        configure_engine()
    return _engine


@dataclass
class RecordSummary:
    record_id: int
    name: str
    status: str
    total_cost_usd: float
    total_latency_ms: float
    resolved_field_count: int
    created_at: str


@dataclass
class RecordDetail:
    record_id: int
    name: str
    domain: Optional[str]
    input_json: dict
    run_id: int
    status: str
    total_cost_usd: float
    total_latency_ms: float
    resolved: dict
    unresolved_fields: list
    created_at: str


def _serialize_resolved(resolved: dict[FieldName, ResolvedField]) -> str:
    """Convert resolved fields dict to a write-once JSON string.

    Each ResolvedField is flattened into a plain dict so the JSON is
    self-describing and the consuming frontend can render it directly
    without needing the Pydantic model definitions. The caller_supplied
    flag is included so the UI can display "Caller-supplied" instead of
    a tier label.
    """
    return json.dumps(
        {
            name: {
                "value": rf.value,
                "tier": rf.tier,
                "confidence": rf.confidence,
                "caller_supplied": rf.caller_supplied,
            }
            for name, rf in resolved.items()
        }
    )


def _deserialize_resolved(s: str) -> dict[str, dict[str, Any]]:
    try:
        return json.loads(s) if s else {}
    except json.JSONDecodeError:
        return {}


def _find_or_create_record(session: Session, request: EnrichmentRequest) -> RecordORM:
    """Look up an existing record by (name, domain), creating one if missing.

    Dedup strategy: a company submitted multiple times with the same
    name and domain reuses the same record row, accumulating runs.
    Different domains (e.g., stripe.com vs stripe.jp) are separate
    records because the enrichment results may differ by region.

    Searches for the most-recently-created match to handle the case
    where the same name+domain pair was created more than once (e.g.,
    schema migration re-inserts).
    """
    stmt = (
        select(RecordORM)
        .where(
            RecordORM.name == request.name,
            RecordORM.domain == request.domain,
        )
        .order_by(RecordORM.id.desc())
        .limit(1)
    )
    existing = session.execute(stmt).scalar_one_or_none()
    if existing is not None:
        return existing
    record = RecordORM(
        name=request.name,
        domain=request.domain,
        request_json=json.dumps(
            {
                "name": request.name,
                "domain": request.domain,
                "known_fields": dict(request.known_fields),
            }
        ),
    )
    session.add(record)
    session.flush()
    return record


def save_run(
    request: EnrichmentRequest,
    result: EnrichmentResult,
    trace_events: list[TraceEvent],
    db: Optional[Session] = None,
) -> tuple[int, int]:
    """Persist a record (or reuse), a run, and trace events.

    Returns (run_id, record_id). Deviation from spec: signature splits
    decision_state into result + trace_events because the trace lives
    separately from EnrichmentResult by design; returns both IDs so the
    API layer doesn't need a separate lookup.
    """
    session = db if db is not None else Session(get_engine())
    try:
        record = _find_or_create_record(session, request)
        run = EnrichmentRunORM(
            record_id=record.id,
            status=result.status,
            total_cost_usd=result.total_cost_usd,
            total_latency_ms=result.total_latency_ms,
            resolved_fields_json=_serialize_resolved(result.resolved),
        )
        session.add(run)
        session.flush()
        for ev in trace_events:
            session.add(
                TraceEventORM(
                    run_id=run.id,
                    node=ev.node,
                    detail=json.dumps(ev.detail),
                )
            )
        if db is None:
            session.commit()
        return run.id, record.id
    except Exception:
        session.rollback()
        raise
    finally:
        if db is None:
            session.close()


def get_record(record_id: int, db: Optional[Session] = None) -> Optional[RecordDetail]:
    session = db if db is not None else Session(get_engine())
    try:
        record = session.get(RecordORM, record_id)
        if record is None:
            return None
        request_data = json.loads(record.request_json) if record.request_json else {}
        if not record.runs:
            return RecordDetail(
                record_id=record.id,
                name=record.name,
                domain=record.domain,
                input_json=request_data,
                run_id=0,
                status="no_runs",
                total_cost_usd=0.0,
                total_latency_ms=0.0,
                resolved={},
                unresolved_fields=[],
                created_at=record.created_at.isoformat() if record.created_at else "",
            )
        latest = record.runs[0]
        return RecordDetail(
            record_id=record.id,
            name=record.name,
            domain=record.domain,
            input_json=request_data,
            run_id=latest.id,
            status=latest.status,
            total_cost_usd=latest.total_cost_usd,
            total_latency_ms=latest.total_latency_ms,
            resolved=_deserialize_resolved(latest.resolved_fields_json),
            unresolved_fields=[],
            created_at=latest.created_at.isoformat() if latest.created_at else "",
        )
    finally:
        if db is None:
            session.close()


def list_records(db: Optional[Session] = None) -> list[RecordSummary]:
    session = db if db is not None else Session(get_engine())
    try:
        out: list[RecordSummary] = []
        for record in session.execute(select(RecordORM).order_by(desc(RecordORM.id))).scalars():
            created = record.created_at.isoformat() if record.created_at else ""
            if record.runs:
                latest = record.runs[0]
                resolved = _deserialize_resolved(latest.resolved_fields_json)
                out.append(
                    RecordSummary(
                        record_id=record.id,
                        name=record.name,
                        status=latest.status,
                        total_cost_usd=latest.total_cost_usd,
                        total_latency_ms=latest.total_latency_ms,
                        resolved_field_count=len(resolved),
                        created_at=created,
                    )
                )
            else:
                out.append(
                    RecordSummary(
                        record_id=record.id,
                        name=record.name,
                        status="no_runs",
                        total_cost_usd=0.0,
                        total_latency_ms=0.0,
                        resolved_field_count=0,
                        created_at=created,
                    )
                )
        return out
    finally:
        if db is None:
            session.close()


def get_trace(run_id: int, db: Optional[Session] = None) -> list[TraceEvent]:
    """Return trace events for a run, ordered by insertion order (id ASC).

    Trace events are ordered by primary key rather than timestamp
    because SQLite's datetime resolution is 1 second, and many events
    within a single run share the same second. The auto-increment id
    guarantees correct ordering.
    """
    session = db if db is not None else Session(get_engine())
    try:
        out: list[TraceEvent] = []
        for row in session.execute(
            select(TraceEventORM)
            .where(TraceEventORM.run_id == run_id)
            .order_by(TraceEventORM.id.asc())
        ).scalars():
            try:
                detail = json.loads(row.detail) if row.detail else {}
            except json.JSONDecodeError:
                detail = {}
            out.append(
                TraceEvent(
                    node=row.node,
                    detail=detail,
                    timestamp=(row.created_at.isoformat() if row.created_at else ""),
                    id=str(row.id),
                )
            )
        return out
    finally:
        if db is None:
            session.close()
