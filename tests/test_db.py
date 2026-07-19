"""Tests for the database layer (``enrichment_router.db``).

All tests use an in-memory SQLite engine so they run without touching disk
and are isolated from each other (each test gets a fresh engine via
``init_db("sqlite:///:memory:")``).

Never uses raw SQL — everything goes through the public ORM API.
"""

from sqlalchemy import inspect
from sqlalchemy.orm import Session

from enrichment_router.db import (
    EnrichmentRunORM,
    RecordORM,
    TraceEventORM,
    init_db,
)


def test_init_db_returns_engine_and_creates_tables():
    """``init_db`` must return a usable engine and create all three tables."""
    engine = init_db("sqlite:///:memory:")
    assert engine is not None

    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    assert "records" in table_names
    assert "enrichment_runs" in table_names
    assert "trace_events" in table_names


def test_insert_and_query_relationships():
    """Insert a record → run → two trace events, then verify the ORM
    relationships and cascade-delete behaviour."""
    engine = init_db("sqlite:///:memory:")

    with Session(engine) as session:
        record = RecordORM(
            name="Acme Corp",
            domain="acme.com",
            request_json='{"name":"Acme Corp","domain":"acme.com"}',
        )
        session.add(record)
        session.flush()

        run = EnrichmentRunORM(
            record_id=record.id,
            status="done_all_resolved",
            total_cost_usd=0.05,
            total_latency_ms=1200.0,
            resolved_fields_json='{"industry":{"name":"industry","value":"Tech","tier":1,"confidence":0.9,"caller_supplied":false}}',
        )
        session.add(run)
        session.flush()

        evt1 = TraceEventORM(run_id=run.id, node="tier_1", detail="called external API")
        evt2 = TraceEventORM(run_id=run.id, node="tier_0", detail="heuristic match")
        session.add_all([evt1, evt2])
        session.commit()

        # Capture IDs as plain ints while the objects are still attached
        # to the session.  After commit the ORM expires them and they
        # become detached; accessing .id on a detached instance triggers
        # a DetachedInstanceError.
        record_id: int = record.id
        run_id: int = run.id
        evt1_id: int = evt1.id
        evt2_id: int = evt2.id

    with Session(engine) as session:
        queried_record = session.get(RecordORM, record_id)
        assert queried_record is not None
        runs = queried_record.runs
        assert len(runs) == 1

        queried_run = runs[0]
        trace_events = queried_run.trace_events
        assert len(trace_events) == 2
        assert trace_events[0].node == "tier_1"
        assert trace_events[1].node == "tier_0"

    # Cascade-delete: removing the record should also remove the run and its
    # trace events.
    with Session(engine) as session:
        rec = session.get(RecordORM, record_id)
        assert rec is not None
        session.delete(rec)
        session.commit()

    with Session(engine) as session:
        assert session.get(RecordORM, record_id) is None
        assert session.get(EnrichmentRunORM, run_id) is None
        assert session.get(TraceEventORM, evt1_id) is None
        assert session.get(TraceEventORM, evt2_id) is None
