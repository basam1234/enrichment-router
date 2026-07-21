from __future__ import annotations

import json

import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from enrichment_router.db import Base, TraceEventORM, create_engine
from enrichment_router.models import EnrichmentRequest, EnrichmentResult, ResolvedField
from enrichment_router.repository import (
    configure_engine,
    get_record,
    get_trace,
    list_records,
    save_run,
)
from enrichment_router.trace import TraceEvent

import enrichment_router.repository as repo


@pytest.fixture(autouse=True)
def _fresh_db():
    """Each test gets a fresh in-memory SQLite database with FK enforcement."""
    engine = create_engine("sqlite:///:memory:", future=True)

    @event.listens_for(engine, "connect")
    def _enable_fks(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    Base.metadata.create_all(engine)
    repo._engine = engine


def _make_request(name="Acme", domain=None, fields_needed=None):
    return EnrichmentRequest(
        name=name,
        domain=domain,
        fields_needed=fields_needed or set(),
    )


def _make_result(status="done_all_resolved", resolved=None, cost=0.0, latency=0.0):
    return EnrichmentResult(
        request=_make_request(),
        resolved=resolved or {},
        status=status,
        total_cost_usd=cost,
        total_latency_ms=latency,
    )


def _make_resolved(name, value, tier=1, confidence=0.85, caller_supplied=False):
    return ResolvedField(
        name=name,
        value=value,
        tier=tier,
        confidence=confidence,
        caller_supplied=caller_supplied,
    )


def _make_event(node="test", **detail):
    return TraceEvent(node=node, detail=dict(detail))


def test_save_run_returns_ids():
    request = _make_request(name="Stripe")
    result = _make_result(
        resolved={
            "industry": _make_resolved("industry", "Finance"),
        }
    )
    with Session(repo.get_engine()) as db:
        run_id, record_id = save_run(request, result, [], db=db)
        db.commit()
    assert run_id >= 1
    assert record_id >= 1


def test_save_run_then_get_record_round_trips():
    request = _make_request(name="Stripe")
    result = _make_result(
        resolved={
            "industry": _make_resolved("industry", "Finance", tier=1, confidence=0.85),
            "is_public": _make_resolved(
                "is_public", False, tier=1, confidence=0.5, caller_supplied=True
            ),
        }
    )
    with Session(repo.get_engine()) as db:
        run_id, record_id = save_run(request, result, [], db=db)
        db.commit()

        detail = get_record(record_id, db=db)
        assert detail is not None
        assert detail.name == "Stripe"
        assert detail.run_id == run_id

        resolved = detail.resolved
        assert resolved["industry"]["value"] == "Finance"
        assert resolved["industry"]["tier"] == 1
        assert resolved["industry"]["confidence"] == 0.85
        assert not resolved["industry"]["caller_supplied"]

        assert resolved["is_public"]["value"] is False
        assert resolved["is_public"]["caller_supplied"] is True


def test_save_run_persists_trace_events():
    request = _make_request(name="Stripe")
    result = _make_result()
    events = [
        _make_event("try_tier", tier=0),
        _make_event("check_sufficiency", decision="escalate"),
        _make_event("finalize_done", status="done"),
    ]
    with Session(repo.get_engine()) as db:
        run_id, _ = save_run(request, result, events, db=db)
        db.commit()

    trace = get_trace(run_id)
    assert len(trace) == 3
    assert trace[0].node == "try_tier"
    assert trace[0].detail["tier"] == 0
    assert trace[1].node == "check_sufficiency"
    assert trace[2].node == "finalize_done"


def test_save_run_reuses_record_for_same_name_domain():
    request = _make_request(name="Stripe", domain="stripe.com")

    with Session(repo.get_engine()) as db:
        _, record_id_1 = save_run(
            request,
            _make_result(
                resolved={
                    "industry": _make_resolved("industry", "Finance"),
                }
            ),
            [],
            db=db,
        )

        _, record_id_2 = save_run(
            request,
            _make_result(
                resolved={
                    "short_description": _make_resolved("short_description", "Payment platform"),
                }
            ),
            [],
            db=db,
        )
        db.commit()

    assert record_id_1 == record_id_2

    records = list_records()
    assert len(records) == 1

    detail = get_record(record_id_1)
    assert detail is not None
    assert "short_description" in detail.resolved


def test_different_domains_create_separate_records():
    with Session(repo.get_engine()) as db:
        _, r1 = save_run(
            _make_request(name="Stripe", domain="stripe.com"), _make_result(), [], db=db
        )
        _, r2 = save_run(
            _make_request(name="Stripe", domain="stripe.jp"), _make_result(), [], db=db
        )
        db.commit()
    assert r1 != r2
    assert len(list_records()) == 2


def test_list_records_newest_first():
    with Session(repo.get_engine()) as db:
        _, r1 = save_run(_make_request(name="First"), _make_result(), [], db=db)
        _, r2 = save_run(_make_request(name="Second"), _make_result(), [], db=db)
        db.commit()
    records = list_records()
    assert records[0].record_id == r2
    assert records[1].record_id == r1


def test_get_record_returns_none_for_unknown_id():
    with Session(repo.get_engine()) as db:
        assert get_record(99999, db=db) is None


def test_get_trace_returns_empty_for_unknown_run_id():
    with Session(repo.get_engine()) as db:
        assert get_trace(99999, db=db) == []


def test_transaction_rollback_on_error():
    engine = repo._engine
    with Session(engine) as session:
        event = TraceEventORM(
            run_id=99999,
            node="bad",
            detail=json.dumps({}),
        )
        session.add(event)
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()

    with Session(repo.get_engine()) as db:
        assert get_trace(99999, db=db) == []


def test_configure_engine_is_idempotent():
    e1 = configure_engine("sqlite:///:memory:")
    e2 = configure_engine("sqlite:///:memory:")
    assert e1 is e2
