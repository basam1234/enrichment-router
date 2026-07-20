from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool

import pytest
from fastapi.testclient import TestClient

from enrichment_router.api.main import app, get_llm_client, get_wiki_fetcher
from enrichment_router.db import Base
from enrichment_router.tools.llm import FakeLLMClient, LLMResponse

import enrichment_router.repository as repo


@pytest.fixture(autouse=True)
def fresh_db():
    """In-memory SQLite shared across threads via StaticPool.

    SQLite ``:memory:`` databases are per-connection — without
    StaticPool each new connection sees an empty database. FastAPI's
    TestClient runs route handlers in a thread pool, so the test
    thread and the handler thread would otherwise get separate
    databases. StaticPool reuses the same connection for every
    session, making the in-memory database visible everywhere.
    ``check_same_thread=False`` allows that single connection to be
    used from any thread.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _enable_fks(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    Base.metadata.create_all(engine)
    repo._engine = engine
    yield


@pytest.fixture
def client():
    return TestClient(app)


def stub_wiki_fetcher(name: str):
    return None


@pytest.fixture
def fake_llm():
    return FakeLLMClient(
        {
            "Acme": LLMResponse(
                text='{"industry":"Technology","country":"US","is_public":true,"short_description":"A co."}',
                prompt_tokens=50,
                completion_tokens=20,
            ),
        }
    )


@pytest.fixture
def override_deps(fake_llm):
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_wiki_fetcher] = lambda: stub_wiki_fetcher
    yield
    app.dependency_overrides.clear()


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_record_missing_name(client, override_deps):
    resp = client.post("/api/records", json={"domain": "x.com"})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    # When name is missing from the JSON body, Pydantic v2 catches it
    # before our route handler runs and returns structured errors.
    errors = detail if isinstance(detail, list) else [detail]
    assert any(err.get("type") == "missing" and "name" in str(err.get("loc", "")) for err in errors)


def test_create_record_empty_name(client, override_deps):
    resp = client.post("/api/records", json={"name": "   "})
    assert resp.status_code == 422


def test_create_record_happy_path(client, override_deps):
    resp = client.post(
        "/api/records",
        json={"name": "Acme", "max_cost_usd": 0.05, "max_latency_ms": 10000},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "done_all_resolved"
    assert data["run_id"] >= 1
    assert data["record_id"] >= 1
    tiers = {rf["name"]: rf["tier"] for rf in data["resolved"]}
    assert tiers.get("industry") == 2
    assert tiers.get("country") == 2
    assert tiers.get("is_public") == 2
    assert tiers.get("short_description") == 2


def test_create_record_all_fields_supplied(client, override_deps):
    resp = client.post(
        "/api/records",
        json={
            "name": "Stripe",
            "industry": "Finance",
            "country": "US",
            "is_public": False,
            "short_description": "Payment platform",
            "max_cost_usd": 0.01,
            "max_latency_ms": 5000,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "done_all_resolved"
    assert data["total_cost_usd"] == 0.0
    assert data["total_latency_ms"] == 0.0
    caller_supplied = all(rf["caller_supplied"] for rf in data["resolved"])
    assert caller_supplied


def test_list_records_non_empty(client, override_deps):
    client.post(
        "/api/records",
        json={"name": "Acme", "max_cost_usd": 0.05, "max_latency_ms": 10000},
    )
    resp = client.get("/api/records")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1


def test_get_record_by_id(client, override_deps):
    post_resp = client.post(
        "/api/records",
        json={"name": "Acme", "max_cost_usd": 0.05, "max_latency_ms": 10000},
    )
    record_id = post_resp.json()["record_id"]

    resp = client.get(f"/api/records/{record_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Acme"


def test_get_record_not_found(client):
    resp = client.get("/api/records/99999")
    assert resp.status_code == 404


def test_get_trace_route(client, override_deps):
    post_resp = client.post(
        "/api/records",
        json={"name": "Acme", "max_cost_usd": 0.05, "max_latency_ms": 10000},
    )
    record_id = post_resp.json()["record_id"]

    resp = client.get(f"/api/records/{record_id}/trace")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert any(ev["node"] == "try_tier" for ev in data)


def test_no_cors_header(client, override_deps):
    client.post(
        "/api/records",
        json={"name": "Acme", "max_cost_usd": 0.05, "max_latency_ms": 10000},
    )
    resp = client.get("/api/records")
    assert "access-control-allow-origin" not in resp.headers
