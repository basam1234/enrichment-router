import pytest
from fastapi.testclient import TestClient

from enrichment_router.api.main import app
from enrichment_router.repository import configure_engine


@pytest.fixture(autouse=True)
def fresh_db():
    configure_engine("sqlite:///:memory:")
    yield


def test_root_serves_index_html():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Enrichment Router" in resp.text


def test_static_index_html_served():
    client = TestClient(app)
    resp = client.get("/static/index.html")
    assert resp.status_code == 200
    assert "Enrichment Router" in resp.text
