import json
from pathlib import Path

import pytest

from enrichment_router.api.main import app  # noqa: F401 — registers startup event
from enrichment_router.repository import configure_engine

DATASET_PATH = Path(__file__).resolve().parent.parent / "eval" / "companies.json"


def _load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text())


@pytest.fixture(autouse=True)
def fresh_db():
    configure_engine("sqlite:///:memory:")
    yield


def test_dataset_has_18_records():
    assert len(_load_dataset()) == 18


def test_every_record_has_required_fields():
    for i, rec in enumerate(_load_dataset()):
        assert "name" in rec and isinstance(rec["name"], str) and rec["name"].strip()
        assert "max_cost_usd" in rec and rec["max_cost_usd"] > 0
        assert "max_latency_ms" in rec and rec["max_latency_ms"] > 0


def test_at_least_5_with_domain():
    assert len([r for r in _load_dataset() if r.get("domain")]) >= 5


def test_at_least_5_fictional_no_keyword():
    from enrichment_router.tools.heuristic import INDUSTRY_KEYWORDS

    fictional = []
    for r in _load_dataset():
        dom = r.get("domain") or ""
        if dom.endswith(".invalid") or not dom:
            name_lower = r["name"].lower()
            if not any(kw in name_lower for kw, _ in INDUSTRY_KEYWORDS):
                fictional.append(r)
    assert len(fictional) >= 5


def test_at_least_4_keyword_bearing():
    from enrichment_router.tools.heuristic import INDUSTRY_KEYWORDS

    count = sum(
        1 for r in _load_dataset() if any(kw in r["name"].lower() for kw, _ in INDUSTRY_KEYWORDS)
    )
    assert count >= 4


def test_at_least_one_all_fields_supplied():
    target = {"industry", "country", "is_public", "short_description"}
    assert len([r for r in _load_dataset() if target.issubset(r.keys())]) >= 1
