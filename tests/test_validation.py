from __future__ import annotations

import pytest

from enrichment_router.validation import MissingNameError, validate_request


def test_minimal_request_name_only():
    result = validate_request({"name": "Stripe"})
    assert result.name == "Stripe"
    assert result.domain is None
    assert result.known_fields == {}
    assert result.fields_needed == {"industry", "country", "is_public", "short_description"}


def test_name_is_stripped():
    result = validate_request({"name": "  Stripe  "})
    assert result.name == "Stripe"


def test_missing_name_empty_dict():
    with pytest.raises(MissingNameError):
        validate_request({})


def test_missing_name_none():
    with pytest.raises(MissingNameError):
        validate_request({"name": None})


def test_missing_name_empty_string():
    with pytest.raises(MissingNameError):
        validate_request({"name": ""})


def test_missing_name_whitespace_only():
    with pytest.raises(MissingNameError):
        validate_request({"name": "   "})


def test_domain_passed_through():
    result = validate_request({"name": "X", "domain": "x.com"})
    assert result.domain == "x.com"


def test_all_fields_supplied():
    result = validate_request(
        {
            "name": "Acme",
            "industry": "SaaS",
            "country": "US",
            "is_public": True,
            "short_description": "Widget maker",
        }
    )
    assert result.fields_needed == set()
    assert set(result.known_fields.keys()) == {
        "industry",
        "country",
        "is_public",
        "short_description",
    }


def test_partial_fields_supplied():
    result = validate_request(
        {
            "name": "Acme",
            "industry": "SaaS",
            "country": "US",
        }
    )
    assert set(result.known_fields.keys()) == {"industry", "country"}
    assert result.fields_needed == {"is_public", "short_description"}


def test_is_public_false_preserved():
    result = validate_request({"name": "X", "is_public": False})
    assert result.known_fields["is_public"] is False


def test_unknown_keys_ignored():
    result = validate_request({"name": "X", "noise": 1})
    assert "noise" not in result.known_fields


def test_is_public_true_preserved():
    result = validate_request({"name": "X", "is_public": True})
    assert result.known_fields["is_public"] is True
