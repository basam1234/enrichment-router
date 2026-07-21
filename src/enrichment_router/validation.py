from __future__ import annotations

from .models import EnrichmentRequest, FieldName, TARGET_FIELDS


class MissingNameError(Exception):
    """Raised when an enrichment request has no usable company name.

    The API layer catches this and converts it to HTTP 422. Kept as a
    custom exception rather than ValueError so the API can dispatch on
    type without string-matching the message.
    """


def validate_request(raw: dict) -> EnrichmentRequest:
    """Validate a raw request dict and return a typed EnrichmentRequest.

    Rules:
      - `name` is required; if missing, None, non-str, empty, or
        whitespace-only, raise MissingNameError. The check is on the
        *string* shape, not just presence — a caller sending {"name": ""}
        or {"name": "   "} gets the same error as omitting the key.
      - `domain` is optional, passed through as-is (may be None).
      - For each of the four target fields, only include it in
        `known_fields` if the caller actually supplied a non-None value.
        The router treats absence and explicit None identically — both
        mean "this field still needs enrichment."
      - `fields_needed` = TARGET_FIELDS minus the keys in known_fields.
        If the caller supplied all four, fields_needed is empty and the
        router must do zero work in later commits.
    """
    name: object = raw.get("name")

    if not isinstance(name, str) or not name.strip():
        raise MissingNameError("name is required and must be a non-empty string")

    known_fields: dict[FieldName, object] = {}
    for field in TARGET_FIELDS:
        value = raw.get(field)
        if value is not None:
            known_fields[field] = value

    fields_needed: set[FieldName] = set(TARGET_FIELDS) - set(known_fields.keys())

    return EnrichmentRequest(
        name=name.strip(),
        domain=raw.get("domain"),
        known_fields=known_fields,
        fields_needed=fields_needed,
    )
