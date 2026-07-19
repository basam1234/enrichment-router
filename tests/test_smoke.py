"""Smoke test for the scaffold — verifies the editable install actually
works before any real feature code exists. This is a genuine test, not
a placeholder: if packaging or the src/ layout breaks, this catches it
immediately instead of surfacing as a mysterious import error three
commits from now.
"""


def test_package_is_importable():
    import enrichment_router

    assert enrichment_router is not None
    