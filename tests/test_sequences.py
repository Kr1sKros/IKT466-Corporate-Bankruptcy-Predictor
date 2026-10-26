"""Planned checks for latest-year anchors and backwards GRU histories."""

import pytest


@pytest.mark.skip(reason="Pending history construction and short-history policy.")
def test_sequences_contract() -> None:
    """Check 1–5-year order, labels, IDs, and matching baseline/GRU anchors."""
    # TODO: Explicit annual fixtures; test latest-year-only baseline selection too.
    raise NotImplementedError("Test contract is pending; keep this test skipped.")
