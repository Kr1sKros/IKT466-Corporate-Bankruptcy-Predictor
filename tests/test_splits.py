"""Planned checks for company-disjoint partitions and reproducibility."""

import pytest


@pytest.mark.skip(reason="Pending company-split implementation and agreed fractions.")
def test_splits_contract() -> None:
    """Check disjoint company IDs, complete row coverage, and seed reproducibility."""
    # TODO: Use companies with multiple years and assert no company crosses splits.
    raise NotImplementedError("Test contract is pending; keep this test skipped.")
