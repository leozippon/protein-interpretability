"""The bootstrap tail floor is one object, re-exported by the modules that use it."""

from __future__ import annotations

from src.transfer import information_bootstrap, prediction_addressed, statistics


def test_minimum_draws_in_tail_is_one_object() -> None:
    assert statistics.MINIMUM_DRAWS_IN_TAIL == 10.0
    assert information_bootstrap.MINIMUM_DRAWS_IN_TAIL is statistics.MINIMUM_DRAWS_IN_TAIL
    assert prediction_addressed.MINIMUM_DRAWS_IN_TAIL is statistics.MINIMUM_DRAWS_IN_TAIL
