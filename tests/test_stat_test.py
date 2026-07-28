"""Tests for reusable statistical helpers."""

from __future__ import annotations

import numpy as np

from src.stat_test import holm_adjust


def test_holm_adjust_preserves_input_order() -> None:
    adjusted = holm_adjust([0.04, 0.01, 0.03])

    np.testing.assert_allclose(adjusted, [0.06, 0.03, 0.06])


def test_holm_adjust_caps_values_at_one() -> None:
    assert holm_adjust([0.8, 0.9]) == [1.0, 1.0]


def test_holm_adjust_accepts_empty_input() -> None:
    assert holm_adjust([]) == []
