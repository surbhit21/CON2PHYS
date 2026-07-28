"""Reusable statistical-test helpers."""

from __future__ import annotations

import numpy as np


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-adjust p-values while preserving their original order."""
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = (count - rank) * p_values[original_index]
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(running_max, 1.0)
    return adjusted.tolist()
