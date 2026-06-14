from __future__ import annotations

import numpy as np


def minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-12:
        return np.ones_like(values) if high > 0 else np.zeros_like(values)
    return (values - low) / (high - low)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores)
    if k <= 0 or scores.size == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, scores.size)
    part = np.argpartition(-scores, kth=k - 1)[:k]
    return part[np.argsort(-scores[part])]
