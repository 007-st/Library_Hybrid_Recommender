from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.math import topk_indices


@dataclass
class PopularityRecall:
    popularity: np.ndarray

    def recommend(self, topk: int, seen_items: set[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        scores = np.log1p(self.popularity.astype(np.float32))
        if seen_items:
            scores = scores.copy()
            scores[np.fromiter(seen_items, dtype=np.int64)] = -np.inf
        indices = topk_indices(scores, topk)
        return indices, scores[indices]
