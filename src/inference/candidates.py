from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.popularity import PopularityRecall


@dataclass
class CandidateScores:
    item_idx: int
    als_score: float = 0.0
    lightgcn_score: float = 0.0
    item2vec_score: float = 0.0
    popularity_score: float = 0.0


class CandidateGenerator:
    def __init__(self, als, lightgcn, item2vec, popularity: np.ndarray) -> None:
        self.als = als
        self.lightgcn = lightgcn
        self.item2vec = item2vec
        self.popularity_values = np.asarray(popularity, dtype=np.float32)
        self.active_items = np.flatnonzero(self.popularity_values > 0).astype(np.int64)
        self.popularity = PopularityRecall(self.popularity_values)

    def generate(self, user_idx: int, seen: set[int], topk_each: int, max_candidates: int) -> list[int]:
        merged: dict[int, CandidateScores] = {}
        sources = [
            ("als_score", self.als),
            ("lightgcn_score", self.lightgcn),
            ("item2vec_score", self.item2vec),
        ]
        for field, model in sources:
            if model is None:
                continue
            items, scores = model.recommend(
                user_idx, topk_each, seen, allowed_items=self.active_items
            )
            for item, score in zip(items, scores):
                record = merged.setdefault(int(item), CandidateScores(int(item)))
                setattr(record, field, float(score))
        items, scores = self.popularity.recommend(topk_each, seen)
        for item, score in zip(items, scores):
            record = merged.setdefault(int(item), CandidateScores(int(item)))
            record.popularity_score = float(score)

        def source_rank(record: CandidateScores) -> float:
            values = [record.als_score, record.lightgcn_score, record.item2vec_score, record.popularity_score]
            return float(sum(np.sign(v) * np.log1p(abs(v)) for v in values))

        ranked = sorted(merged.values(), key=source_rank, reverse=True)
        return [record.item_idx for record in ranked[:max_candidates]]
