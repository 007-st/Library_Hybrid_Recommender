from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from src.utils.math import topk_indices


@dataclass
class Item2VecRecall:
    user_factors: np.ndarray
    item_factors: np.ndarray

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        window: int,
        epochs: int,
        negative: int,
        workers: int,
        recent_history_length: int,
        seed: int,
    ) -> "Item2VecRecall":
        sequences: list[list[str]] = []
        histories: dict[int, list[int]] = {}
        ordered = train.sort_values(["user_idx", "borrow_time", "inter_id"], kind="stable")
        for user_idx, group in ordered.groupby("user_idx", sort=False):
            sequence = group["item_idx"].astype(int).tolist()
            histories[int(user_idx)] = sequence
            if len(sequence) >= 2:
                sequences.append([str(item) for item in sequence])

        item_factors = np.zeros((num_items, embedding_dim), dtype=np.float32)
        if sequences:
            model = Word2Vec(
                sentences=sequences,
                vector_size=embedding_dim,
                window=window,
                min_count=1,
                workers=workers,
                sg=1,
                negative=negative,
                epochs=epochs,
                seed=seed,
            )
            for item in range(num_items):
                token = str(item)
                if token in model.wv:
                    item_factors[item] = model.wv[token]

        item_norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
        item_factors = item_factors / np.maximum(item_norms, 1e-12)
        user_factors = np.zeros((num_users, embedding_dim), dtype=np.float32)
        for user, history in histories.items():
            recent = history[-recent_history_length:]
            if not recent:
                continue
            vectors = item_factors[np.asarray(recent, dtype=np.int64)]
            weights = np.linspace(0.5, 1.0, num=len(recent), dtype=np.float32)
            user_vector = np.average(vectors, axis=0, weights=weights)
            norm = np.linalg.norm(user_vector)
            if norm > 0:
                user_factors[user] = user_vector / norm
        return cls(user_factors, item_factors)

    def score_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return np.sum(self.user_factors[users] * self.item_factors[items], axis=1).astype(np.float32)

    def recommend(
        self,
        user_idx: int,
        topk: int,
        seen_items: set[int] | None = None,
        allowed_items: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if allowed_items is None:
            item_ids = np.arange(len(self.item_factors), dtype=np.int64)
            scores = self.item_factors @ self.user_factors[user_idx]
        else:
            item_ids = np.asarray(allowed_items, dtype=np.int64)
            scores = self.item_factors[item_ids] @ self.user_factors[user_idx]
        if seen_items:
            mask = np.isin(item_ids, np.fromiter(seen_items, dtype=np.int64))
            scores = scores.copy()
            scores[mask] = -np.inf
        indices = topk_indices(scores, topk)
        return item_ids[indices], scores[indices].astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, user_factors=self.user_factors, item_factors=self.item_factors)

    @classmethod
    def load(cls, path: str | Path) -> "Item2VecRecall":
        data = np.load(path)
        return cls(data["user_factors"], data["item_factors"])
