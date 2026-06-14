from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from tqdm import trange

from src.utils.logging import get_logger
from src.utils.math import topk_indices

LOGGER = get_logger(__name__)


@dataclass
class ALSRecall:
    user_factors: np.ndarray
    item_factors: np.ndarray

    @classmethod
    def fit_torch(
        cls,
        weighted_matrix: sparse.csr_matrix,
        factors: int,
        iterations: int,
        regularization: float,
        alpha: float,
        batch_rows: int,
        device: torch.device,
        seed: int,
    ) -> "ALSRecall":
        """纯 PyTorch 隐式反馈 ALS，支持 CUDA 批量线性方程求解。"""
        rng = np.random.default_rng(seed)
        user_item = weighted_matrix.astype(np.float32).tocsr(copy=True)
        confidence = user_item.copy()
        confidence.data = 1.0 + alpha * confidence.data
        item_user = confidence.T.tocsr()

        item_init = rng.normal(0.0, 0.01, size=(confidence.shape[1], factors)).astype(np.float32)
        item_factors = torch.as_tensor(item_init, device=device)

        for iteration in trange(iterations, desc="TorchALS"):
            user_factors = _least_squares_update(
                confidence, item_factors, regularization, batch_rows, device
            )
            item_factors = _least_squares_update(
                item_user, user_factors, regularization, batch_rows, device
            )
            if iteration == 0 or (iteration + 1) % 5 == 0:
                LOGGER.info("ALS iteration %d/%d", iteration + 1, iterations)

        return cls(
            user_factors=user_factors.detach().cpu().numpy().astype(np.float32),
            item_factors=item_factors.detach().cpu().numpy().astype(np.float32),
        )

    @classmethod
    def fit_implicit(
        cls,
        weighted_matrix: sparse.csr_matrix,
        factors: int,
        iterations: int,
        regularization: float,
        alpha: float,
        seed: int,
    ) -> "ALSRecall":
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as exc:
            raise RuntimeError(
                "未安装 implicit。可将 als.backend 改为 torch，或在 Python 3.10/3.11 环境安装 "
                "pip install implicit==0.7.2。"
            ) from exc
        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            random_state=seed,
        )
        model.fit(weighted_matrix.astype(np.float32))
        return cls(
            user_factors=np.asarray(model.user_factors, dtype=np.float32),
            item_factors=np.asarray(model.item_factors, dtype=np.float32),
        )

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
        finite = np.isfinite(scores)
        if not finite.any():
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        indices = topk_indices(scores, topk)
        return item_ids[indices], scores[indices].astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, user_factors=self.user_factors, item_factors=self.item_factors)

    @classmethod
    def load(cls, path: str | Path) -> "ALSRecall":
        data = np.load(path)
        return cls(data["user_factors"], data["item_factors"])


def _least_squares_update(
    interactions: sparse.csr_matrix,
    fixed_factors: torch.Tensor,
    regularization: float,
    batch_rows: int,
    device: torch.device,
) -> torch.Tensor:
    interactions = interactions.tocsr()
    num_rows = interactions.shape[0]
    factor_dim = fixed_factors.shape[1]
    output = torch.zeros((num_rows, factor_dim), dtype=torch.float32, device=device)
    gram = fixed_factors.T @ fixed_factors
    eye = torch.eye(factor_dim, dtype=torch.float32, device=device)
    base = gram + regularization * eye

    indptr = interactions.indptr
    indices = interactions.indices
    data = interactions.data

    for start in range(0, num_rows, batch_rows):
        end = min(start + batch_rows, num_rows)
        counts_np = indptr[start + 1 : end + 1] - indptr[start:end]
        batch_size = end - start
        edge_start = int(indptr[start])
        edge_end = int(indptr[end])
        if edge_end == edge_start:
            continue

        counts = torch.as_tensor(counts_np, dtype=torch.long, device=device)
        local_rows = torch.repeat_interleave(
            torch.arange(batch_size, device=device, dtype=torch.long), counts
        )
        neighbor_ids = torch.as_tensor(indices[edge_start:edge_end], dtype=torch.long, device=device)
        confidence = torch.as_tensor(data[edge_start:edge_end], dtype=torch.float32, device=device)
        vectors = fixed_factors[neighbor_ids]

        outer = vectors.unsqueeze(2) * vectors.unsqueeze(1)
        weighted_outer = outer * (confidence - 1.0).view(-1, 1, 1)
        delta_flat = torch.zeros(
            (batch_size, factor_dim * factor_dim), dtype=torch.float32, device=device
        )
        delta_flat.index_add_(0, local_rows, weighted_outer.reshape(-1, factor_dim * factor_dim))

        rhs = torch.zeros((batch_size, factor_dim), dtype=torch.float32, device=device)
        rhs.index_add_(0, local_rows, vectors * confidence.view(-1, 1))

        matrices = base.unsqueeze(0) + delta_flat.view(batch_size, factor_dim, factor_dim)
        solved = torch.linalg.solve(matrices, rhs.unsqueeze(-1)).squeeze(-1)
        output[start:end] = solved
    return output
