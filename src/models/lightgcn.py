from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch import nn
from tqdm import trange

from src.utils.logging import get_logger
from src.utils.math import topk_indices

LOGGER = get_logger(__name__)


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, layers: int) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.layers = layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def propagate(self, normalized_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        initial = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings = [initial]
        current = initial
        for _ in range(self.layers):
            current = torch.sparse.mm(normalized_adj, current)
            embeddings.append(current)
        final = torch.stack(embeddings, dim=0).mean(dim=0)
        return final[: self.num_users], final[self.num_users :]


@dataclass
class LightGCNRecall:
    user_factors: np.ndarray
    item_factors: np.ndarray

    @classmethod
    def fit(
        cls,
        binary_matrix: sparse.csr_matrix,
        embedding_dim: int,
        layers: int,
        epochs: int,
        samples_per_epoch: int,
        learning_rate: float,
        weight_decay: float,
        early_stop: int,
        device: torch.device,
        seed: int,
    ) -> "LightGCNRecall":
        num_users, num_items = binary_matrix.shape
        model = LightGCN(num_users, num_items, embedding_dim, layers).to(device)
        adjacency = _build_normalized_adjacency(binary_matrix, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        rng = np.random.default_rng(seed)

        rows, cols = binary_matrix.nonzero()
        rows = rows.astype(np.int64)
        cols = cols.astype(np.int64)
        seen = [set(binary_matrix[u].indices.tolist()) for u in range(num_users)]

        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        patience_count = 0
        for epoch in trange(epochs, desc="LightGCN"):
            sample_size = min(samples_per_epoch, max(len(rows), 1))
            chosen = rng.integers(0, len(rows), size=sample_size)
            users_np = rows[chosen]
            positives_np = cols[chosen]
            negatives_np = _sample_negatives(users_np, num_items, seen, rng)

            users = torch.as_tensor(users_np, dtype=torch.long, device=device)
            positives = torch.as_tensor(positives_np, dtype=torch.long, device=device)
            negatives = torch.as_tensor(negatives_np, dtype=torch.long, device=device)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            user_all, item_all = model.propagate(adjacency)
            pos_scores = (user_all[users] * item_all[positives]).sum(dim=1)
            neg_scores = (user_all[users] * item_all[negatives]).sum(dim=1)
            ranking_loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
            raw_u = model.user_embedding(users)
            raw_p = model.item_embedding(positives)
            raw_n = model.item_embedding(negatives)
            reg_loss = (raw_u.square().sum() + raw_p.square().sum() + raw_n.square().sum()) / (2 * sample_size)
            loss = ranking_loss + weight_decay * reg_loss
            loss.backward()
            optimizer.step()

            value = float(loss.detach().cpu())
            if value + 1e-5 < best_loss:
                best_loss = value
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
            if (epoch + 1) % 5 == 0:
                LOGGER.info("LightGCN epoch=%d loss=%.6f", epoch + 1, value)
            if patience_count >= early_stop:
                LOGGER.info("LightGCN early stopping at epoch %d", epoch + 1)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            user_all, item_all = model.propagate(adjacency)
        return cls(
            user_all.cpu().numpy().astype(np.float32),
            item_all.cpu().numpy().astype(np.float32),
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
        indices = topk_indices(scores, topk)
        return item_ids[indices], scores[indices].astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, user_factors=self.user_factors, item_factors=self.item_factors)

    @classmethod
    def load(cls, path: str | Path) -> "LightGCNRecall":
        data = np.load(path)
        return cls(data["user_factors"], data["item_factors"])


def _build_normalized_adjacency(matrix: sparse.csr_matrix, device: torch.device) -> torch.Tensor:
    num_users, num_items = matrix.shape
    users, items = matrix.nonzero()
    item_nodes = items + num_users
    rows = np.concatenate([users, item_nodes]).astype(np.int64)
    cols = np.concatenate([item_nodes, users]).astype(np.int64)
    degrees = np.bincount(rows, minlength=num_users + num_items).astype(np.float32)
    degrees = np.maximum(degrees, 1.0)
    values = 1.0 / np.sqrt(degrees[rows] * degrees[cols])
    indices = torch.as_tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
    values_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices, values_tensor, size=(num_users + num_items, num_users + num_items), device=device
    ).coalesce()


def _sample_negatives(
    users: np.ndarray,
    num_items: int,
    seen: list[set[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = rng.integers(0, num_items, size=len(users), dtype=np.int64)
    invalid = np.array([item in seen[int(user)] for user, item in zip(users, negatives)], dtype=bool)
    attempts = 0
    while invalid.any():
        negatives[invalid] = rng.integers(0, num_items, size=int(invalid.sum()), dtype=np.int64)
        invalid = np.array([item in seen[int(user)] for user, item in zip(users, negatives)], dtype=bool)
        attempts += 1
        if attempts > 100:
            raise RuntimeError("负采样失败：部分用户可能与几乎全部物品发生过交互。")
    return negatives
