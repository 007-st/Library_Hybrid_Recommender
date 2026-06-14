from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


class NeuralRanker(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_embedding_dim: int,
        item_embedding_dim: int,
        user_feature_dim: int,
        item_feature_dim: int,
        numeric_feature_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        self.item_embedding = nn.Embedding(num_items, item_embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)

        input_dim = (
            user_embedding_dim + item_embedding_dim + user_feature_dim
            + item_feature_dim + numeric_feature_dim
        )
        layers: list[nn.Module] = []
        current = input_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(current, int(hidden)),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current = int(hidden)
        layers.append(nn.Linear(current, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([
            self.user_embedding(users),
            self.item_embedding(items),
            user_features,
            item_features,
            numeric_features,
        ], dim=1)
        return self.mlp(features).squeeze(1)


@dataclass
class RankerCheckpoint:
    model_kwargs: dict
    state_dict: dict[str, torch.Tensor]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_kwargs": self.model_kwargs, "state_dict": self.state_dict}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "RankerCheckpoint":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        return cls(payload["model_kwargs"], payload["state_dict"])

    def build_model(self, device: torch.device) -> NeuralRanker:
        model = NeuralRanker(**self.model_kwargs)
        model.load_state_dict(self.state_dict)
        return model.to(device)


class RankerFeatureStore:
    """统一构造精排模型所需的 ID、元数据、召回分数和统计特征。"""

    def __init__(
        self,
        user_features: np.ndarray,
        item_features: np.ndarray,
        popularity: np.ndarray,
        renew_user: np.ndarray,
        renew_item: np.ndarray,
        als=None,
        lightgcn=None,
        item2vec=None,
    ) -> None:
        self.user_features = np.asarray(user_features, dtype=np.float32)
        self.item_features = np.asarray(item_features, dtype=np.float32)
        self.popularity = np.asarray(popularity, dtype=np.float32)
        self.renew_user = np.asarray(renew_user, dtype=np.float32)
        self.renew_item = np.asarray(renew_item, dtype=np.float32)
        self.als = als
        self.lightgcn = lightgcn
        self.item2vec = item2vec
        popularity_max = float(np.log1p(self.popularity).max()) if self.popularity.size else 0.0
        renew_user_max = float(self.renew_user.max()) if self.renew_user.size else 0.0
        renew_item_max = float(self.renew_item.max()) if self.renew_item.size else 0.0
        self.popularity_scale = max(popularity_max, 1.0)
        self.renew_scale = max(renew_user_max, renew_item_max, 1.0)

    @property
    def numeric_feature_dim(self) -> int:
        return 6

    @staticmethod
    def _stable_score(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return np.sign(values) * np.log1p(np.abs(values))

    def numpy_batch(self, users: np.ndarray, items: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        zeros = np.zeros(len(users), dtype=np.float32)
        als_scores = self.als.score_pairs(users, items) if self.als is not None else zeros
        lgn_scores = self.lightgcn.score_pairs(users, items) if self.lightgcn is not None else zeros
        i2v_scores = self.item2vec.score_pairs(users, items) if self.item2vec is not None else zeros
        numeric = np.column_stack([
            self._stable_score(als_scores),
            self._stable_score(lgn_scores),
            self._stable_score(i2v_scores),
            np.log1p(self.popularity[items]) / self.popularity_scale,
            self.renew_user[users] / self.renew_scale,
            self.renew_item[items] / self.renew_scale,
        ]).astype(np.float32)
        return self.user_features[users], self.item_features[items], numeric

    def torch_batch(
        self,
        users: np.ndarray,
        items: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_meta, item_meta, numeric = self.numpy_batch(users, items)
        return (
            torch.as_tensor(users, dtype=torch.long, device=device),
            torch.as_tensor(items, dtype=torch.long, device=device),
            torch.as_tensor(user_meta, dtype=torch.float32, device=device),
            torch.as_tensor(item_meta, dtype=torch.float32, device=device),
            torch.as_tensor(numeric, dtype=torch.float32, device=device),
        )
