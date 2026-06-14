from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.utils.io import load_json


@dataclass
class ProcessedData:
    root: Path
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    users: pd.DataFrame
    books: pd.DataFrame
    train_matrix: sparse.csr_matrix
    binary_matrix: sparse.csr_matrix
    user_features: np.ndarray
    item_features: np.ndarray
    popularity: np.ndarray
    renew_user: np.ndarray
    renew_item: np.ndarray
    mappings: dict
    meta: dict

    @classmethod
    def load(cls, root: str | Path) -> "ProcessedData":
        root = Path(root)

        def read_split(name: str) -> pd.DataFrame:
            frame = pd.read_csv(root / f"{name}.csv", encoding="utf-8-sig", low_memory=False)
            for column in ("borrow_time", "return_time", "renew_time"):
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
            return frame

        return cls(
            root=root,
            train=read_split("train"),
            valid=read_split("valid"),
            test=read_split("test"),
            users=pd.read_csv(root / "users.csv", encoding="utf-8-sig", low_memory=False),
            books=pd.read_csv(root / "books.csv", encoding="utf-8-sig", low_memory=False),
            train_matrix=sparse.load_npz(root / "train_matrix.npz").tocsr(),
            binary_matrix=sparse.load_npz(root / "train_binary_matrix.npz").tocsr(),
            user_features=np.load(root / "user_features.npy"),
            item_features=np.load(root / "item_features.npy"),
            popularity=np.load(root / "popularity.npy"),
            renew_user=np.load(root / "renew_user.npy"),
            renew_item=np.load(root / "renew_item.npy"),
            mappings=load_json(root / "mappings.json"),
            meta=load_json(root / "meta.json"),
        )

    def seen_sets(self, include_valid: bool = False) -> list[set[int]]:
        seen = [set(self.binary_matrix[user].indices.tolist()) for user in range(int(self.meta["num_users"]))]
        if include_valid and not self.valid.empty:
            for row in self.valid.itertuples(index=False):
                seen[int(row.user_idx)].add(int(row.item_idx))
        return seen
