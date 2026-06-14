from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.processed import ProcessedData
from src.inference.candidates import CandidateGenerator
from src.models.als import ALSRecall
from src.models.item2vec import Item2VecRecall
from src.models.lightgcn import LightGCNRecall
from src.models.ranker import RankerCheckpoint, RankerFeatureStore
from src.rules.sequence_gap import SequenceGapRecommender
from src.utils.config import AttrDict, project_path
from src.utils.math import minmax_scale
from src.utils.seed import resolve_device


class HybridRecommender:
    def __init__(self, cfg: AttrDict) -> None:
        self.cfg = cfg
        self.data = ProcessedData.load(project_path(cfg, cfg.paths.processed_dir))
        self.checkpoint_dir = project_path(cfg, cfg.paths.checkpoint_dir)
        self.device = resolve_device(str(cfg.project.device))
        self.als = ALSRecall.load(self.checkpoint_dir / "als.npz") if (self.checkpoint_dir / "als.npz").exists() else None
        self.lightgcn = LightGCNRecall.load(self.checkpoint_dir / "lightgcn.npz") if (self.checkpoint_dir / "lightgcn.npz").exists() else None
        self.item2vec = Item2VecRecall.load(self.checkpoint_dir / "item2vec.npz") if (self.checkpoint_dir / "item2vec.npz").exists() else None
        self.candidates = CandidateGenerator(self.als, self.lightgcn, self.item2vec, self.data.popularity)
        self.feature_store = RankerFeatureStore(
            self.data.user_features, self.data.item_features, self.data.popularity,
            self.data.renew_user, self.data.renew_item,
            self.als, self.lightgcn, self.item2vec,
        )
        ranker_path = self.checkpoint_dir / "ranker.pt"
        self.ranker = None
        if ranker_path.exists():
            checkpoint = RankerCheckpoint.load(ranker_path, map_location=self.device)
            self.ranker = checkpoint.build_model(self.device)
            self.ranker.eval()

        self.rule_model = SequenceGapRecommender(
            self.data.books,
            int(cfg.rule.max_id_gap),
            int(cfg.rule.min_sequence_items),
            int(cfg.rule.max_candidates_per_user),
            dict(cfg.rule.weights),
        ) if bool(cfg.rule.enabled) else None
        self.user_to_idx = {str(k): int(v) for k, v in self.data.mappings["user_to_idx"].items()}
        self.idx_to_item = self.data.mappings["idx_to_item"]
        self.train_seen = self.data.seen_sets(include_valid=False)
        self.test_seen = self.data.seen_sets(include_valid=True)
        self.book_lookup = self.data.books.set_index("item_idx")

    def _history(self, user_idx: int, include_valid: bool) -> pd.DataFrame:
        history = self.data.train[self.data.train["user_idx"] == user_idx]
        if include_valid and not self.data.valid.empty:
            history = pd.concat([
                history,
                self.data.valid[self.data.valid["user_idx"] == user_idx],
            ], ignore_index=True)
        return history

    def _deep_scores(self, user_idx: int, items: np.ndarray) -> np.ndarray:
        users = np.full(len(items), user_idx, dtype=np.int64)
        if self.ranker is not None:
            with torch.no_grad():
                batch = self.feature_store.torch_batch(users, items, self.device)
                return self.ranker(*batch).cpu().numpy().astype(np.float32)
        _, _, numeric = self.feature_store.numpy_batch(users, items)
        # 无精排模型时，对三路召回分数与流行度做稳健线性回退。
        return (numeric[:, 0] + numeric[:, 1] + numeric[:, 2] + 0.2 * numeric[:, 3]).astype(np.float32)

    def recommend_by_index(
        self,
        user_idx: int,
        topk: int = 10,
        include_valid_history: bool = False,
    ) -> pd.DataFrame:
        seen = self.test_seen[user_idx] if include_valid_history else self.train_seen[user_idx]
        history = self._history(user_idx, include_valid_history)
        rules = self.rule_model.recommend(history) if self.rule_model is not None else []
        rule_map = {candidate.item_idx: candidate for candidate in rules if candidate.item_idx not in seen}

        deep_items = self.candidates.generate(
            user_idx,
            seen,
            int(self.cfg.recall.candidate_topk_each),
            int(self.cfg.recall.final_candidate_size),
        )
        item_set = list(dict.fromkeys(deep_items + list(rule_map.keys())))
        if not item_set:
            return pd.DataFrame(columns=["rank", "book_id", "title", "final_score"])
        items = np.asarray(item_set, dtype=np.int64)
        deep_raw = self._deep_scores(user_idx, items)
        deep_scaled = minmax_scale(deep_raw)

        rows: list[dict] = []
        for item, deep_score in zip(items, deep_scaled):
            rule_candidate = rule_map.get(int(item))
            confidence = float(rule_candidate.confidence) if rule_candidate else 0.0
            if confidence >= float(self.cfg.fusion.high_confidence):
                rule_weight = float(self.cfg.fusion.high_rule_weight)
            elif confidence >= float(self.cfg.fusion.medium_confidence):
                rule_weight = float(self.cfg.fusion.medium_rule_weight)
            elif confidence >= float(self.cfg.fusion.low_confidence):
                rule_weight = float(self.cfg.fusion.low_rule_weight)
            else:
                rule_weight = 0.0
            deep_weight = 1.0 - rule_weight
            final_score = rule_weight * confidence + deep_weight * float(deep_score)
            book = self.book_lookup.loc[int(item)]
            rows.append({
                "item_idx": int(item),
                "book_id": str(book["book_id"]),
                "title": str(book.get("title", "")),
                "author": str(book.get("author", "")),
                "publisher": str(book.get("publisher", "")),
                "category1": str(book.get("category1", "")),
                "category2": str(book.get("category2", "")),
                "deep_score": float(deep_score),
                "rule_confidence": confidence,
                "rule_weight": rule_weight,
                "final_score": float(final_score),
                "reason": rule_candidate.reason if rule_candidate else "ALS/LightGCN/Item2Vec 多路召回后由神经网络精排",
            })
        result = pd.DataFrame(rows).sort_values("final_score", ascending=False).head(topk).reset_index(drop=True)
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        return result

    def recommend(self, user_id: str, topk: int = 10) -> pd.DataFrame:
        key = str(user_id)
        if key not in self.user_to_idx:
            raise KeyError(f"用户 {user_id!r} 不在训练数据中，当前实现仅支持已知用户推荐。")
        return self.recommend_by_index(self.user_to_idx[key], topk=topk)
