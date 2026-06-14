from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


_TRAILING_NUMBER = re.compile(r"(\d+)$")


def _numeric_book_id(book_id: object) -> int | None:
    match = _TRAILING_NUMBER.search(str(book_id).strip())
    return int(match.group(1)) if match else None


def _id_gap_score(avg_gap: float) -> float:
    if avg_gap <= 5:
        return 1.0
    if avg_gap <= 10:
        return 0.7
    if avg_gap <= 50:
        return 0.3
    return 0.0


def _density_score(density: float) -> float:
    if density >= 0.6:
        return 1.0
    if density >= 0.4:
        return 0.6
    if density >= 0.2:
        return 0.3
    return 0.0


def _temporal_score(months: float) -> float:
    if months <= 3:
        return 1.0
    if months <= 6:
        return 0.8
    if months <= 12:
        return 0.5
    if months <= 18:
        return 0.2
    return 0.0


def _significance_score(value: float) -> float:
    if value >= 0.3:
        return 1.0
    if value >= 0.2:
        return 0.7
    if value >= 0.1:
        return 0.4
    return 0.2


def _topic_score(value: float) -> float:
    if value >= 0.90:
        return 1.0
    if value >= 0.75:
        return 0.8
    if value >= 0.60:
        return 0.6
    if value >= 0.50:
        return 0.4
    return 0.2


@dataclass
class RuleCandidate:
    item_idx: int
    confidence: float
    reason: str


class SequenceGapRecommender:
    """按 PDF 中五维置信度实现的图书 ID 序列缺口推荐器。"""

    def __init__(
        self,
        books: pd.DataFrame,
        max_id_gap: int,
        min_sequence_items: int,
        max_candidates_per_user: int,
        weights: dict[str, float],
    ) -> None:
        self.books = books.copy()
        self.max_id_gap = int(max_id_gap)
        self.min_sequence_items = int(min_sequence_items)
        self.max_candidates_per_user = int(max_candidates_per_user)
        self.weights = weights
        self.books["numeric_id"] = self.books["book_id"].map(_numeric_book_id)
        self.number_to_items: dict[int, list[int]] = {}
        for row in self.books.dropna(subset=["numeric_id"]).itertuples(index=False):
            self.number_to_items.setdefault(int(row.numeric_id), []).append(int(row.item_idx))
        self.book_by_idx = self.books.set_index("item_idx")

    def recommend(self, user_history: pd.DataFrame) -> list[RuleCandidate]:
        if user_history.empty:
            return []
        history = user_history.merge(
            self.books[["item_idx", "numeric_id", "category1", "category2", "publisher"]],
            on="item_idx",
            how="left",
        ).dropna(subset=["numeric_id"])
        if len(history) < self.min_sequence_items:
            return []

        history = history.sort_values("numeric_id")
        observed = sorted(set(history["numeric_id"].astype(int).tolist()))
        clusters: list[list[int]] = []
        current: list[int] = []
        for value in observed:
            if not current or value - current[-1] <= self.max_id_gap:
                current.append(value)
            else:
                if len(current) >= self.min_sequence_items:
                    clusters.append(current)
                current = [value]
        if len(current) >= self.min_sequence_items:
            clusters.append(current)

        borrowed = set(user_history["item_idx"].astype(int).tolist())
        total_borrowed = max(len(user_history), 1)
        candidates: dict[int, RuleCandidate] = {}
        for cluster in clusters:
            start, end = min(cluster), max(cluster)
            if end - start > self.max_id_gap * max(len(cluster), 2):
                continue
            missing_numbers = [number for number in range(start, end + 1) if number not in set(cluster)]
            if not missing_numbers:
                continue

            cluster_history = history[history["numeric_id"].astype(int).isin(cluster)]
            categories = cluster_history["category1"].fillna("未知").astype(str)
            main_category = categories.mode().iloc[0] if not categories.mode().empty else "未知"
            topic_ratio = float((categories == main_category).mean())
            gaps = np.diff(cluster)
            avg_gap = float(gaps.mean()) if len(gaps) else 0.0
            sequence_length = max(end - start + 1, 1)
            density = len(cluster) / sequence_length
            dates = pd.to_datetime(cluster_history["borrow_time"], errors="coerce")
            span_months = float((dates.max() - dates.min()).days / 30.0) if dates.notna().any() else 999.0
            significance = len(cluster_history) / total_borrowed

            confidence = (
                self.weights["id_gap"] * _id_gap_score(avg_gap)
                + self.weights["completeness"] * _density_score(density)
                + self.weights["temporal"] * _temporal_score(span_months)
                + self.weights["significance"] * _significance_score(significance)
                + self.weights["topic"] * _topic_score(topic_ratio)
            )

            for number in missing_numbers:
                for item_idx in self.number_to_items.get(number, []):
                    if item_idx in borrowed:
                        continue
                    book = self.book_by_idx.loc[item_idx]
                    category_match = str(book.get("category1", "未知")) == main_category
                    adjusted = confidence if category_match else confidence * 0.70
                    reason = (
                        f"检测到编号 {start}-{end} 的借阅序列缺口 {number}；"
                        f"主题={main_category}，五维置信度={adjusted:.3f}"
                    )
                    old = candidates.get(item_idx)
                    if old is None or adjusted > old.confidence:
                        candidates[item_idx] = RuleCandidate(item_idx, float(adjusted), reason)

        return sorted(candidates.values(), key=lambda item: item.confidence, reverse=True)[
            : self.max_candidates_per_user
        ]
