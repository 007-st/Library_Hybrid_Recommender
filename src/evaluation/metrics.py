from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def user_metrics(recommended: list[int], truth: set[int], k: int) -> dict[str, float]:
    ranked = recommended[:k]
    hits = [1 if item in truth else 0 for item in ranked]
    hit_count = sum(hits)
    precision = hit_count / k if k > 0 else 0.0
    recall = hit_count / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    reciprocal_rank = 0.0
    for index, hit in enumerate(hits):
        if hit:
            reciprocal_rank = 1.0 / (index + 1)
            break
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ndcg": ndcg,
        "hit_rate": 1.0 if hit_count > 0 else 0.0,
        "mrr": reciprocal_rank,
    }


def aggregate_metrics(per_user: list[dict[str, float]]) -> dict[str, float]:
    if not per_user:
        return {name: 0.0 for name in ("precision", "recall", "f1", "ndcg", "hit_rate", "mrr")}
    result = defaultdict(float)
    for metrics in per_user:
        for name, value in metrics.items():
            result[name] += value
    return {name: value / len(per_user) for name, value in result.items()}
