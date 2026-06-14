from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from src.evaluation.metrics import aggregate_metrics, user_metrics
from src.inference.hybrid import HybridRecommender
from src.utils.config import AttrDict, project_path
from src.utils.io import save_json
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def evaluate(cfg: AttrDict, split: str = "valid") -> dict:
    if split not in {"valid", "test"}:
        raise ValueError("split 必须为 valid 或 test")
    system = HybridRecommender(cfg)
    frame = system.data.valid if split == "valid" else system.data.test
    if frame.empty:
        raise RuntimeError(f"{split} 集为空，请检查 preprocess.split_strategy。")
    ground_truth = {
        int(user): set(group["item_idx"].astype(int).tolist())
        for user, group in frame.groupby("user_idx")
    }
    topks = sorted({int(k) for k in cfg.evaluation.topk})
    max_k = max(topks)
    per_k: dict[int, list[dict[str, float]]] = {k: [] for k in topks}
    include_valid = split == "test"
    for user, truth in tqdm(ground_truth.items(), desc=f"Evaluate-{split}"):
        recommendations = system.recommend_by_index(
            user, topk=max_k, include_valid_history=include_valid
        )["item_idx"].astype(int).tolist()
        for k in topks:
            per_k[k].append(user_metrics(recommendations, truth, k))

    report = {
        "split": split,
        "num_evaluated_users": len(ground_truth),
        "metrics": {f"@{k}": aggregate_metrics(per_k[k]) for k in topks},
    }
    report_path = project_path(cfg, cfg.paths.report_dir) / f"metrics_{split}.json"
    save_json(report, report_path)
    LOGGER.info("评估结果：%s", report)
    LOGGER.info("报告已保存：%s", report_path)
    return report
