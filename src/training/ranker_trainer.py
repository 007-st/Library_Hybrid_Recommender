from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import trange

from src.data.processed import ProcessedData
from src.inference.candidates import CandidateGenerator
from src.models.als import ALSRecall
from src.models.item2vec import Item2VecRecall
from src.models.lightgcn import LightGCNRecall
from src.models.ranker import NeuralRanker, RankerCheckpoint, RankerFeatureStore
from src.utils.config import AttrDict, project_path
from src.utils.logging import get_logger
from src.utils.seed import resolve_device

LOGGER = get_logger(__name__)


def _load_optional(path: Path, loader, enabled: bool):
    return loader(path) if enabled and path.exists() else None


def _sample_popularity_negative(
    user: int,
    seen: list[set[int]],
    popularity_cdf: np.ndarray,
    rng: np.random.Generator,
) -> int:
    for _ in range(100):
        item = int(np.searchsorted(popularity_cdf, rng.random(), side="right"))
        item = min(item, len(popularity_cdf) - 1)
        if item not in seen[user]:
            return item
    available = np.setdiff1d(np.arange(len(popularity_cdf)), np.fromiter(seen[user], dtype=np.int64))
    if len(available) == 0:
        raise RuntimeError(f"用户 {user} 已与所有物品交互，无法负采样。")
    return int(rng.choice(available))


def _build_hard_pools(
    generator: CandidateGenerator,
    seen: list[set[int]],
    num_users: int,
    topk_each: int,
    max_candidates: int,
) -> list[list[int]]:
    pools: list[list[int]] = []
    for user in range(num_users):
        pools.append(generator.generate(user, seen[user], topk_each, max_candidates))
    return pools


def train_ranker(cfg: AttrDict) -> None:
    if not bool(cfg.ranker.enabled):
        LOGGER.info("ranker.enabled=false，跳过精排训练。")
        return
    data = ProcessedData.load(project_path(cfg, cfg.paths.processed_dir))
    checkpoint_dir = project_path(cfg, cfg.paths.checkpoint_dir)
    device = resolve_device(str(cfg.project.device))
    als = _load_optional(checkpoint_dir / "als.npz", ALSRecall.load, bool(cfg.als.enabled))
    lightgcn = _load_optional(checkpoint_dir / "lightgcn.npz", LightGCNRecall.load, bool(cfg.lightgcn.enabled))
    item2vec = _load_optional(checkpoint_dir / "item2vec.npz", Item2VecRecall.load, bool(cfg.item2vec.enabled))

    feature_store = RankerFeatureStore(
        data.user_features, data.item_features, data.popularity,
        data.renew_user, data.renew_item, als, lightgcn, item2vec,
    )
    model_kwargs = {
        "num_users": int(data.meta["num_users"]),
        "num_items": int(data.meta["num_items"]),
        "user_embedding_dim": int(cfg.ranker.user_embedding_dim),
        "item_embedding_dim": int(cfg.ranker.item_embedding_dim),
        "user_feature_dim": int(data.user_features.shape[1]),
        "item_feature_dim": int(data.item_features.shape[1]),
        "numeric_feature_dim": feature_store.numeric_feature_dim,
        "hidden_dims": list(cfg.ranker.hidden_dims),
        "dropout": float(cfg.ranker.dropout),
    }
    model = NeuralRanker(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.ranker.learning_rate), weight_decay=float(cfg.ranker.weight_decay)
    )

    seen = data.seen_sets()
    generator = CandidateGenerator(als, lightgcn, item2vec, data.popularity)
    LOGGER.info("构建困难负样本池")
    hard_pools = _build_hard_pools(
        generator, seen, int(data.meta["num_users"]),
        int(cfg.recall.candidate_topk_each), int(cfg.recall.final_candidate_size),
    )

    users_all = data.train["user_idx"].to_numpy(np.int64)
    items_all = data.train["item_idx"].to_numpy(np.int64)
    popularity_probability = np.power(data.popularity + 1.0, 0.75)
    popularity_probability /= popularity_probability.sum()
    popularity_cdf = np.cumsum(popularity_probability)
    popularity_cdf[-1] = 1.0
    rng = np.random.default_rng(int(cfg.project.seed))

    # 固定验证对：真实验证物品 vs 流行度负样本。
    if not data.valid.empty:
        val_users = data.valid["user_idx"].to_numpy(np.int64)
        val_pos = data.valid["item_idx"].to_numpy(np.int64)
        val_neg = np.asarray([
            _sample_popularity_negative(int(u), seen, popularity_cdf, rng) for u in val_users
        ], dtype=np.int64)
    else:
        val_users = users_all[: min(2000, len(users_all))]
        val_pos = items_all[: len(val_users)]
        val_neg = np.asarray([
            _sample_popularity_negative(int(u), seen, popularity_cdf, rng) for u in val_users
        ], dtype=np.int64)

    best_val = float("inf")
    best_state = None
    patience = 0
    batch_size = int(cfg.ranker.batch_size)
    for epoch in trange(int(cfg.ranker.epochs), desc="Ranker"):
        model.train()
        sample_count = int(cfg.ranker.train_samples_per_epoch)
        chosen = rng.integers(0, len(users_all), size=sample_count)
        sampled_users = users_all[chosen]
        sampled_pos = items_all[chosen]
        sampled_neg = np.empty(sample_count, dtype=np.int64)
        use_hard = rng.random(sample_count) < float(cfg.ranker.hard_negative_ratio)
        for index, user in enumerate(sampled_users):
            pool = hard_pools[int(user)]
            if use_hard[index] and pool:
                sampled_neg[index] = int(rng.choice(pool))
            else:
                sampled_neg[index] = _sample_popularity_negative(
                    int(user), seen, popularity_cdf, rng
                )

        order = rng.permutation(sample_count)
        total_loss = 0.0
        total_rows = 0
        for start in range(0, sample_count, batch_size):
            batch_idx = order[start : start + batch_size]
            users = sampled_users[batch_idx]
            pos = sampled_pos[batch_idx]
            neg = sampled_neg[batch_idx]
            pos_batch = feature_store.torch_batch(users, pos, device)
            neg_batch = feature_store.torch_batch(users, neg, device)

            optimizer.zero_grad(set_to_none=True)
            pos_score = model(*pos_batch)
            neg_score = model(*neg_batch)
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_idx)
            total_rows += len(batch_idx)

        model.eval()
        with torch.no_grad():
            val_pos_score = model(*feature_store.torch_batch(val_users, val_pos, device))
            val_neg_score = model(*feature_store.torch_batch(val_users, val_neg, device))
            val_loss = float((-F.logsigmoid(val_pos_score - val_neg_score).mean()).cpu())
        LOGGER.info(
            "Ranker epoch=%d train_loss=%.6f val_pair_loss=%.6f",
            epoch + 1, total_loss / max(total_rows, 1), val_loss,
        )
        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= int(cfg.ranker.patience):
            LOGGER.info("Ranker early stopping at epoch %d", epoch + 1)
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    RankerCheckpoint(model_kwargs, best_state).save(checkpoint_dir / "ranker.pt")
    LOGGER.info("精排模型已保存：%s", checkpoint_dir / "ranker.pt")
