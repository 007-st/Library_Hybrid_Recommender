from __future__ import annotations

from pathlib import Path

from src.data.processed import ProcessedData
from src.models.als import ALSRecall
from src.models.item2vec import Item2VecRecall
from src.models.lightgcn import LightGCNRecall
from src.utils.config import AttrDict, project_path
from src.utils.logging import get_logger
from src.utils.seed import resolve_device

LOGGER = get_logger(__name__)


def train_recall_models(cfg: AttrDict) -> None:
    data = ProcessedData.load(project_path(cfg, cfg.paths.processed_dir))
    checkpoint_dir = project_path(cfg, cfg.paths.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(cfg.project.device))

    if bool(cfg.als.enabled):
        LOGGER.info("训练 ALS，backend=%s，device=%s", cfg.als.backend, device)
        if str(cfg.als.backend).lower() == "implicit":
            als = ALSRecall.fit_implicit(
                data.train_matrix,
                int(cfg.als.factors),
                int(cfg.als.iterations),
                float(cfg.als.regularization),
                float(cfg.als.confidence_alpha),
                int(cfg.project.seed),
            )
        else:
            als_device = device if bool(cfg.als.use_gpu) else resolve_device("cpu")
            als = ALSRecall.fit_torch(
                data.train_matrix,
                int(cfg.als.factors),
                int(cfg.als.iterations),
                float(cfg.als.regularization),
                float(cfg.als.confidence_alpha),
                int(cfg.als.batch_rows),
                als_device,
                int(cfg.project.seed),
            )
        als.save(checkpoint_dir / "als.npz")
        LOGGER.info("ALS 已保存：%s", checkpoint_dir / "als.npz")

    if bool(cfg.lightgcn.enabled):
        LOGGER.info("训练 LightGCN，device=%s", device)
        lightgcn = LightGCNRecall.fit(
            data.binary_matrix,
            int(cfg.lightgcn.embedding_size),
            int(cfg.lightgcn.layer_count),
            int(cfg.lightgcn.epochs),
            int(cfg.lightgcn.samples_per_epoch),
            float(cfg.lightgcn.learning_rate),
            float(cfg.lightgcn.weight_decay),
            int(cfg.lightgcn.early_stop),
            device,
            int(cfg.project.seed),
        )
        lightgcn.save(checkpoint_dir / "lightgcn.npz")
        LOGGER.info("LightGCN 已保存：%s", checkpoint_dir / "lightgcn.npz")

    if bool(cfg.item2vec.enabled):
        LOGGER.info("训练 Item2Vec")
        item2vec = Item2VecRecall.fit(
            data.train,
            int(data.meta["num_users"]),
            int(data.meta["num_items"]),
            int(cfg.item2vec.embedding_size),
            int(cfg.item2vec.context_window),
            int(cfg.item2vec.epochs),
            int(cfg.item2vec.negative),
            int(cfg.item2vec.workers),
            int(cfg.item2vec.recent_history_length),
            int(cfg.project.seed),
        )
        item2vec.save(checkpoint_dir / "item2vec.npz")
        LOGGER.info("Item2Vec 已保存：%s", checkpoint_dir / "item2vec.npz")
