from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.data.processed import ProcessedData
from src.models.ranker import RankerCheckpoint
from src.utils.config import AttrDict, project_path


def export_int8_ranker(cfg: AttrDict) -> Path:
    """将精排 MLP 的 Linear 层动态量化为 INT8，并导出 TorchScript CPU 模型。"""
    checkpoint_dir = project_path(cfg, cfg.paths.checkpoint_dir)
    checkpoint = RankerCheckpoint.load(checkpoint_dir / "ranker.pt", map_location="cpu")
    model = checkpoint.build_model(torch.device("cpu")).eval()
    quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    data = ProcessedData.load(project_path(cfg, cfg.paths.processed_dir))
    example = (
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        torch.zeros((1, data.user_features.shape[1]), dtype=torch.float32),
        torch.zeros((1, data.item_features.shape[1]), dtype=torch.float32),
        torch.zeros((1, 6), dtype=torch.float32),
    )
    traced = torch.jit.trace(quantized, example)
    output = checkpoint_dir / "ranker_int8_torchscript.pt"
    traced.save(str(output))
    return output
