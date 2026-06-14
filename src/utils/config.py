from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class AttrDict(dict):
    """支持 cfg.section.key 访问方式的递归字典。"""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__


def _to_attr_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({k: _to_attr_dict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(v) for v in value]
    return value


def load_config(path: str | Path) -> AttrDict:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    cfg = _to_attr_dict(cfg)
    cfg._config_path = str(path)
    cfg._project_root = str(path.parent.parent)
    return cfg


def project_path(cfg: AttrDict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(cfg._project_root) / path


def ensure_project_dirs(cfg: AttrDict) -> None:
    for key in ("raw_dir", "processed_dir", "checkpoint_dir", "recommendation_dir", "report_dir"):
        project_path(cfg, cfg.paths[key]).mkdir(parents=True, exist_ok=True)
