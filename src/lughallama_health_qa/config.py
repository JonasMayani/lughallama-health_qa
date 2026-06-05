from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def torch_dtype_from_config(cfg: dict[str, Any]) -> torch.dtype:
    dtype_name = str(cfg["model"].get("torch_dtype", "bfloat16")).lower()
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_name not in aliases:
        raise ValueError(f"Unsupported model.torch_dtype: {dtype_name}")
    return aliases[dtype_name]


def workspace_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["workspace"])


def model_run_name(cfg: dict[str, Any]) -> str:
    name = cfg["model"]["base_model"].replace("/", "_")
    suffix = cfg["training"].get("output_dir_suffix", "")
    return f"{name}_{suffix}" if suffix else name


def model_output_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["models"]) / model_run_name(cfg)


def best_adapter_dir(cfg: dict[str, Any]) -> Path:
    return model_output_dir(cfg) / "best"


def ensure_project_dirs(cfg: dict[str, Any]) -> None:
    for key in ("data_raw", "data_cleaned", "data_augmented", "models", "submissions", "reports", "logs"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
