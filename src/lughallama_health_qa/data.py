from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from lughallama_health_qa.cleaning import clean_qa_dataframe, read_qa_csv


def resolve_dataset_path(cfg: dict[str, Any], split: str, explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path)

    split_to_key = {
        "train": "train_file",
        "val": "val_file",
        "validation": "val_file",
        "test": "test_file",
    }
    if split not in split_to_key:
        raise ValueError(f"Unknown split: {split}")

    return Path(cfg["paths"]["workspace"]) / cfg["dataset"][split_to_key[split]]


def load_qa_split(
    path: str | Path,
    cfg: dict[str, Any],
    *,
    require_output: bool,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = read_qa_csv(path)
    cols = cfg["dataset"]
    id_col = cols.get("id_col", "ID")
    input_col = cols.get("input_col", "input")
    output_col = cols.get("output_col", "output")
    subset_col = cols.get("subset_col", "subset")

    required = [input_col, subset_col]
    if require_output:
        required.append(output_col)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    keep_cols = [col for col in [id_col, input_col, output_col, subset_col] if col in df.columns]
    df = df[keep_cols].copy()

    if cfg.get("cleaning", {}).get("enabled", True) and cfg.get("cleaning", {}).get("apply_on_load", True):
        split = "train" if require_output and "train" in path.name.lower() else "val" if require_output else "test"
        df, _ = clean_qa_dataframe(
            df,
            cfg,
            split=split,
            require_output=require_output,
            preserve_rows=not require_output,
        )
    else:
        for col in [input_col, subset_col]:
            df[col] = df[col].astype(str).str.strip()
        df = df[(df[input_col] != "") & (df[subset_col] != "")]

        if require_output:
            df = df.dropna(subset=[output_col])
            df[output_col] = df[output_col].astype(str).str.strip()
            df = df[df[output_col] != ""]

    return df.reset_index(drop=True)


def normalize_for_dataset(df: pd.DataFrame, cfg: dict[str, Any], *, require_output: bool) -> pd.DataFrame:
    cols = cfg["dataset"]
    rename = {
        cols.get("input_col", "input"): "input",
        cols.get("subset_col", "subset"): "subset",
    }
    id_col = cols.get("id_col", "ID")
    output_col = cols.get("output_col", "output")

    if id_col in df.columns:
        rename[id_col] = "ID"
    if output_col in df.columns:
        rename[output_col] = "output"

    out = df.rename(columns=rename)
    required = ["input", "subset"]
    if require_output:
        required.append("output")
    return out[required]
