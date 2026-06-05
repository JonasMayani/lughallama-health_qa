from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from loguru import logger

from lughallama_health_qa.cleaning import clean_qa_dataframe, read_qa_csv, write_clean_csv
from lughallama_health_qa.config import ensure_project_dirs, load_config
from lughallama_health_qa.data import resolve_dataset_path


def setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def resolve_output_path(cfg: dict[str, Any], split: str) -> Path:
    key = f"cleaned_{split}_file"
    configured = cfg.get("cleaning", {}).get(key)
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else Path(cfg["paths"]["workspace"]) / path
    return Path(cfg["paths"]["data_cleaned"]) / f"{split}_clean_lugha.csv"


def clean_split(cfg: dict[str, Any], split: str) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    require_output = split != "test"
    preserve_rows = split == "test"
    input_path = resolve_dataset_path(cfg, split)
    output_path = resolve_output_path(cfg, split)

    raw_df = read_qa_csv(input_path)
    clean_df, issues = clean_qa_dataframe(
        raw_df,
        cfg,
        split=split,
        require_output=require_output,
        preserve_rows=preserve_rows,
    )
    write_clean_csv(clean_df, output_path)

    summary = {
        "split": split,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "raw_rows": int(len(raw_df)),
        "clean_rows": int(len(clean_df)),
        "issues": int(len(issues)),
        "preserve_rows": preserve_rows,
    }
    logger.success(
        f"{split}: {summary['raw_rows']:,} raw rows -> {summary['clean_rows']:,} clean rows at {output_path}"
    )
    return output_path, issues, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_lugha.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)

    all_issues: list[dict[str, Any]] = []
    summaries = []
    for split in args.splits:
        _, issues, summary = clean_split(cfg, split)
        all_issues.extend(issues)
        summaries.append(summary)

    report_file = Path(cfg["cleaning"].get("report_file", "reports/cleaning_report.json"))
    if not report_file.is_absolute():
        report_file = Path(cfg["paths"]["workspace"]) / report_file
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps({"splits": summaries}, indent=2, ensure_ascii=False), encoding="utf-8")

    issues_file = Path(cfg["cleaning"].get("issues_file", "reports/cleaning_issues.csv"))
    if not issues_file.is_absolute():
        issues_file = Path(cfg["paths"]["workspace"]) / issues_file
    issues_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_issues).to_csv(issues_file, index=False, encoding="utf-8")

    logger.success(f"Cleaning report saved to {report_file}")
    logger.success(f"Cleaning issue log saved to {issues_file}")


if __name__ == "__main__":
    main()
