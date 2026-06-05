from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT / "src", EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd
import torch
from loguru import logger

from evaluate_lugha import score_rows
from generate_lugha import (
    default_submission_path,
    generate_prediction_frame,
    load_model_for_generation,
    make_zindi_submission,
    resolve_checkpoint,
)
from lughallama_health_qa.config import ensure_project_dirs, load_config
from lughallama_health_qa.data import load_qa_split, resolve_dataset_path


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )


def adapter_eval_dir(cfg: dict[str, Any], checkpoint: Path) -> Path:
    if (checkpoint / "adapter_config.json").exists():
        return checkpoint.parent / "final_eval"
    return Path(cfg["paths"]["reports"]) / "final_eval"


def metrics_table(summary: dict[str, Any]) -> pd.DataFrame:
    rows = list(summary["by_subset"])
    overall = {"subset": "OVERALL", **summary["overall"]}
    rows.append(overall)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_lugha.yaml")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Debug cap applied to both val and test when non-zero.")
    parser.add_argument("--make-submission", action="store_true")
    parser.add_argument("--submission-tag", default=None)
    parser.add_argument("--submission-file", default=None)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation and test generation.")

    checkpoint = resolve_checkpoint(cfg, args.checkpoint)
    model, tokenizer = load_model_for_generation(cfg, checkpoint)
    out_dir = adapter_eval_dir(cfg, checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_path = resolve_dataset_path(cfg, "val")
    val_df = load_qa_split(val_path, cfg, require_output=True)
    if args.limit:
        val_df = val_df.head(args.limit).copy()
    logger.info(f"Generating validation predictions for {len(val_df):,} rows")
    val_predictions = generate_prediction_frame(
        val_df,
        model,
        tokenizer,
        cfg,
        split="val",
        batch_size=max(1, args.batch_size),
    )
    val_predictions.to_csv(out_dir / "val_predictions_full.csv", index=False)

    scored_df, summary = score_rows(val_predictions.dropna(subset=["prediction", "output"]), cfg)
    scored_df.to_csv(out_dir / "val_scored_predictions.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    table = metrics_table(summary)
    table.to_csv(out_dir / "metrics_by_subset.csv", index=False)

    print("\n" + "=" * 72)
    print("VALIDATION METRICS (Lugha-Llama-8B bf16 LoRA)")
    print("=" * 72)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    logger.success(f"Validation evaluation saved to {out_dir}")

    if args.make_submission:
        test_path = resolve_dataset_path(cfg, "test")
        test_df = load_qa_split(test_path, cfg, require_output=False)
        if args.limit:
            test_df = test_df.head(args.limit).copy()
        logger.info(f"Generating test predictions for {len(test_df):,} rows")
        test_predictions = generate_prediction_frame(
            test_df,
            model,
            tokenizer,
            cfg,
            split="test",
            batch_size=max(1, args.batch_size),
        )
        test_predictions.to_csv(out_dir / "test_predictions_full.csv", index=False)
        Path(cfg["paths"]["submissions"]).mkdir(parents=True, exist_ok=True)

        submission_file = args.submission_file
        if submission_file is None:
            submission_file = default_submission_path(cfg, args.submission_tag)
        make_zindi_submission(test_predictions, cfg, submission_file)


if __name__ == "__main__":
    main()
