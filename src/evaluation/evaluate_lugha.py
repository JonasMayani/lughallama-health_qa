from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from loguru import logger
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF

from lughallama_health_qa.config import ensure_project_dirs, load_config


def setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


class IntlTokenizer:
    """Unicode-aware tokenizer for ROUGE on Amharic, Akan, Swahili, and Luganda."""

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        text = unicodedata.normalize("NFKC", text.lower())
        return re.findall(r"\w+", text, flags=re.UNICODE)


def score_rows(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"prediction", "output", "subset"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Predictions file missing required columns: {missing}")

    rouge_types = cfg["evaluation"].get("rouge_types", ["rouge1", "rougeL"])
    scorer = rouge_scorer.RougeScorer(rouge_types, tokenizer=IntlTokenizer(), use_stemmer=False)
    chrf_metric = CHRF(word_order=2)

    scored_rows = []
    for _, row in df.iterrows():
        pred = str(row["prediction"]).strip()
        ref = str(row["output"]).strip()
        rouge = scorer.score(ref, pred)
        chrf = chrf_metric.sentence_score(pred, [ref]).score / 100.0

        scored = row.to_dict()
        for key in rouge_types:
            scored[f"{key}_f1"] = rouge[key].fmeasure
            scored[f"{key}_precision"] = rouge[key].precision
            scored[f"{key}_recall"] = rouge[key].recall
        scored["chrf"] = chrf
        scored_rows.append(scored)

    scored_df = pd.DataFrame(scored_rows)
    weights = cfg["evaluation"].get("metric_weights", {})
    if weights:
        composite = 0.0
        for metric_name, weight in weights.items():
            if metric_name in scored_df.columns:
                composite += scored_df[metric_name] * float(weight)
        scored_df["composite_score"] = composite

    numeric_cols = [
        col for col in scored_df.columns
        if col.endswith("_f1") or col.endswith("_precision") or col.endswith("_recall") or col in {"chrf", "composite_score"}
    ]

    overall = {col: float(scored_df[col].mean()) for col in numeric_cols}
    by_subset = (
        scored_df.groupby("subset")[numeric_cols]
        .mean()
        .sort_index()
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "rows": int(len(scored_df)),
        "overall": overall,
        "by_subset": by_subset,
    }
    return scored_df, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_lugha.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-scored-csv", default=None)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)

    predictions_path = Path(args.predictions)
    df = pd.read_csv(predictions_path).dropna(subset=["prediction", "output"])
    scored_df, summary = score_rows(df, cfg)

    output_json = Path(args.output_json) if args.output_json else Path(cfg["paths"]["reports"]) / "val_metrics.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_scored_csv:
        scored_path = Path(args.output_scored_csv)
        scored_path.parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(scored_path, index=False)

    logger.info(json.dumps(summary["overall"], indent=2))
    logger.success(f"Metrics saved to {output_json}")


if __name__ == "__main__":
    main()
