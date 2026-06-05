from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import torch
from loguru import logger
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lughallama_health_qa.config import best_adapter_dir, ensure_project_dirs, load_config, torch_dtype_from_config
from lughallama_health_qa.data import load_qa_split, resolve_dataset_path
from lughallama_health_qa.prompts import build_prompt, language_for_subset


def setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def resolve_checkpoint(cfg: dict[str, Any], checkpoint: str) -> Path:
    if checkpoint == "best":
        return best_adapter_dir(cfg)
    return Path(checkpoint)


def load_model_for_generation(cfg: dict[str, Any], checkpoint: Path) -> tuple[torch.nn.Module, AutoTokenizer]:
    dtype = torch_dtype_from_config(cfg)
    trust_remote_code = bool(cfg["model"].get("trust_remote_code", False))

    if checkpoint.exists() and (checkpoint / "adapter_config.json").exists():
        peft_cfg = PeftConfig.from_pretrained(str(checkpoint))
        base_model_name = peft_cfg.base_model_name_or_path or cfg["model"]["base_model"]
        tokenizer_source = str(checkpoint)
        logger.info(f"Loading PEFT adapter: {checkpoint}")
    else:
        base_model_name = cfg["model"]["base_model"]
        tokenizer_source = str(checkpoint) if checkpoint.exists() else base_model_name
        logger.warning(f"No adapter_config.json found at {checkpoint}; loading base model only")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": {"": 0},
        "low_cpu_mem_usage": bool(cfg["model"].get("low_cpu_mem_usage", True)),
        "trust_remote_code": trust_remote_code,
    }
    attn_impl = cfg["model"].get("attn_implementation")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    if checkpoint.exists() and (checkpoint / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, str(checkpoint), torch_dtype=dtype)
    else:
        model = base_model

    model.eval()
    return model, tokenizer


def generation_kwargs(cfg: dict[str, Any], subset: str) -> dict[str, Any]:
    defaults = dict(cfg["decoding"]["default"])
    per_language = cfg["decoding"].get("per_language", {}).get(subset, {})
    defaults.update(per_language)

    kwargs = {
        "max_new_tokens": int(defaults.get("max_new_tokens", cfg["model"]["max_new_tokens"])),
        "min_new_tokens": int(defaults.get("min_new_tokens", 0)),
        "do_sample": bool(defaults.get("do_sample", False)),
        "repetition_penalty": float(defaults.get("repetition_penalty", 1.0)),
        "no_repeat_ngram_size": int(defaults.get("no_repeat_ngram_size", 0)),
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = float(defaults.get("temperature", 0.7))
        kwargs["top_p"] = float(defaults.get("top_p", 0.9))
    return kwargs


def generate_one(model, tokenizer, prompt: str, kwargs: dict[str, Any]) -> str:
    kwargs = dict(kwargs)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(kwargs.pop("max_prompt_length")),
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            **kwargs,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )[0]

    new_tokens = output_ids[prompt_length:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    kwargs: dict[str, Any],
    max_prompt_length: int,
) -> list[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            **kwargs,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    return [
        tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip()
        for row in output_ids
    ]


def generate_prediction_frame(
    df: pd.DataFrame,
    model,
    tokenizer,
    cfg: dict[str, Any],
    *,
    split: str,
    batch_size: int = 4,
) -> pd.DataFrame:
    cols = cfg["dataset"]
    id_col = cols.get("id_col", "ID")
    input_col = cols.get("input_col", "input")
    output_col = cols.get("output_col", "output")
    subset_col = cols.get("subset_col", "subset")
    max_prompt_length = int(cfg["model"]["max_prompt_length"])

    work = df.copy().reset_index(drop=True)
    work["_subset"] = work[subset_col].astype(str)
    work["_language"] = [language_for_subset(subset, cfg) for subset in work["_subset"]]
    work["_prompt"] = [
        build_prompt(question, language, cfg["prompt"])
        for question, language in zip(work[input_col], work["_language"])
    ]

    predictions = [""] * len(work)
    per_language_cfg = cfg["decoding"].get("per_language", {})
    can_batch = batch_size > 1 and not per_language_cfg
    t0 = time.time()

    if can_batch:
        base_kwargs = generation_kwargs(cfg, "")
        for start in tqdm(range(0, len(work), batch_size), desc=f"Generate {split}"):
            batch = work.iloc[start:start + batch_size]
            prompts = batch["_prompt"].tolist()
            try:
                batch_preds = generate_batch(model, tokenizer, prompts, base_kwargs, max_prompt_length)
            except torch.cuda.OutOfMemoryError:
                logger.warning("CUDA OOM during batch generation; retrying this batch one row at a time")
                torch.cuda.empty_cache()
                batch_preds = [
                    generate_one(
                        model,
                        tokenizer,
                        prompt,
                        {**base_kwargs, "max_prompt_length": max_prompt_length},
                    )
                    for prompt in prompts
                ]
            predictions[start:start + len(batch_preds)] = batch_preds
    else:
        iterator = tqdm(work.iterrows(), total=len(work), desc=f"Generate {split}")
        for idx, row in iterator:
            kwargs = generation_kwargs(cfg, str(row["_subset"]))
            kwargs["max_prompt_length"] = max_prompt_length
            predictions[idx] = generate_one(model, tokenizer, row["_prompt"], kwargs)

    logger.info(f"{split} generation finished in {(time.time() - t0) / 60:.1f} min")

    rows = []
    for idx, row in work.iterrows():
        out = {
            "ID": row[id_col] if id_col in work.columns else idx,
            "subset": row["_subset"],
            "language": row["_language"],
            "input": row[input_col],
            "prediction": predictions[idx],
        }
        if output_col in work.columns:
            out["output"] = row[output_col]
        rows.append(out)

    return pd.DataFrame(rows)


def make_zindi_submission(predictions_df: pd.DataFrame, cfg: dict[str, Any], out_path: str | Path) -> pd.DataFrame:
    if "ID" not in predictions_df.columns or "prediction" not in predictions_df.columns:
        raise ValueError("Zindi submission requires prediction rows with ID and prediction columns.")

    target_columns = cfg.get("submission", {}).get(
        "target_columns",
        ["TargetRLF1", "TargetR1F1", "TargetLLM"],
    )
    submission = pd.DataFrame({"ID": predictions_df["ID"]})
    for column in target_columns:
        submission[column] = predictions_df["prediction"].fillna("").astype(str)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    logger.success(f"Zindi submission saved to {out_path} ({len(submission):,} rows)")
    return submission


def default_submission_path(cfg: dict[str, Any], tag: str | None = None) -> Path:
    if tag:
        return Path(cfg["paths"]["submissions"]) / f"submission_{tag}.csv"

    configured = cfg.get("submission", {}).get("default_file")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else Path(cfg["paths"]["workspace"]) / path

    default_tag = cfg.get("submission", {}).get("default_tag", "lugha")
    return Path(cfg["paths"]["submissions"]) / f"submission_{default_tag}.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_lugha.yaml")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--make-submission", action="store_true")
    parser.add_argument("--submission-file", default=None)
    parser.add_argument("--submission-tag", default=None)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for generation.")

    checkpoint = resolve_checkpoint(cfg, args.checkpoint)
    model, tokenizer = load_model_for_generation(cfg, checkpoint)

    input_path = resolve_dataset_path(cfg, args.split, args.input_file)
    require_output = args.split == "val"
    df = load_qa_split(input_path, cfg, require_output=require_output)
    if args.limit:
        df = df.head(args.limit).copy()

    predictions_df = generate_prediction_frame(
        df,
        model,
        tokenizer,
        cfg,
        split=args.split,
        batch_size=max(1, args.batch_size),
    )

    output_file = args.output_file
    if output_file is None:
        output_file = str(Path(cfg["paths"]["submissions"]) / f"{args.split}_predictions.csv")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    logger.success(f"Predictions saved to {output_path}")

    if args.make_submission or args.submission_file:
        if args.split != "test":
            logger.warning("Creating a Zindi submission from a non-test split.")
        submission_file = args.submission_file
        if submission_file is None:
            submission_file = default_submission_path(cfg, args.submission_tag)
        make_zindi_submission(predictions_df, cfg, submission_file)


if __name__ == "__main__":
    main()
