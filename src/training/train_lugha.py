from __future__ import annotations

import argparse
import inspect
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", message=r".*length_penalty.*", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from lughallama_health_qa.config import (
    ensure_project_dirs,
    load_config,
    model_output_dir,
    torch_dtype_from_config,
    workspace_path,
)
from lughallama_health_qa.data import load_qa_split, normalize_for_dataset
from lughallama_health_qa.prompts import build_prompt, language_for_subset


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )


def seed_everything(cfg: dict[str, Any]) -> None:
    seed = int(cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if cfg.get("deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if cfg["training"].get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def configure_cuda_device() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def tokenize_example(example: dict[str, Any], tokenizer: AutoTokenizer, cfg: dict[str, Any]) -> dict[str, list[int]]:
    language = language_for_subset(example["subset"], cfg)
    prompt_text = build_prompt(example["input"], language, cfg["prompt"])

    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    max_prompt = int(cfg["model"]["max_prompt_length"])
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[:max_prompt]

    answer_text = " " + str(example["output"]).strip()
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    answer_ids.append(tokenizer.eos_token_id)

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    max_len = int(cfg["model"]["max_seq_length"])
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def build_datasets(cfg: dict[str, Any], tokenizer: AutoTokenizer) -> tuple[Dataset, Dataset]:
    ws = workspace_path(cfg)
    train_path = ws / cfg["dataset"]["train_file"]
    val_path = ws / cfg["dataset"]["val_file"]

    train_df = normalize_for_dataset(load_qa_split(train_path, cfg, require_output=True), cfg, require_output=True)
    val_df = normalize_for_dataset(load_qa_split(val_path, cfg, require_output=True), cfg, require_output=True)

    logger.info(f"Train rows: {len(train_df):,} | Validation rows: {len(val_df):,}")

    n_eval = cfg["training"].get("eval_max_samples")
    if n_eval and len(val_df) > int(n_eval):
        n_langs = max(1, val_df["subset"].nunique())
        per_lang = max(1, int(n_eval) // n_langs)
        pieces = [
            group.sample(min(len(group), per_lang), random_state=int(cfg["seed"]))
            for _, group in val_df.groupby("subset")
        ]
        val_df = pd.concat(pieces, ignore_index=True)
        logger.info(f"Validation subsample: {len(val_df):,} rows, stratified by subset")

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)

    def tok_fn(example: dict[str, Any]) -> dict[str, list[int]]:
        return tokenize_example(example, tokenizer, cfg)

    train_ds = train_ds.map(tok_fn, remove_columns=train_ds.column_names, desc="Tokenize train")
    val_ds = val_ds.map(tok_fn, remove_columns=val_ds.column_names, desc="Tokenize validation")
    return train_ds, val_ds


def load_model_and_tokenizer(cfg: dict[str, Any]) -> tuple[torch.nn.Module, AutoTokenizer]:
    base_model = cfg["model"]["base_model"]
    dtype = torch_dtype_from_config(cfg)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if cfg["model"].get("load_in_4bit", False):
        raise ValueError("This project is configured for bf16 training. Set model.load_in_4bit to false.")

    logger.info(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": bool(cfg["model"].get("low_cpu_mem_usage", True)),
        "trust_remote_code": bool(cfg["model"].get("trust_remote_code", False)),
    }
    if world_size <= 1:
        model_kwargs["device_map"] = {"": 0}
    attn_impl = cfg["model"].get("attn_implementation")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    logger.info(f"Loading base model in {dtype}: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model.config.use_cache = False

    if cfg["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if cfg["lora"].get("enabled", True):
        lora_config = LoraConfig(
            r=int(cfg["lora"]["r"]),
            lora_alpha=int(cfg["lora"]["lora_alpha"]),
            lora_dropout=float(cfg["lora"]["lora_dropout"]),
            target_modules=cfg["lora"]["target_modules"],
            bias=cfg["lora"]["bias"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        logger.warning("LoRA is disabled. Full bf16 fine-tuning may exceed one A100 80GB depending on sequence length.")

    return model, tokenizer


class NanGuardTrainer(Trainer):
    def training_step(self, model, inputs, *args, **kwargs):
        try:
            loss = super().training_step(model, inputs, *args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM in training_step; clearing cache and skipping batch")
            torch.cuda.empty_cache()
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
            logger.warning(f"NaN/Inf loss ({loss}); zeroing step")
            return torch.zeros_like(loss)
        return loss


def build_training_args(cfg: dict[str, Any], output_dir: Path) -> TrainingArguments:
    training = cfg["training"]
    optim = cfg["optimiser"]
    report_to = "none" if cfg.get("tracking", {}).get("backend", "none") == "none" else cfg["tracking"]["backend"]

    args: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": training["epochs"],
        "per_device_train_batch_size": training["per_device_train_batch"],
        "per_device_eval_batch_size": training["per_device_eval_batch"],
        "gradient_accumulation_steps": training["gradient_accumulation"],
        "learning_rate": float(optim["lr"]),
        "weight_decay": float(optim["weight_decay"]),
        "lr_scheduler_type": optim["lr_scheduler"],
        "warmup_ratio": float(optim["warmup_ratio"]),
        "max_grad_norm": float(optim["max_grad_norm"]),
        "optim": optim["optim"],
        "bf16": bool(training.get("bf16", True)),
        "fp16": bool(training.get("fp16", False)),
        "tf32": bool(training.get("tf32", True)),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_num_workers": int(training["dataloader_num_workers"]),
        "logging_steps": int(training["logging_steps"]),
        "save_strategy": training["save_strategy"],
        "save_steps": int(training["save_steps"]),
        "save_total_limit": int(training["save_total_limit"]),
        "load_best_model_at_end": bool(training["load_best_model_at_end"]),
        "metric_for_best_model": training["metric_for_best_model"],
        "greater_is_better": bool(training["greater_is_better"]),
        "report_to": report_to,
        "run_name": cfg.get("tracking", {}).get("run_name"),
        "seed": int(cfg["seed"]),
        "remove_unused_columns": False,
    }

    signature_keys = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature_keys:
        args["eval_strategy"] = training["eval_strategy"]
    else:
        args["evaluation_strategy"] = training["eval_strategy"]

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        args["ddp_find_unused_parameters"] = False

    return TrainingArguments(**args)


def find_resume_checkpoint(output_dir: Path, cfg: dict[str, Any]) -> bool | str | None:
    resume = cfg["training"].get("resume_from_checkpoint", "auto")
    if resume == "auto":
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        if checkpoints:
            logger.info(f"Resuming from latest checkpoint under {output_dir}")
            return True
        return None
    if resume:
        return str(resume)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_lugha.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Lugha-Llama fine-tuning.")

    local_rank = configure_cuda_device()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    free, total = torch.cuda.mem_get_info()
    logger.info(
        f"GPU rank {local_rank}/{world_size}: {torch.cuda.get_device_name(local_rank)} | "
        f"Free memory: {free / 1e9:.1f}/{total / 1e9:.1f} GB"
    )

    seed_everything(cfg)
    model, tokenizer = load_model_and_tokenizer(cfg)
    train_ds, val_ds = build_datasets(cfg, tokenizer)

    output_dir = model_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    callbacks = []
    patience = cfg["training"].get("early_stopping_patience")
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(patience)))

    trainer = NanGuardTrainer(
        model=model,
        args=build_training_args(cfg, output_dir),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    logger.info("Starting bf16 Lugha-Llama fine-tuning")
    trainer.train(resume_from_checkpoint=find_resume_checkpoint(output_dir, cfg))

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.success(f"Best adapter saved to {best_dir}")


if __name__ == "__main__":
    main()
