"""
src/training/train_lugha.py — Lugha-Llama-8B QLoRA fine-tuning (causal LM).

Lugha-Llama is Llama-3.1-8B continued-pretrained on African languages. It is a
BASE model (NOT instruction-tuned), so there is NO chat template. We build a
plain instruction prompt ending in an explicit "Answer:" marker, append the
answer, and mask the loss over everything up to and including the marker — so
the model learns only to generate the answer continuation.

Differences from train_aya.py:
  • No apply_chat_template — plain string prompt from cfg["prompt"]["template"].
  • Prompt masking finds the boundary by tokenizing prompt and answer separately.

Usage:
    python src/training/train_lugha.py --config configs/config_lugha.yaml
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*length_penalty.*", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from loguru import logger
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType,
)
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments,
    EarlyStoppingCallback,
)


def setup_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Prompt construction (plain instruction; NO chat template) ──────────────

def build_prompt(question: str, language: str, prompt_cfg: dict) -> str:
    """Plain instruction prompt ending in 'Answer:'. The answer is appended
    during training and generated after it at inference."""
    return prompt_cfg["template"].format(
        question=str(question).strip(), language=language)


def tokenize_example(example, tokenizer, cfg):
    """Tokenize one row for causal-LM training with prompt masking.

    Builds:  <prompt ending in 'Answer:'><space+answer><eos>
    Labels:  [-100 over prompt] + [answer ids] + [eos]
    so loss is computed ONLY on the answer tokens.
    """
    prompt_cfg = cfg["prompt"]
    lang = cfg["language_names"].get(example["subset"], example["subset"])
    prompt_text = build_prompt(example["input"], lang, prompt_cfg)

    # Tokenize the prompt (with BOS, no EOS). Base Llama has a BOS token.
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    max_prompt = cfg["model"]["max_prompt_length"]
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[:max_prompt]

    # Answer: prepend a space so the first answer token tokenizes naturally
    # after "Answer:", then add EOS so the model learns to stop.
    answer = " " + str(example["output"]).strip()
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids   # mask prompt, train on answer

    max_len = cfg["model"]["max_seq_length"]
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]

    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}


# ─── Data ────────────────────────────────────────────────────────────────────

def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["input", "output", "subset"]).copy()
    for c in ["input", "output", "subset"]:
        df[c] = df[c].astype(str).str.strip()
    df = df[(df["input"] != "") & (df["output"] != "") & (df["subset"] != "")]
    return df.reset_index(drop=True)


def build_datasets(cfg: dict, tokenizer):
    ws = Path(cfg["paths"]["workspace"])
    train_df = load_split(ws / cfg["dataset"]["train_file"])
    val_df = load_split(ws / cfg["dataset"]["val_file"])
    logger.info(f"Train: {len(train_df):,}  Val: {len(val_df):,}")

    # Stratified val subsample (explicit loop avoids the pandas groupby-apply
    # column bug that bit the Aya run).
    n_eval = cfg["training"].get("eval_max_samples", 600)
    if len(val_df) > n_eval:
        n_langs = val_df["subset"].nunique()
        per_lang = max(1, n_eval // n_langs)
        pieces = []
        for _, grp in val_df.groupby("subset"):
            pieces.append(grp.sample(min(len(grp), per_lang), random_state=cfg["seed"]))
        val_df = pd.concat(pieces, ignore_index=True)
        logger.info(f"Eval subsample: {len(val_df):,} (stratified)")

    train_ds = Dataset.from_pandas(train_df[["input", "output", "subset"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["input", "output", "subset"]], preserve_index=False)

    tok_fn = lambda ex: tokenize_example(ex, tokenizer, cfg)
    train_ds = train_ds.map(tok_fn, remove_columns=train_ds.column_names, desc="Tokenize train")
    val_ds = val_ds.map(tok_fn, remove_columns=val_ds.column_names, desc="Tokenize val")
    return train_ds, val_ds


# ─── Model (4-bit QLoRA) ────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    base = cfg["model"]["base_model"]
    logger.info(f"Loading tokenizer: {base}")
    tokenizer = AutoTokenizer.from_pretrained(base)
    # Llama base models often have no pad token — use EOS.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # right padding for training

    compute_dtype = getattr(torch, cfg["model"]["bnb_4bit_compute_dtype"])
    quant = BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg["model"]["bnb_4bit_use_double_quant"],
    )
    logger.info(f"Loading base in 4-bit NF4 (compute={compute_dtype}): {base}")
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=quant, torch_dtype=compute_dtype, device_map={"": 0})
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"])

    lora_config = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"]["bias"], task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if cfg["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model, tokenizer


# ─── NaN/OOM-guarded Trainer ─────────────────────────────────────────────────

class NanGuardTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM in training_step; clearing cache and skipping batch")
            torch.cuda.empty_cache()
            return torch.tensor(0.0, device=model.device, requires_grad=True)
        if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
            logger.warning(f"NaN/Inf loss ({loss}); zeroing step")
            return torch.zeros_like(loss)
        return loss


def build_training_args(cfg: dict, output_dir: Path) -> TrainingArguments:
    t, o = cfg["training"], cfg["optimiser"]
    logging_steps = int(t.get("logging_steps", 20))
    eval_steps = int(t.get("eval_steps", 500))
    save_steps = int(t.get("save_steps", eval_steps))
    eval_strategy = str(t.get("eval_strategy", "steps"))
    save_strategy = str(t.get("save_strategy", eval_strategy))

    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=float(t["epochs"]),
        per_device_train_batch_size=int(t["per_device_train_batch"]),
        per_device_eval_batch_size=int(t["per_device_eval_batch"]),
        gradient_accumulation_steps=int(t["gradient_accumulation"]),
        learning_rate=float(o["lr"]),
        weight_decay=float(o["weight_decay"]),
        lr_scheduler_type=o["lr_scheduler"],
        warmup_ratio=float(o.get("warmup_ratio", 0.0)),
        max_grad_norm=float(o["max_grad_norm"]),
        optim=o["optim"],
        bf16=bool(t.get("bf16", False)),
        fp16=bool(t.get("fp16", False)),
        tf32=bool(t.get("tf32", False)),
        gradient_checkpointing=bool(t["gradient_checkpointing"]),
        dataloader_num_workers=int(t["dataloader_num_workers"]),
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=int(t["save_total_limit"]),
        load_best_model_at_end=bool(t["load_best_model_at_end"]),
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=bool(t["greater_is_better"]),
        report_to="none",
        seed=int(cfg["seed"]),
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_strategy
    else:
        kwargs["evaluation_strategy"] = eval_strategy

    if "warmup_steps" in o and o["warmup_steps"] is not None:
        kwargs["warmup_steps"] = int(o["warmup_steps"])
        kwargs.pop("warmup_ratio", None)

    kwargs = {k: v for k, v in kwargs.items() if k in params}
    args = TrainingArguments(**kwargs)
    logger.info(
        "TRAINER CADENCE | "
        f"logging_steps={args.logging_steps} | "
        f"eval_strategy={getattr(args, 'eval_strategy', getattr(args, 'evaluation_strategy', None))} | "
        f"eval_steps={args.eval_steps} | "
        f"save_strategy={args.save_strategy} | "
        f"save_steps={args.save_steps}"
    )
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_lugha.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    logger.info(f"Config file: {args.config}")

    if bool(cfg["training"].get("tf32", False)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    free, total = torch.cuda.mem_get_info()
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}  Free: {free/1e9:.1f}/{total/1e9:.1f} GB")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    model, tokenizer = load_model_and_tokenizer(cfg)
    train_ds, val_ds = build_datasets(cfg, tokenizer)

    output_name = cfg["model"]["base_model"].replace("/", "_")
    suffix = cfg["training"].get("output_dir_suffix", "")
    if suffix:
        output_name = f"{output_name}_{suffix}"
    output_dir = Path(cfg["paths"]["models"]) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt")

    callbacks = []
    if cfg["training"].get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=cfg["training"]["early_stopping_patience"]))

    trainer = NanGuardTrainer(
        model=model, args=build_training_args(cfg, output_dir),
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, callbacks=callbacks)

    resume = cfg["training"].get("resume_from_checkpoint", "auto")
    resume_arg = None
    if resume == "auto":
        if list(output_dir.glob("checkpoint-*")):
            resume_arg = True
            logger.info(f"Resuming from latest checkpoint in {output_dir}")
    elif resume:
        resume_arg = resume

    logger.info("Starting training …")
    trainer.train(resume_from_checkpoint=resume_arg)

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.success(f"Best adapter saved → {best_dir}")


if __name__ == "__main__":
    main()
