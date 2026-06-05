# LughaLlama Health QA Fine-Tuning

Training and evaluation pipeline for bf16 fine-tuning of `Lugha-Llama/Lugha-Llama-8B-wura_edu` on multilingual low-resource African language health QA.

The project is adapted from the provided Lugha-Llama QLoRA sample, but it intentionally does not use 4-bit quantisation or bitsandbytes. The base model is loaded directly in `torch.bfloat16` for an A100 80GB RunPod instance, then fine-tuned with LoRA adapters.

## Project Layout

```text
configs/config_lugha.yaml          Main training/evaluation config
src/data/clean_lugha_data.py       Cleans train/validation/test CSV files
src/training/train_lugha.py        bf16 LoRA trainer
src/evaluation/eval_lugha.py       Validation scoring plus test submission
src/evaluation/generate_lugha.py   Generates validation/test predictions
src/evaluation/evaluate_lugha.py   Computes ROUGE and chrF metrics
scripts/runpod_setup.sh            RunPod environment setup
scripts/clean_data.sh              Materializes cleaned CSV files and reports
scripts/clean_data_local.ps1       Windows local cleaning entrypoint
scripts/train_3xa40.sh             Distributed training on 3x A40 GPUs
scripts/train_and_evaluate.sh      Full training, validation scoring, and test submission
scripts/train_and_evaluate_3xa40.sh Full 3x A40 training plus submission
scripts/evaluate_and_submit.sh     Validation scoring plus Zindi submission
scripts/train.sh                   Training entrypoint
scripts/evaluate_val.sh            Validation generation + scoring
scripts/predict_test.sh            Test prediction entrypoint
```

## RunPod Setup

Use a RunPod PyTorch image with an A100 80GB GPU. Copy this project to:

```bash
/workspace/lughallama-health_qa
```

Then run:

```bash
cd /workspace/lughallama-health_qa
bash scripts/runpod_setup.sh
huggingface-cli login
```

The Lugha-Llama model is hosted on Hugging Face, so the pod needs internet access and a token with access to the model if Hugging Face requires it.

## Data

Place files to match `configs/config_lugha.yaml`:

```text
data/augmented/final_train_v2.csv
data/augmented/final_val_v2.csv
data/raw/Test.csv
```

Expected columns:

```text
ID,input,output,subset
```

`output` is required for train and validation rows. Test rows may omit `output`.

Configured language subsets:

```text
Eng_Uga, Aka_Gha, Eng_Gha, Eng_Eth, Lug_Uga, Eng_Ken, Swa_Ken, Amh_Eth
```

## Train

Cleaned text is applied automatically when training/evaluation loads data. To materialize cleaned CSVs and inspect the cleaning report first, run:

```bash
bash scripts/clean_data.sh
```

On Windows locally, use:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/clean_data_local.ps1
```

This writes:

```text
data/cleaned/train_clean_lugha.csv
data/cleaned/val_clean_lugha.csv
data/cleaned/test_clean_lugha.csv
reports/cleaning_report.json
reports/cleaning_issues.csv
```

```bash
bash scripts/train.sh
```

or:

```bash
python src/training/train_lugha.py --config configs/config_lugha.yaml
```

For a 3x A40 RunPod, use distributed training:

```bash
bash scripts/train_3xa40.sh
```

For a full 3x A40 train, validation, test, and submission run:

```bash
bash scripts/train_and_evaluate_3xa40.sh
```

The A40 config uses a conservative per-GPU batch size of 4 with gradient accumulation 8. If you see OOM, reduce `training.per_device_train_batch` to 2. If memory is comfortable, try 6 or 8.

The default config is tuned for a fast single A100 run:

```yaml
per_device_train_batch: 8
per_device_eval_batch: 8
gradient_accumulation: 4
eval_steps: 500
save_steps: 500
save_total_limit: 2
```

If your A100 is 40 GB and you see OOM, lower `per_device_train_batch` and `per_device_eval_batch` to `4`.

The best LoRA adapter and tokenizer are saved under:

```text
models/checkpoints/Lugha-Llama_Lugha-Llama-8B-wura_edu_lugha8b_bf16_lora_a100_fast_v1/best
```

## Train And Evaluate

```bash
bash scripts/train_and_evaluate.sh
```

This trains the adapter, scores the validation split, runs inference on the test split, and writes the Zindi submission CSV.

## Evaluate And Submit

```bash
bash scripts/evaluate_and_submit.sh
```

This loads the best trained adapter once, writes validation outputs under `models/checkpoints/.../final_eval`, and creates:

```text
submissions/submission_lugha8b_bf16_lora_a100_fast_v1.csv
```

## Evaluate Validation Split

```bash
bash scripts/evaluate_val.sh
```

This writes predictions to `submissions/val_predictions.csv` and metrics to `reports/val_metrics.json`.

## Predict Test Split

```bash
bash scripts/predict_test.sh
```

This writes:

```text
submissions/test_predictions.csv
submissions/submission_lugha8b_bf16_lora_a100_fast_v1.csv
```

## Notes

- Lugha-Llama is a base model, not an instruction-tuned chat model.
- The prompt is a plain instruction format ending with `Answer:`.
- Training masks prompt tokens and computes loss only on answer tokens.
- `bf16: true` and `tf32: true` are enabled for A100 throughput.
- No `load_in_4bit`, `BitsAndBytesConfig`, or `prepare_model_for_kbit_training` is used.
