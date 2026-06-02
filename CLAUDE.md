# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoLaR (Compressed Latent Reasoning) — a research framework for dynamically compressing LLM reasoning chains in latent space. Uses a two-stage training approach: SFT with auxiliary compressed embedding prediction, then optional GRPO-based RL. Built on PyTorch Lightning + HuggingFace Transformers + LoRA (PEFT).

## Commands

```bash
# Training (SFT)
python run.py --model=colar --dataset=qsa --devices=all --do_test \
  dataset_name=gsm model_id=Llama-3.2-1B-Instruct \
  batch_size=256 max_compression_factor=5 compression_factor=5 \
  max_new_tokens=16 max_epochs=50

# Training with RL (load SFT checkpoint first, then train with do_rl=True)
python run.py --model=colar --dataset=qsa --devices=all \
  --load_ckpt_path=/path/to/sft_checkpoint.ckpt \
  do_rl=True lr=1e-6 max_epochs=5

# Evaluation only
python run.py --test_ckpt_path=/path/to/model.ckpt

# Quick debug run (tiny dataset, no logging)
python run.py --model=colar --dataset=qsa --no_log tiny_dataset=True

# View training logs
tensorboard --logdir=logs/
```

`run.py` is the single entry point. It merges three OmegaConf configs (`trainer`, `model`, `dataset`) and any `key=value` command-line overrides. Config keys are matched breadth-first across the merged tree — you can use short keys like `batch_size=128` or full paths like `model.model_kwargs.lora_config.lora_alpha=64`.

## Architecture

### Model hierarchy (all inherit `LitCoTModelBase` → `pl.LightningModule`)

| File | Class | Method | Key difference |
|------|-------|--------|---------------|
| `src/models/cot.py` | `LitCot` | Token-level CoT baseline | Standard next-token prediction on full reasoning chain |
| `src/models/coconut.py` | `LitCoconut` | Fixed-length latent | Progressive curriculum: replaces N CoT steps with N×k latent tokens per stage. Uses `past_key_values` for incremental forward |
| `src/models/icot.py` | `LitICoT` | Truncated CoT | Same curriculum as Coconut but truncates CoT steps instead of replacing with latents — no latent generation at all |
| `src/models/colar.py` | `LitCoLaR` | **Dynamic latent compression (main method)** | Compresses CoT steps by factor r∈[1,R], trains `LatentPolicy` (Gaussian head) to predict next compressed embedding. Supports GRPO RL for exploration |
| `src/models/distill.py` | `LitCoLaR` | Hidden-state distillation | Uses `MLPProjector` to distill teacher hidden states (':') into student latent forward. Note: config targets `src.models.distill.LitDistill` but the class is `LitCoLaR` |

### Key modules (`src/modules/`)

- **`projector.py`**: `LatentPolicy` — Gaussian distribution head (mean + log_std) over embeddings for stochastic latent generation. `MLPProjector` — deterministic MLP projector used in distillation.
- **`grpo.py`**: GRPO implementation — `Experience` dataclass (stores rollout data), `ReplayBuffer`, `GRPOLoss` (clipped PPO-style loss on latent + answer logprobs), group advantage normalization.
- **`embeddings.py`**: Sinusoidal positional encoding (not currently used in main models).

### Data flow

`QSADataModule` (in `src/datasets/qsa.py`) loads JSON split files from `{workspace_path}/datasets/text_reasoning/{dataset_name}/` with structure `[{question, answer, steps: [...]}]`. Batches produce `{idx, question, steps, answer}`.

Each model's `forward()` assembles input embeddings from question + (compressed) steps + answer segments, runs through the LLM with LoRA, and computes a composite loss. The base class `LitCoTModelBase` handles tokenizer setup, LoRA application, optimizer/scheduler config, text/JSON logging, and evaluation via `eval_generation()`.

### Evaluation pipeline

`eval_generation()` dispatches to the appropriate generation method based on `sft_method`:
- `colar` → `latent_generate()` — auto-regressive latent forward until EOL token `###`, then answer generation
- `coconut`/`distill` → `fixed_length_latent_generate()` — fixed N latent steps
- `cot`/`icot` → `text_generate()` — standard token generation

Answers are extracted via `extract_answer_from_output()` (splits on `Answer:` separator) and verified with `verify_answer()` (normalized string/float comparison).

### Configuration system

Three YAML config layers merged by `run.py`:
1. `src/configs/trainer/{trainer}.yaml` — PyTorch Lightning Trainer, callbacks, logger
2. `src/configs/models/{model}.yaml` — model class target, all hyperparameters (LoRA, latent config, RL config, optimizer)
3. `src/configs/datasets/{dataset}.yaml` — data module target, dataset name

Config overrides work via BFS key matching: any `key=value` on command line sets **every** matching key. Short keys like `batch_size=256` work; full paths like `model.model_kwargs.latent_cot_config.max_compression_factor=5` also work.

### Checkpoint handling

`on_save_checkpoint()` saves only trainable parameters (LoRA weights + latent policy head). Checkpoints go to `{log_dir}/checkpoints/`. Top-3 are kept based on the `monitor` metric (default: `monitor` key logged in validation). Loading uses `strict=False` to allow partial weight loading (e.g., loading a CoT checkpoint into CoLaR for SFT initialization).

### Embedding standard deviation constants

`src/utils/constants.py` stores `MODEL_EMB_STD` — per-model embedding standard deviations used to normalize latent embeddings before feeding to `LatentPolicy.log_prob()`. These are empirically measured values critical for stable latent policy training.

### Workspace path

Default `workspace_path` is `/workspace/images-ks3-starfs/workspace/wenhui` (a cluster path). LLMs are expected at `{workspace_path}/models/llms/{model_id}/`. Datasets at `{workspace_path}/datasets/text_reasoning/{dataset_name}/`. Override with `--workspace_path`.

## Data preprocessing

Scripts in `data_preprocessing/` convert raw datasets (GSM8K, MATH, GSM-Hard, SVAMP, MultiArith, GQPA) into the QSA JSON format with question/steps/answer triplets. Each script is standalone.
