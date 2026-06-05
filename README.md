# MetadataLearn

**Automated Dataset Metadata Generation — Fine-Tuning Qwen 3 8B**

MetadataLearn fine-tunes an open-weights **Qwen 3 8B** model to generate concise, grounded **dataset cards** straight from raw tabular metadata (column names, types, sample values). It runs on consumer hardware — the 4-bit base plus the LoRA adapter fits in **under 8 GB of VRAM** — giving teams a zero-API-cost, fully private documentation tool. The whole pipeline lives in four notebooks.

## Why

Good data documentation is a prerequisite for Responsible AI but is tedious to write, so it gets skipped. Frontier LLMs help but conflate verbosity with quality (hallucinating use cases, padding with filler) and require sending sensitive data to third-party APIs. Our solution fixes both with a small, local model aligned specifically for **brevity** and **grounding**. Three ideas make it work: (1) a DPO pass that explicitly trains *against* verbosity via "gold vs. degraded" preference pairs; (2) local, privacy-preserving deployment under 8 GB VRAM; (3) strict grounding in the provided schema only. Output follows Google's Data Cards format and transfers cleanly to Hugging Face cards.

## How it works

One multi-task adapter trains two tasks, signalled by the instruction text: **`dataset_description`** (name + schema → dataset description) and **`column_description`** (dataset context + one column → that column's description).

**Data** comes from the [Washington State Open Data Portal](https://data.wa.gov/) via the Socrata Discovery API. Datasets lacking a usable description or with fewer than 10 rows are dropped; the rest are split **held-out by dataset** 80/10/10 (seed 42) → 466 / 58 / 58 datasets (5,572 / 637 / 699 examples).

**Training** is QLoRA SFT then a DPO pass. SFT loads `Qwen/Qwen3-8B` in 4-bit NF4 with LoRA (r=16, α=32, dropout=0.05) on all 7 attention+MLP projections — only ~2.2M params (0.025%) trainable; 3 epochs, LR 2e-4, effective batch 16, seq len 4,096. DPO continues the SFT adapter (β=0.1, LR 5e-6, 1 epoch) with programmatic preference pairs: *verbosity rejection* (gold vs. filler-padded gold) for datasets, *cross-contamination rejection* (gold vs. another column's gold) for columns — no teacher rollout needed.

**Evaluation** scores both variants with ROUGE-1/2/L, BERTScore F1 (`roberta-large`), and a length ratio flagged above 1.15× of the reference.

## Notebooks

| Notebook | Does | GPU |
|---|---|---|
| `build_metadata_store.ipynb` | Crawls `data.wa.gov` via the Socrata API → `all_metadata.json`. Pure stdlib, resumable. | No |
| `finetune_descriptions.ipynb` | Builds the split + SFT/DPO data, runs SFT then DPO → final adapter. | Yes |
| `evaluate_descriptions.ipynb` | Runs baseline (adapter off) and fine-tuned (adapter on) on the test split, renders side-by-side tables. | Yes |
| `demo.ipynb` | Point at any live Socrata dataset → base vs. fine-tuned descriptions + a Markdown dataset card. | Yes |

Data-prep cells (`# @dryrun`) are pure Python; on Colab each notebook re-roots paths under a Drive `PROJECT_DIR` so artifacts persist.

## Getting started

GPU steps were developed on an H200 and validated on Colab (T4 — set `bf16=False, fp16=True`).

1. Add a Hugging Face token (`HF_TOKEN`, read scope) to `.env` to pull Qwen 3 weights. `.env` is gitignored.
2. Run `build_metadata_store.ipynb` → `all_metadata.json`.
3. Run `finetune_descriptions.ipynb` → `splits.json`, SFT/DPO data, and `adapters/qwen3-8b-desc-dpo/`.
4. Run `evaluate_descriptions.ipynb` → baseline vs. fine-tuned metrics (`comparison_results.json`).
5. Run `demo.ipynb` (e.g. `f6w7-q2d2` @ `data.wa.gov`) to compare and export a dataset card.

Dependencies: `transformers>=4.51`, `trl>=0.12`, `peft>=0.13`, `bitsandbytes>=0.45`, `datasets>=2.20`, `accelerate>=0.34`, `rouge-score`, `bert-score`, `pandas`. Each GPU notebook installs these in its first cell.

## Results

Fine-tuned model (`Qwen/Qwen3-8B` + `qwen3-8b-desc-dpo`) vs. untuned baseline on 58 held-out test datasets. Verbosity threshold = 15% over the human-written reference length.

| Variant        | Task          | ROUGE-1   | ROUGE-L   | BERTScore F1 | Length Ratio | % Over    |
| -------------- | ------------- | --------- | --------- | ------------ | ------------ | --------- |
| Baseline       | Overall       | 0.268     | 0.220     | 0.179        | 3.13×        | 73.4%     |
| **Fine-tuned** | Overall       | **0.661** | **0.634** | **0.629**    | **1.14×**    | **20.3%** |
|                | Dataset desc. | 0.508     | 0.453     | 0.415        | 1.58×        | 34.5%     |
|                | Column desc.  | 0.675     | 0.651     | 0.648        | 1.10×        | 19.0%     |

Every metric improved: the model became **both more faithful and far more concise**, cutting over-length outputs from ~3/4 to ~1/5 of the test set. Since brevity shrinks the surface area for hallucination, the two gains reinforce each other.
