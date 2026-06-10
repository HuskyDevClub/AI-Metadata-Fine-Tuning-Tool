AI-Metadata-Fine-Tuning-Tool

**Automated Column-Description Generation — Fine-Tuning Qwen3.6-35B-A3B (MoE)**

This tool fine-tunes an open-weights **Qwen3.6-35B-A3B** model — a Mixture-of-Experts checkpoint (~3B active params per token) — to generate concise, grounded **column descriptions** for tabular datasets straight from raw column metadata (names, types, per-column statistics, and sample values). The training prompt is aligned 1:1 with the production **AI Metadata Improvement Tool** (`prompts/system.md` + `prompts/column.md`), so the adapter drops straight into that tool's column endpoint. Loaded in 4-bit with a LoRA adapter, the base is ~20 GB of weights and trains on a single high-memory GPU (e.g. an H200), giving teams a zero-API-cost, fully private documentation tool. The whole pipeline lives in three notebooks.

## Why

Good data documentation is a prerequisite for Responsible AI but is tedious to write, so it gets skipped. Column-level descriptions are the most repetitive part — a portal can have tens of thousands of undocumented columns. Frontier LLMs help but conflate verbosity with quality (hallucinating meanings, padding with filler) and require sending sensitive data to third-party APIs. Our solution fixes both with a local, open-weights model aligned specifically for **brevity** and **grounding**. Three ideas make it work: (1) a DPO pass that explicitly trains *against* verbosity and *against* ungrounded text via "gold vs. degraded" preference pairs; (2) local, privacy-preserving deployment (4-bit on a single GPU); (3) strict grounding in the provided schema, statistics, and sample values only.

## How it works

One adapter trains a single task — **`column_description`**: given the dataset context (name + its description) plus one column's name, type, per-column statistics (completeness, null count, distinct-value count, value range/aggregates), and sample values, write that column's plain-language description. The prompt mirrors `prompts/column.md` from the production tool — the same five required elements (definition & significance, unit of measurement, possible values, empty cells, methods & standards), the same ~50-word / 2–5-sentence target, and the same `<<<UNTRUSTED_DATA>>>` fences.

**Data** comes from the [Washington State Open Data Portal](https://data.wa.gov/) via the Socrata Discovery API. Each column's `cachedContents` profile (non-null/null counts, cardinality, min/max/avg/sum) is captured alongside its samples and gold description. Datasets with no usable column description, or fewer than 10 rows, are dropped; the rest are split **held-out by dataset** 80/10/10 (seed 42) so a dataset's columns never straddle two splits — one `column_description` example per documented column. The sponsor's **golden** datasets — a curated allowlist of UIDs — are tagged during the crawl and pinned to the held-out test split as an independent eval anchor (their stored descriptions are ignored; the latest metadata is fetched live by UID). Exact split and example counts depend on the live crawl and how many golden datasets are pinned.

**Training** is QLoRA SFT then a DPO pass. SFT loads `Qwen/Qwen3.6-35B-A3B` in 4-bit NF4 with LoRA (r=16, α=32, dropout=0.05); because it is a Mixture-of-Experts model the target modules are **auto-detected** (attention + every expert's MLP projections, skipping the MoE router) rather than hard-coded; 3 epochs, LR 2e-4, effective batch 16, seq len 4,096. DPO continues the SFT adapter (β=0.1, LR 5e-6, 1 epoch) with two programmatic preference strategies per column, no teacher rollout needed: *cross-contamination rejection* (gold vs. another column's gold — trains grounding) and *verbosity rejection* (gold vs. filler-padded gold — trains brevity).

**Evaluation** scores both variants with ROUGE-1/2/L, BERTScore F1 (`roberta-large`), and a length ratio flagged above 1.15× of the reference. A final **gold benchmark** re-scores the same predictions on just the sponsor golden datasets.

## Notebooks

| Notebook | Does | GPU |
|---|---|---|
| `build_metadata_store.ipynb` | Crawls `data.wa.gov` via the Socrata API → `all_metadata.json` (per-column stats + gold descriptions); tags the sponsor golden datasets. Stdlib + `openpyxl` (golden allowlist), resumable. | No |
| `finetune_descriptions.ipynb` | Builds the split + SFT/DPO data, runs SFT then DPO → final adapter. | Yes |
| `evaluate_descriptions.ipynb` | Runs baseline (adapter off) and fine-tuned (adapter on) on the test split, renders side-by-side tables. | Yes |

Data-prep cells are pure Python; on Colab each notebook re-roots paths under a Drive `PROJECT_DIR` so artifacts persist.

## Getting started

GPU steps were developed on an H200. The 35B MoE in 4-bit (~20 GB) needs a high-memory GPU — a Colab T4 (16 GB) is not enough; point `MODEL_ID` at a smaller checkpoint to experiment there. Set `LORA_ATTENTION_ONLY=True` for a lighter, faster run.

**Running off Colab (local machine or remote GPU server).** Each notebook auto-detects the absence of Colab and keeps everything on the local disk — no Drive mount, no upload widgets. By default artifacts and the Hugging Face cache live in the current directory; export `PROJECT_DIR` to put them elsewhere (handy for a scratch/data volume on a server):

```bash
export PROJECT_DIR=/data/metadata-finetune   # artifacts + hf_cache live here
export HF_TOKEN=hf_xxx                        # or keep it in .env (see below)
```

The same `PROJECT_DIR` is shared by all three notebooks, so the crawl, training, and eval read and write the same folder.

The sponsor **golden allowlist** (`FourMusketeersCapstone_DatasetsWithSolidMetadata(DataWA)_*.xlsx`) is the eval-anchor input. Keep it in the project root before step 2 so `build_metadata_store.ipynb` can tag those datasets `golden` — only the UID column is read (the stored descriptions are stale; the latest metadata is fetched live). Golden datasets are held out of training and scored separately as the gold benchmark in step 4. Set `GOLDEN_XLSX = None` to skip the anchor entirely.

1. Add a Hugging Face token (`HF_TOKEN`, read scope) to pull the (gated) Gemma weights — either export it in the environment or put it in `.env` (the GPU notebooks read `HF_TOKEN=` from `.env` in the cwd or `PROJECT_DIR`). `.env` is gitignored.
2. Run `build_metadata_store.ipynb` → `all_metadata.json` (tags the golden datasets if the allowlist xlsx is present).
3. Run `finetune_descriptions.ipynb` → `splits.json`, SFT/DPO data, and `adapters/qwen3.6-35b-a3b-coldesc-dpo/`.
4. Run `evaluate_descriptions.ipynb` → baseline vs. fine-tuned metrics (`comparison_results.json`), plus a gold benchmark on the golden datasets.

Dependencies: `transformers>=4.51`, `trl>=0.12`, `peft>=0.13`, `bitsandbytes>=0.45`, `datasets>=2.20`, `accelerate>=0.34`, `rouge-score`, `bert-score`, `pandas`. Each GPU notebook installs these in its first cell.

## Results

> ⚠️ The table below is from the original **Qwen3-8B**, two-task run (dataset + column), kept for reference only. The pipeline now trains the **column description** task alone with the `column.md`-aligned prompt — so the **Column desc.** row is the relevant one, and the **Dataset desc.** row reflects the retired task. Re-run `evaluate_descriptions.ipynb` after fine-tuning `Qwen3.6-35B-A3B` with the updated prompts (and the gold benchmark) to refresh these numbers.

Fine-tuned model (`Qwen/Qwen3-8B` + `qwen3-8b-desc-dpo`) vs. untuned baseline on 58 held-out test datasets. Verbosity threshold = 15% over the human-written reference length.

| Variant        | Task          | ROUGE-1   | ROUGE-L   | BERTScore F1 | Length Ratio | % Over    |
| -------------- | ------------- | --------- | --------- | ------------ | ------------ | --------- |
| Baseline       | Overall       | 0.268     | 0.220     | 0.179        | 3.13×        | 73.4%     |
| **Fine-tuned** | Overall       | **0.661** | **0.634** | **0.629**    | **1.14×**    | **20.3%** |
|                | Dataset desc. | 0.508     | 0.453     | 0.415        | 1.58×        | 34.5%     |
|                | Column desc.  | 0.675     | 0.651     | 0.648        | 1.10×        | 19.0%     |

Every metric improved: the model became **both more faithful and far more concise**, cutting over-length outputs from ~3/4 to ~1/5 of the test set. Since brevity shrinks the surface area for hallucination, the two gains reinforce each other.
