AI-Metadata-Fine-Tuning-Tool

**Automated Column-Description Generation — Fine-Tuning Gemma-4-31B**

This tool fine-tunes an open-weights **Gemma-4-31B** model (`google/gemma-4-31B`, a dense 31B-parameter multimodal checkpoint — only the language model is adapted) to generate concise, grounded **column descriptions** for tabular datasets straight from raw column metadata (names, types, per-column statistics, and sample values). The training prompt is aligned 1:1 with the production **AI Metadata Improvement Tool**: `prompts/system.md` + `prompts/column.md` are loaded from that repo's checkout when present (falling back to this repo's committed `prompts/` copies, then to embedded strings) and filled as real `{token}` templates — with the production sanitizers mirrored — so the adapter drops straight into that tool's column endpoint; `scripts/check_consistency.py` asserts the copies never drift. Loaded in 4-bit with a LoRA adapter, the base is ~22 GB of weights and trains on a single A100 (40 GB or 80 GB), giving teams a zero-API-cost, fully private documentation tool. The whole pipeline lives in three notebooks.

## Why

Good data documentation is a prerequisite for Responsible AI but is tedious to write, so it gets skipped. Column-level descriptions are the most repetitive part — a portal can have tens of thousands of undocumented columns. Frontier LLMs help but conflate verbosity with quality (hallucinating meanings, padding with filler) and require sending sensitive data to third-party APIs. Our solution fixes both with a local, open-weights model aligned specifically for **brevity** and **grounding**. Three ideas make it work: (1) a DPO pass that explicitly trains *against* verbosity and *against* ungrounded text via "gold vs. degraded" preference pairs; (2) local, privacy-preserving deployment (4-bit on a single GPU); (3) strict grounding in the provided schema, statistics, and sample values only.

## How it works

One adapter trains a single task — **`column_description`**: given the dataset context (name + its description) plus one column's name, type, per-column statistics (completeness, null count, distinct-value count, value range/aggregates), and sample values, write that column's plain-language description. The prompt **is** `prompts/column.md` from the production tool, used as a real template — the same five required elements (definition & significance, unit of measurement, possible values, empty cells, methods & standards), the same ~50-word / 2–5-sentence target, the same `<<<UNTRUSTED_DATA>>>` fences, and the same always-present existing-description REFERENCE block — with `{columnStats}` / `{sampleValues}` / `{dataType}` rendered in the production formatter's shapes (categorical top-value percentage lists, the `<50`-distinct / `<0.5`-repeat-ratio categorical rule, `Text (Categorical)`-style type labels). When a checkout of the Improvement Tool sits next to this repo (or `IMPROVEMENT_TOOL_DIR` points at one), the notebooks read the live `prompts/*.md` files; otherwise they fall back to embedded copies.

**Data** comes from the [Washington State Open Data Portal](https://data.wa.gov/) via the Socrata Discovery API. Each column's `cachedContents` profile (non-null/null counts, cardinality, min/max/avg/sum, top values with counts; numeric columns add sample-based median/quartiles/mode) is captured alongside its samples and gold description. Datasets with no usable column description, or fewer than 10 rows, are dropped; the rest are split **held-out by dataset** 80/10/10 (seed 42) so a dataset's columns never straddle two splits — one `column_description` example per documented column. The sponsor's **golden** datasets — a curated allowlist of UIDs — are tagged during the crawl and pinned to the held-out test split as an independent eval anchor (their stored descriptions are ignored; the latest metadata is fetched live by UID). Exact split and example counts depend on the live crawl and how many golden datasets are pinned.

**The crawl is style truth, not ground truth.** Human-written portal descriptions teach grounding and brevity, but some violate the very plain-language rules the fine-tune targets, and some contain content no model could infer (statute citations, links) — imitating those teaches fabrication. Train/val targets therefore pass a three-tier curation before SFT: **FIX** (mechanical, meaning-safe WA word substitutions), **DROP** (uninferable content, >90 words, unfixable jargon, deadly-7 overuse across 2+ sentences), and **PREFER** (multi-sentence golds inside the 30–70-word window are duplicated ×2 so the five-element house style anchors training). Per the sponsor's gold-standard analysis, "This …" openers and terse one-liners are house style and are kept. Test targets stay raw — evaluation measures against real human text.

**Training** is QLoRA SFT then a DPO pass. SFT loads `google/gemma-4-31B` in 4-bit NF4 with LoRA (r=16, α=32, dropout=0.05); the target modules are **auto-detected** (the language model's attention + MLP projections — the checkpoint's vision tower is excluded, since this is a text-only task); 3 epochs, LR 2e-4, effective batch 16, seq len 4,096. About a quarter of the curated train examples are duplicated as **improve-path** twins — the prompt's existing-description REFERENCE block carries a one-axis-degraded gold while the target stays the clean gold — so the adapter also learns the production tool's *improve an existing description* flow without ever being rewarded for copying the reference. Gemma-4-31B is a **base** checkpoint with no chat template, so the notebooks install a plain-text template that owns the special tokens — one `<bos>` at the start and `<eos>` after the assistant turn, which is what teaches the adapter to *stop* instead of running to the token ceiling; a parity guard asserts both before training. DPO continues the SFT adapter (β=0.1, LR 5e-6, 1 epoch) with **six programmatic preference strategies**, no teacher rollout and no judge reward to hack — each rejected sample degrades the gold on exactly one axis, so the preference signal stays correct even where the gold itself is imperfect: *cross-contamination* (a different dataset's gold — never a near-duplicate and never a same-named column, whose description would usually be valid here — grounding), *verbosity* (gold + 1–3 sampled filler sentences — description length), *run-on* (gold sentences merged into one >20-word sentence — the observed LLM failure), *jargon* (the WA substitution list applied in reverse — word choice), *acronym-unexpanded* ("Full Name (ABC)" collapsed to bare "ABC" — only where the human gold itself expanded, so guessed expansions are never rewarded), and *improve-no-echo* (a degraded gold sits in the REFERENCE block and is itself the rejected side — trains improving a reference instead of echoing it).

**Evaluation** scores both variants with ROUGE-1/2/L, BERTScore F1 (`roberta-large`), a length ratio flagged above 1.15× of the reference (plus the share of generations hitting the max-token ceiling), and **deterministic WA plain-language checks** — word-count window, sentence length (mean words/sentence + share over the WA 20-word rule), unexpanded acronyms, Flesch-Kincaid grade, WA jargon list, "deadly 7 verbs" — mirroring the capstone Output Evaluation Report. Because test targets are raw human text while training targets were curated, the WA-check deltas and length ratio are the primary outcome, read against a **gold-reference calibration row** (what sponsor-quality human text scores on the same rules); ROUGE/BERTScore serve as grounding sanity checks. A **gold benchmark** re-scores the same predictions on just the sponsor golden datasets, and a **novel-target subset** re-scores only test examples whose gold text never appears among the train targets (a control for the boilerplate descriptions portals reuse across datasets). An **improve-path slice** re-runs generation with a degraded gold in the REFERENCE block and reports an echo rate — whether the model improves the reference or merely copies it.

## Notebooks

| Notebook | Does | GPU |
|---|---|---|
| `build_metadata_store.ipynb` | Crawls `data.wa.gov` via the Socrata API → `all_metadata.json` (per-column stats + gold descriptions); tags the sponsor golden datasets. Stdlib + `openpyxl` (golden allowlist), resumable. | No |
| `finetune_descriptions.ipynb` | Builds the split, curates the gold targets (FIX/DROP/PREFER funnel) + SFT/DPO data (incl. `test_examples.jsonl` for eval), runs SFT then DPO → final adapter. | Yes |
| `evaluate_descriptions.ipynb` | Runs baseline (adapter off) and fine-tuned (adapter on) on the rendered test examples; side-by-side tables, gold benchmark, novel-target subset. | Yes |

Data-prep cells are pure Python; on Colab each notebook re-roots paths under a Drive `PROJECT_DIR` so artifacts persist.

`scripts/check_consistency.py` (stdlib, no GPU) enforces the cross-file invariants from the capstone Failure Log: the chat template is byte-identical across the two GPU notebooks, `split_uids()` logic is identical across notebooks, and the embedded prompts match `prompts/*.md` here and in the Improvement Tool checkout. Run it after touching any of those.

## Getting started

The dense 31B in 4-bit (~22 GB) needs a high-memory GPU — an A100 (40 GB+) is recommended; a Colab T4 (16 GB) is not enough; point `MODEL_ID` at a smaller checkpoint to experiment there. Set `LORA_ATTENTION_ONLY=True` for a lighter, faster run.

**Running off Colab (local machine or remote GPU server).** Each notebook auto-detects the absence of Colab and keeps everything on the local disk — no Drive mount, no upload widgets. By default artifacts and the Hugging Face cache live in the current directory; export `PROJECT_DIR` to put them elsewhere (handy for a scratch/data volume on a server):

```bash
export PROJECT_DIR=/data/metadata-finetune   # artifacts + hf_cache live here
export HF_TOKEN=hf_xxx                        # or keep it in .env (see below)
```

The same `PROJECT_DIR` is shared by all three notebooks, so the crawl, training, and eval read and write the same folder.

The sponsor **golden allowlist** (`FourMusketeersCapstone_DatasetsWithSolidMetadata(DataWA)_*.xlsx`) is the eval-anchor input. Keep it in the project root before step 2 so `build_metadata_store.ipynb` can tag those datasets `golden` — only the UID column is read (the stored descriptions are stale; the latest metadata is fetched live). Golden datasets are held out of training and scored separately as the gold benchmark in step 4. Set `GOLDEN_XLSX = None` to skip the anchor entirely.

1. `google/gemma-4-31B` itself is public, so a Hugging Face token is optional — set `HF_TOKEN` (read scope) only if you point `MODEL_ID` at a gated checkpoint, either exported in the environment or in `.env` (the GPU notebooks read `HF_TOKEN=` from `.env` in the cwd or `PROJECT_DIR`). `.env` is gitignored.
2. Run `build_metadata_store.ipynb` → `all_metadata.json` (tags the golden datasets if the allowlist xlsx is present). Stores crawled before June 2026 lack the top-values/percentile stats fields the production-shaped prompts use — re-run once with `OVERWRITE = True` to pick them up.
3. Run `finetune_descriptions.ipynb` → `splits.json`, SFT/DPO data + `sft_data/test_examples.jsonl`, and `adapters/gemma-4-31b-coldesc-dpo/`.
4. Run `evaluate_descriptions.ipynb` → baseline vs. fine-tuned metrics (`comparison_results.json`), plus the gold benchmark and the novel-target subset.

Dependencies: `transformers>=5.11` (the gemma4 architecture plus the text-only-training fix from transformers PR #45454), `trl>=0.12`, `peft>=0.13`, `bitsandbytes>=0.45`, `datasets>=2.20`, `accelerate>=0.34`, `rouge-score`, `bert-score`, `pandas`. Each GPU notebook installs these in its first cell.

## Results

> ⚠️ The table below is from the original **Qwen3-8B**, two-task run (dataset + column), kept for reference only. The pipeline now trains the **column description** task alone with the `column.md`-aligned prompt — so the **Column desc.** row is the relevant one, and the **Dataset desc.** row reflects the retired task. Re-run `evaluate_descriptions.ipynb` after fine-tuning `Gemma-4-31B` with the updated prompts (plus the gold benchmark, novel-target subset, and WA plain-language checks) to refresh these numbers.

Fine-tuned model (`Qwen/Qwen3-8B` + `qwen3-8b-desc-dpo`) vs. untuned baseline on 58 held-out test datasets. Verbosity threshold = 15% over the human-written reference length.

| Variant        | Task          | ROUGE-1   | ROUGE-L   | BERTScore F1 | Length Ratio | % Over    |
| -------------- | ------------- | --------- | --------- | ------------ | ------------ | --------- |
| Baseline       | Overall       | 0.268     | 0.220     | 0.179        | 3.13×        | 73.4%     |
| **Fine-tuned** | Overall       | **0.661** | **0.634** | **0.629**    | **1.14×**    | **20.3%** |
|                | Dataset desc. | 0.508     | 0.453     | 0.415        | 1.58×        | 34.5%     |
|                | Column desc.  | 0.675     | 0.651     | 0.648        | 1.10×        | 19.0%     |

Every metric improved: the model became **both more faithful and far more concise**, cutting over-length outputs from ~3/4 to ~1/5 of the test set. Since brevity shrinks the surface area for hallucination, the two gains reinforce each other.
