# Amazon Transfer-Learning Experiment (Strategy 3)

Implements §6.3 of the OpenOneRec report — **Strategy 3: Text-Augmented Itemic
Tokens** — to measure the transfer-learning performance of your AREC²/OneRec
model on the 10 Amazon-2014 domains.

## What Strategy 3 is

Each item is represented as the concatenation of its **original 3-layer
pretrained itemic tokens** and **5 distinctive keywords** from its metadata:

```
<|sid_begin|><s_a_c1><s_b_c2><s_c_c3><|sid_end|> kw1 kw2 kw3 kw4 kw5
```

The 3-layer itemic tokens are **preserved unchanged** (no 4th layer, no
structural extension), so the hierarchical semantics learned in pretraining
stay intact; the keywords add semantic disambiguation (paper collision rate
≈ 0.47%) and let the model exploit its linguistic core.

## Files

```
arec2/amazon/
  data_amazon.py   # parse meta/reviews JSON, 5-core, leave-one-out
  embeddings.py    # Qwen3-Embedding item embeddings + 5-keyword (TF-IDF) extraction
  rqkmeans.py      # 3-layer residual K-Means tokenizer (codebook 8192)
  item_repr.py     # Strategy-3 [itemic_tokens] + [keywords] builder
  dataset.py       # sequential-rec SFT samples (chat format, answer = SID-only)
  metrics.py       # Recall@K / NDCG@K
configs/amazon_transfer_config.yaml
scripts/
  A1_build_amazon_tokens.py     # embed -> RQ-Kmeans -> keywords -> cache
  A2_train_amazon.py            # LoRA fine-tune (single / joint)
  A3_eval_amazon.py             # Recall@{5,10}, NDCG@{5,10}
  A4_run_amazon_experiment.py   # orchestrate build -> train -> eval
```

## Data layout (already present)

```
data/amazon14/
  meta_<Category>.parquet          # item metadata (asin, title, description, categories, brand)
  reviews_<Category>_5.parquet     # 5-core interactions (reviewerID, asin, unixReviewTime, overall)
```

The loader reads parquet via pandas/pyarrow and uses the standard Amazon-2014
column names (with tolerant fallback to common aliases like `user_id`/`item_id`).

## Run it

Your AREC² fine-tune is an **unmerged LoRA adapter** (`full_lora_8b/`) on top
of the **OneRec-8B** base. The scripts load the base, merge this adapter, then
proceed. Pass it via `--sft-adapter` (default `./full_lora_8b`).

Full headline experiment (multi-domain joint, Strategy 3, all 10 domains):

```bash
python scripts/A4_run_amazon_experiment.py \
  --base-model OpenOneRec/OneRec-8B \
  --sft-adapter ./full_lora_8b
```

Quick smoke (3 domains, 1 epoch, capped users):

```bash
python scripts/A4_run_amazon_experiment.py --quick
```

Stage by stage:

```bash
# A1: build itemic tokens + keywords (uses Qwen3-Embedding) -- model-independent
python scripts/A1_build_amazon_tokens.py --embed-model Qwen/Qwen3-Embedding-0.6B

# A2: fine-tune (joint over all 10, or single per domain)
python scripts/A2_train_amazon.py --regime joint \
  --base-model OpenOneRec/OneRec-8B --sft-adapter ./full_lora_8b
python scripts/A2_train_amazon.py --regime single --category Toys_and_Games \
  --base-model OpenOneRec/OneRec-8B --sft-adapter ./full_lora_8b

# A3: evaluate (merges full_lora_8b, then the Amazon adapter from A2)
python scripts/A3_eval_amazon.py \
  --base-model OpenOneRec/OneRec-8B --sft-adapter ./full_lora_8b \
  --adapter ./checkpoints/amazon/strategy3_joint/final
```

**Model loading:** `--base-model` is the OneRec base, `--sft-adapter` is your
AREC² LoRA (merged into the base first). At eval, `--adapter` is the Amazon
transfer LoRA from A2 (merged on top). If you ever produce an already-merged
full model, point `--base-model` at it and pass `--sft-adapter ""`.

## Regimes & ablations (paper §6.3.2)

- `--regime joint`  — Multi-Domain joint training (paper's headline; OneRec
  gains ~+2.3% under joint vs single).
- `--regime single` — Domain-Specific training (one adapter per domain).
- `--ablation itemonly` (A4) / `--no-keywords` (A2) — drops the keywords to
  reproduce the "itemic tokens only" contrast.

## Notes on fidelity

- **Embedding model**: defaults to `Qwen/Qwen3-Embedding-0.6B` so it runs
  anywhere; pass `--embed-model Qwen/Qwen3-Embedding-8B` to match the report
  exactly.
- **Vocabulary**: Amazon codes reuse the existing OneRec SID vocab
  (`<s_a_0..8191>` etc., already present in the tokenizer — verified in
  `try.py`). No vocab expansion is needed because Strategy 3 keeps the 3-layer
  tokens and only appends plain-text keywords.
- **Eval protocol**: leave-one-out with sampled 1+99 candidate ranking by
  default (`--num-negatives`); use `--full-ranking` for exact full-corpus
  ranking. Metrics: Recall@{5,10}, NDCG@{5,10} (paper §6.1).
- **Zero-shot baseline**: run A3 without `--adapter` (but keep `--sft-adapter
  ./full_lora_8b`) to measure your AREC² model's transfer *before* any Amazon
  fine-tuning.

## Dependencies

Beyond your existing `requirements.txt`, A1 uses:

```bash
pip install sentence-transformers   # for Qwen3-Embedding (falls back to transformers)
```

`scikit-learn` (already in requirements) provides both K-Means and TF-IDF.