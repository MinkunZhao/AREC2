"""Item semantic embeddings (Qwen3-Embedding) and 5-keyword extraction.

This mirrors the paper's pipeline (§4.1.1): item text metadata is encoded by
Qwen3-Embedding to produce semantic embeddings, which are later quantized by
RQ-Kmeans into 3-layer itemic tokens. Separately, for Strategy 3 (§6.3.1) we
extract 5 distinctive keywords per item from its metadata for the
``[itemic_tokens] + [keywords]`` representation.

Design notes:
  - We build a single canonical text per item (title + brand + category +
    truncated description). This text serves both embedding and keyword
    extraction so the two views stay consistent.
  - Qwen3-Embedding is loaded via sentence-transformers if available (it ships
    a ST wrapper), else via transformers with mean pooling. The model id is
    "Qwen/Qwen3-Embedding-0.6B" by default (smallest variant; the report uses
    Qwen3-8B-Embedding, swap via --embed-model for full fidelity).
  - Keyword extraction is TF-IDF based across the domain corpus, which yields
    "distinctive" keywords (rare-in-corpus, frequent-in-item) exactly as the
    paper intends ("5 distinctive keywords extracted from its metadata").
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def build_item_text(meta: dict, max_desc_chars: int = 512) -> str:
    """Concatenate the most informative metadata fields into one string.

    Order matters for embeddings: title first (most salient), then brand,
    category path, then a truncated description.
    """
    parts: list[str] = []
    title = meta.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())

    brand = meta.get("brand")
    if isinstance(brand, str) and brand.strip():
        parts.append(f"Brand: {brand.strip()}")

    # categories is typically a list of lists in Amazon-2014 meta.
    cats = meta.get("categories")
    cat_terms: list[str] = []
    if isinstance(cats, list):
        for c in cats:
            if isinstance(c, list):
                cat_terms.extend(str(x) for x in c)
            elif isinstance(c, str):
                cat_terms.append(c)
    if cat_terms:
        parts.append("Category: " + " > ".join(dict.fromkeys(cat_terms))[:256])

    desc = meta.get("description")
    if isinstance(desc, str) and desc.strip():
        parts.append(desc.strip()[:max_desc_chars])

    text = ". ".join(parts).strip()
    return text if text else "unknown item"


# ---------------------------------------------------------------------------
# Qwen3-Embedding encoder
# ---------------------------------------------------------------------------
class ItemEmbedder:
    """Encode item texts into semantic embeddings with Qwen3-Embedding.

    Tries sentence-transformers first (Qwen3-Embedding ships an ST wrapper),
    then falls back to transformers + last-token / mean pooling.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 512,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self._st_model = None
        self._hf_model = None
        self._hf_tok = None

        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._load(model_name, device)

    def _load(self, model_name: str, device: str):
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model via sentence-transformers: %s", model_name)
            self._st_model = SentenceTransformer(model_name, device=device)
            return
        except Exception as e:  # noqa: BLE001
            logger.info("sentence-transformers path unavailable (%s); using transformers.", e)

        import torch
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading embedding model via transformers: %s", model_name)
        self._hf_tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._hf_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        self._hf_model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._st_model is not None:
            embs = self._st_model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            return embs.astype(np.float32)
        return self._encode_hf(texts)

    def _encode_hf(self, texts: list[str]) -> np.ndarray:
        import torch
        from tqdm import tqdm

        out: list[np.ndarray] = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding items"):
            batch = texts[i : i + self.batch_size]
            enc = self._hf_tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                model_out = self._hf_model(**enc)
            hidden = model_out.last_hidden_state  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).type_as(hidden)
            # Last-token pooling (Qwen3-Embedding convention): take the final
            # non-padded token. We compute it robustly via the mask.
            lengths = enc["attention_mask"].sum(dim=1) - 1  # (B,)
            idx = lengths.clamp(min=0).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
            pooled = hidden.gather(1, idx).squeeze(1)  # (B, H)
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


def embed_items(
    item_ids: list[str],
    item_meta: dict[str, dict],
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    device: str = "auto",
    batch_size: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings [N, D], item_texts) aligned to item_ids order."""
    texts = [build_item_text(item_meta.get(iid, {})) for iid in item_ids]
    embedder = ItemEmbedder(model_name=model_name, device=device, batch_size=batch_size)
    embs = embedder.encode(texts)
    logger.info("Embedded %d items -> shape %s", len(item_ids), embs.shape)
    return embs, texts


# ---------------------------------------------------------------------------
# 5-keyword extraction (TF-IDF "distinctive" keywords)
# ---------------------------------------------------------------------------
def extract_keywords(
    item_ids: list[str],
    item_texts: list[str],
    top_k: int = 5,
) -> dict[str, list[str]]:
    """Extract the top_k most distinctive keywords per item via TF-IDF.

    "Distinctive" = high TF-IDF: terms frequent in the item but rare across the
    domain corpus. This matches the paper's "5 distinctive keywords extracted
    from its metadata" (§6.3.1). Returns asin -> [kw1..kw5].
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:  # noqa: BLE001
        raise ImportError("scikit-learn is required for keyword extraction") from e

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"[A-Za-z][A-Za-z\-']+",
        stop_words="english",
        max_features=200000,
        ngram_range=(1, 2),  # allow short phrases like "memory foam"
        min_df=2,
    )
    matrix = vectorizer.fit_transform(item_texts)  # (N, V) sparse
    vocab = np.array(vectorizer.get_feature_names_out())

    keywords: dict[str, list[str]] = {}
    matrix = matrix.tocsr()
    for row_idx, iid in enumerate(item_ids):
        row = matrix.getrow(row_idx)
        if row.nnz == 0:
            # Fallback: first few words of the title-ish text.
            words = _WORD_RE.findall(item_texts[row_idx].lower())
            keywords[iid] = words[:top_k]
            continue
        data = row.data
        cols = row.indices
        order = np.argsort(data)[::-1][:top_k]
        keywords[iid] = [vocab[cols[j]] for j in order]

    logger.info("Extracted up to %d keywords for %d items", top_k, len(item_ids))
    return keywords
