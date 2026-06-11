"""RQ-Kmeans: 3-layer residual-quantized K-Means itemic tokenizer.

Reproduces the tokenization scheme from the OpenOneRec report (§4.1.1):
  - Three quantization layers, codebook size 8192 per layer.
  - Layer 1 quantizes the embedding; layer 2 quantizes the residual of layer 1;
    layer 3 quantizes the residual of layer 2.
  - Each item -> tuple of hierarchical codes (c1, c2, c3), so items with similar
    semantics share prefixes (the property the paper relies on for transfer).

This produces codes compatible with the project's SID token format
``<s_a_{c1}><s_b_{c2}><s_c_{c3}>`` (a/b/c == layer 1/2/3). The codebook size
defaults to 8192 to match both the paper and the project's tokenizer vocab.

NOTE on collisions: with three 8192-way layers the theoretical space is huge,
but K-Means assigns items to the *nearest* centroid, so semantically identical
items can collide. The paper reports Strategy-3 collisions at 0.47% because the
appended keywords disambiguate downstream; the raw 3-layer code collision rate
is computed and logged here for transparency.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _kmeans(
    x: np.ndarray,
    n_clusters: int,
    n_iter: int = 50,
    seed: int = 42,
    minibatch: bool = True,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Run K-Means; return (centroids [K, D], assignments [N]).

    Uses scikit-learn MiniBatchKMeans for speed on large item sets, falling
    back to full KMeans for small corpora. ``n_clusters`` is capped at the
    number of samples (cannot have more clusters than points).
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans

    n_samples = x.shape[0]
    k = min(n_clusters, n_samples)
    if k < n_clusters:
        logger.warning(
            "Requested %d clusters but only %d samples; capping K=%d.",
            n_clusters, n_samples, k,
        )

    if minibatch and n_samples > 5000:
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            max_iter=n_iter,
            batch_size=max(batch_size, k),
            n_init=3,
        )
    else:
        km = KMeans(n_clusters=k, random_state=seed, max_iter=n_iter, n_init=3)

    assignments = km.fit_predict(x)
    return km.cluster_centers_.astype(np.float32), assignments.astype(np.int64)


def resolve_codebook_size(
    n_items: int,
    requested: int = 0,
    factor: float = 4.0,
    k_min: int = 64,
    k_max: int = 8192,
) -> int:
    """Choose a per-layer codebook size that does NOT degenerate to one-hot.

    The paper uses 8192 because its corpus has tens of millions of items. For a
    single Amazon domain (a few thousand to tens of thousands of items), 8192
    clusters >= item count, so layer 1 fits the embeddings perfectly (residual
    MSE = 0) and layers 2/3 cluster all-zero residuals -- the 3-layer hierarchy
    collapses to 1 layer.

    To keep all 3 layers meaningful, each cluster must aggregate several items,
    so we need K << N. We pick:

        K = clamp(round(sqrt(N) * factor), k_min, k_max),  then K <= N // 2

    The N//2 cap guarantees >= 2 items per cluster on average, so the residual
    after layer 1 is non-zero and layers 2/3 actually quantize structure.

    Args:
        n_items: number of items in this domain.
        requested: if > 0, use this fixed size (still capped at N//2); if 0,
            auto-resolve.
        factor / k_min / k_max: auto-resolution knobs.

    Returns:
        Per-layer K to use for this domain.
    """
    import math

    hard_cap = max(2, n_items // 2)  # >= 2 items/cluster on average

    if requested and requested > 0:
        k = min(requested, hard_cap)
        if k < requested:
            logger.warning(
                "Requested codebook_size=%d but only %d items; capping to N//2=%d "
                "to keep the 3-layer hierarchy non-degenerate.",
                requested, n_items, k,
            )
        return max(k, 1)

    k = int(round(math.sqrt(max(n_items, 1)) * factor))
    k = max(k_min, min(k, k_max))
    k = min(k, hard_cap)
    return max(k, 1)


@dataclass
class RQKMeansTokenizer:
    """Hierarchical 3-layer residual K-Means quantizer."""

    n_layers: int = 3
    codebook_size: int = 8192
    seed: int = 42
    centroids: list[np.ndarray] = field(default_factory=list)  # per-layer [K, D]

    def fit(self, embeddings: np.ndarray) -> "RQKMeansTokenizer":
        """Fit the residual codebooks on item embeddings [N, D].

        The effective per-layer K is resolved against the number of items so
        the hierarchy never collapses (see resolve_codebook_size). The resolved
        value REPLACES self.codebook_size so encode()/save() stay consistent.
        """
        x = embeddings.astype(np.float32)
        n_items = x.shape[0]

        effective_k = resolve_codebook_size(n_items, requested=self.codebook_size)
        if effective_k != self.codebook_size:
            logger.info(
                "Resolved per-layer codebook size: %d (was %d) for %d items.",
                effective_k, self.codebook_size, n_items,
            )
        self.codebook_size = effective_k

        residual = x.copy()
        self.centroids = []

        for layer in range(self.n_layers):
            logger.info(
                "RQ-Kmeans layer %d/%d: clustering %d residual vectors into %d codes ...",
                layer + 1, self.n_layers, residual.shape[0], self.codebook_size,
            )
            cents, assign = _kmeans(
                residual, self.codebook_size, seed=self.seed + layer
            )
            self.centroids.append(cents)
            # Subtract the assigned centroid to form the next-layer residual.
            residual = residual - cents[assign]
            mse = float((residual ** 2).sum(axis=1).mean())
            logger.info("  layer %d residual MSE = %.6f", layer + 1, mse)

        return self

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """Encode embeddings [N, D] -> codes [N, n_layers] (int)."""
        if not self.centroids:
            raise RuntimeError("Tokenizer not fitted; call fit() first.")
        x = embeddings.astype(np.float32)
        residual = x.copy()
        codes = np.zeros((x.shape[0], self.n_layers), dtype=np.int64)

        for layer in range(self.n_layers):
            cents = self.centroids[layer]  # [K, D]
            # nearest-centroid assignment via squared L2
            # ||r - c||^2 = ||r||^2 - 2 r·c + ||c||^2 ; drop ||r||^2 (constant per row)
            dots = residual @ cents.T  # [N, K]
            cent_sq = (cents ** 2).sum(axis=1)  # [K]
            dist = cent_sq[None, :] - 2.0 * dots  # [N, K]
            assign = dist.argmin(axis=1)
            codes[:, layer] = assign
            residual = residual - cents[assign]

        return codes

    def collision_rate(self, codes: np.ndarray) -> float:
        """Fraction of items sharing an identical (c1, c2, c3) tuple."""
        n = codes.shape[0]
        if n == 0:
            return 0.0
        seen: dict[tuple, int] = {}
        collided = 0
        for row in codes:
            key = tuple(int(v) for v in row)
            if key in seen:
                collided += 1
            seen[key] = seen.get(key, 0) + 1
        return collided / n

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "n_layers": self.n_layers,
                    "codebook_size": self.codebook_size,
                    "seed": self.seed,
                    "centroids": self.centroids,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("RQKMeansTokenizer saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "RQKMeansTokenizer":
        with open(path, "rb") as f:
            d = pickle.load(f)
        tok = cls(
            n_layers=d["n_layers"],
            codebook_size=d["codebook_size"],
            seed=d["seed"],
        )
        tok.centroids = d["centroids"]
        return tok


def build_item_codes(
    item_ids: list[str],
    embeddings: np.ndarray,
    codebook_size: int = 0,
    seed: int = 42,
) -> tuple[dict[str, tuple[int, int, int]], RQKMeansTokenizer, float]:
    """Fit RQ-Kmeans and encode items.

    Args:
        codebook_size: per-layer K. 0 (default) = auto-resolve from the item
            count so the 3-layer hierarchy stays non-degenerate; a positive
            value forces that size (still capped at N//2).

    Returns:
        asin2code: asin -> (c1, c2, c3)
        tokenizer: fitted RQKMeansTokenizer (for persistence / new items)
        collision_rate: fraction of items with a duplicated 3-tuple
    """
    tokenizer = RQKMeansTokenizer(n_layers=3, codebook_size=codebook_size, seed=seed)
    tokenizer.fit(embeddings)
    codes = tokenizer.encode(embeddings)
    collision = tokenizer.collision_rate(codes)
    logger.info(
        "Raw 3-layer itemic-token collision rate: %.2f%% (per-layer K=%d)",
        collision * 100, tokenizer.codebook_size,
    )

    asin2code = {
        iid: (int(codes[i, 0]), int(codes[i, 1]), int(codes[i, 2]))
        for i, iid in enumerate(item_ids)
    }
    return asin2code, tokenizer, collision