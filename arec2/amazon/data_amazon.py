"""Amazon-2014 data loading (parquet), 5-core filtering, leave-one-out splitting.

The expected layout (matches the uploaded data/amazon14/ folder) is, per
category C:

    data/amazon14/meta_C.parquet          # item metadata
    data/amazon14/reviews_C_5.parquet      # 5-core reviews

Columns use the standard Amazon-2014 field names:
    meta:    asin, title, description, categories, brand
    reviews: reviewerID, asin, unixReviewTime, overall

We are tolerant of a few common aliases (e.g. user_id / item_id / timestamp)
as a safety net, but default to the standard names above.

We follow the paper's protocol (section 6.1):
  - keep users/items with >= 5 interactions (the *_5.parquet files are already
    5-core, but we re-filter defensively after any pruning);
  - leave-one-out: per user, sort interactions by timestamp; the last item is
    the test target, the second-to-last is validation, the rest are history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# The 10 Amazon domains used in the paper (section 6.1), with the exact file
# stems present in data/amazon14/ (see the uploaded folder screenshot).
AMAZON_CATEGORIES = [
    "Baby",
    "Beauty",
    "Cell_Phones_and_Accessories",
    "Grocery_and_Gourmet_Food",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Pet_Supplies",
    "Sports_and_Outdoors",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]

# Short aliases the paper uses in tables (Baby, Beauty, Cell, Grocery, ...).
CATEGORY_SHORT = {
    "Baby": "Baby",
    "Beauty": "Beauty",
    "Cell_Phones_and_Accessories": "Cell",
    "Grocery_and_Gourmet_Food": "Grocery",
    "Health_and_Personal_Care": "Health",
    "Home_and_Kitchen": "Home",
    "Pet_Supplies": "Pet",
    "Sports_and_Outdoors": "Sports",
    "Tools_and_Home_Improvement": "Tools",
    "Toys_and_Games": "Toys",
}

# Column-name resolution: standard Amazon-2014 names first, then common aliases.
_META_ASIN_COLS = ["asin", "item_id", "itemID", "iid", "product_id"]
_REV_USER_COLS = ["reviewerID", "user_id", "userID", "uid", "reviewer_id"]
_REV_ITEM_COLS = ["asin", "item_id", "itemID", "iid", "product_id"]
_REV_TIME_COLS = ["unixReviewTime", "timestamp", "time", "reviewTime", "unix_review_time"]


def _resolve_col(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find a column for {what}. Tried {candidates}; "
        f"available columns: {list(df.columns)}"
    )


@dataclass
class AmazonDomain:
    """A single Amazon category with parsed interactions + item metadata.

    Attributes:
        category: Full category name (e.g., "Toys_and_Games").
        item_ids: Ordered list of unique item ASINs that survived 5-core.
        item_meta: asin -> raw metadata dict (title/description/categories/...).
        user_sequences: uid -> chronologically ordered list of asins.
        train_histories: uid -> history asins (all but last two).
        val_targets: uid -> validation target asin (second-to-last).
        test_targets: uid -> test target asin (last).
    """

    category: str
    item_ids: list[str]
    item_meta: dict[str, dict]
    user_sequences: dict[str, list[str]]
    train_histories: dict[str, list[str]] = field(default_factory=dict)
    val_targets: dict[str, str] = field(default_factory=dict)
    test_targets: dict[str, str] = field(default_factory=dict)

    @property
    def short_name(self) -> str:
        return CATEGORY_SHORT.get(self.category, self.category)

    @property
    def n_users(self) -> int:
        return len(self.user_sequences)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_interactions(self) -> int:
        return sum(len(s) for s in self.user_sequences.values())


def _to_native(value):
    """Convert numpy/pandas scalars and arrays to plain Python for metadata."""
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_native(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    return value


def _load_meta(meta_path: Path) -> dict[str, dict]:
    """Load item metadata keyed by ASIN from a parquet file."""
    df = pd.read_parquet(meta_path)
    asin_col = _resolve_col(df, _META_ASIN_COLS, "meta item id (asin)")

    item_meta: dict[str, dict] = {}
    columns = list(df.columns)
    for row in df.itertuples(index=False, name=None):
        rec = {col: _to_native(val) for col, val in zip(columns, row)}
        asin = rec.get(asin_col)
        if asin is None:
            continue
        asin = str(asin)
        # Normalise the asin key to "asin" so downstream build_item_text works.
        rec["asin"] = asin
        item_meta[asin] = rec

    logger.info("  Loaded metadata for %d items from %s", len(item_meta), meta_path.name)
    return item_meta


def _load_reviews(reviews_path: Path) -> list[tuple[str, str, float]]:
    """Load (uid, asin, timestamp) interaction tuples from a reviews parquet."""
    df = pd.read_parquet(reviews_path)
    user_col = _resolve_col(df, _REV_USER_COLS, "review user id (reviewerID)")
    item_col = _resolve_col(df, _REV_ITEM_COLS, "review item id (asin)")

    # Timestamp is optional; if absent we fall back to row order.
    time_col = None
    for c in _REV_TIME_COLS:
        if c in df.columns:
            time_col = c
            break

    interactions: list[tuple[str, str, float]] = []
    if time_col is not None:
        cols = df[[user_col, item_col, time_col]]
        for order, (uid, asin, ts) in enumerate(cols.itertuples(index=False, name=None)):
            if pd.isna(uid) or pd.isna(asin):
                continue
            try:
                ts_val = float(ts) if not pd.isna(ts) else float(order)
            except (TypeError, ValueError):
                ts_val = float(order)
            interactions.append((str(uid), str(asin), ts_val))
    else:
        logger.warning(
            "  No timestamp column found in %s; using file row order as the sort key.",
            reviews_path.name,
        )
        cols = df[[user_col, item_col]]
        for order, (uid, asin) in enumerate(cols.itertuples(index=False, name=None)):
            if pd.isna(uid) or pd.isna(asin):
                continue
            interactions.append((str(uid), str(asin), float(order)))

    logger.info("  Loaded %d interactions from %s", len(interactions), reviews_path.name)
    return interactions


def _five_core_filter(
    interactions: list[tuple[str, str, float]], min_count: int = 5
) -> list[tuple[str, str, float]]:
    """Iteratively drop users/items with < min_count interactions until stable."""
    from collections import Counter

    cur = interactions
    while True:
        user_cnt = Counter(u for u, _, _ in cur)
        item_cnt = Counter(i for _, i, _ in cur)
        nxt = [
            (u, i, t)
            for (u, i, t) in cur
            if user_cnt[u] >= min_count and item_cnt[i] >= min_count
        ]
        if len(nxt) == len(cur):
            return nxt
        cur = nxt


def _build_sequences(
    interactions: list[tuple[str, str, float]]
) -> dict[str, list[str]]:
    """Group by user and sort chronologically; de-duplicate consecutive repeats."""
    from collections import defaultdict

    by_user: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    for order, (uid, asin, ts) in enumerate(interactions):
        by_user[uid].append((ts, order, asin))

    sequences: dict[str, list[str]] = {}
    for uid, events in by_user.items():
        events.sort(key=lambda x: (x[0], x[1]))  # ts, then file order as tiebreak
        seq: list[str] = []
        for _, _, asin in events:
            # Keep order; allow repeats only if not immediately consecutive.
            if not seq or seq[-1] != asin:
                seq.append(asin)
        sequences[uid] = seq
    return sequences


def load_amazon_domain(
    data_dir: str,
    category: str,
    min_count: int = 5,
    max_hist_len: int = 50,
) -> AmazonDomain:
    """Load one Amazon category (parquet) and produce leave-one-out splits.

    Args:
        data_dir: Path to data/amazon14.
        category: Full category name (e.g., "Toys_and_Games").
        min_count: 5-core threshold.
        max_hist_len: Truncate each history to its most recent N items.

    Returns:
        Populated AmazonDomain with train_histories / val_targets / test_targets.
    """
    data_path = Path(data_dir)
    meta_path = data_path / f"meta_{category}.parquet"
    reviews_path = data_path / f"reviews_{category}_5.parquet"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    if not reviews_path.exists():
        raise FileNotFoundError(f"Missing reviews file: {reviews_path}")

    logger.info("Loading Amazon domain '%s' from %s", category, data_dir)
    item_meta = _load_meta(meta_path)
    interactions = _load_reviews(reviews_path)

    interactions = _five_core_filter(interactions, min_count=min_count)
    sequences = _build_sequences(interactions)

    # Leave-one-out: need at least 3 events (history >=1, val, test).
    train_hist: dict[str, list[str]] = {}
    val_tgt: dict[str, str] = {}
    test_tgt: dict[str, str] = {}
    used_items: set[str] = set()

    for uid, seq in sequences.items():
        if len(seq) < 3:
            continue
        hist = seq[:-2]
        if max_hist_len and len(hist) > max_hist_len:
            hist = hist[-max_hist_len:]
        train_hist[uid] = hist
        val_tgt[uid] = seq[-2]
        test_tgt[uid] = seq[-1]
        used_items.update(hist)
        used_items.add(seq[-2])
        used_items.add(seq[-1])

    item_ids = sorted(used_items)

    domain = AmazonDomain(
        category=category,
        item_ids=item_ids,
        item_meta=item_meta,
        user_sequences={u: sequences[u] for u in train_hist},
        train_histories=train_hist,
        val_targets=val_tgt,
        test_targets=test_tgt,
    )

    logger.info(
        "  Domain '%s' ready: %d users, %d items, %d interactions (5-core, leave-one-out)",
        category, domain.n_users, domain.n_items, domain.n_interactions,
    )
    return domain
