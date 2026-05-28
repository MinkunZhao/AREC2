"""Offline stores for AREC² retrieval pipeline.

Stores are built from onerec_bench_release.parquet and cached to ./caches/.
Each store exposes a simple query API used by the agentic tools at inference time.
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ProfileStore:
    """Per-user profile summarizing long-term preferences.

    For each user, stores:
      - top_sids: most frequently interacted SIDs (by count)
      - top_liked_sids: SIDs with positive label signals (like, follow, forward)
      - interaction_stats: {total, longview_rate, like_rate, follow_rate, forward_rate}
      - domain_distribution: {video: N, ad: N, product: N}
    """

    def __init__(self):
        self.profiles: dict[int, dict] = {}

    def build(self, data_path: str, pid2sid_video: dict, pid2sid_product: dict):
        import pyarrow.parquet as pq

        logger.info("Building ProfileStore from %s ...", data_path)
        pf = pq.ParquetFile(data_path)
        cols = [
            "uid", "hist_video_pid", "hist_video_like", "hist_video_longview",
            "hist_video_follow", "hist_video_forward", "hist_video_not_interested",
            "hist_ad_pid", "hist_goods_pid",
        ]

        for batch in pf.iter_batches(batch_size=5000, columns=cols):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                uid = int(row["uid"])
                profile = self._build_user_profile(
                    row, pid2sid_video, pid2sid_product
                )
                self.profiles[uid] = profile

        logger.info("ProfileStore built: %d users", len(self.profiles))

    def _build_user_profile(self, row, pid2sid_video, pid2sid_product) -> dict:
        hist_pids = row["hist_video_pid"]
        if hist_pids is None:
            hist_pids = np.array([], dtype=np.int64)

        likes = row["hist_video_like"]
        longview = row["hist_video_longview"]
        follow = row["hist_video_follow"]
        forward = row["hist_video_forward"]
        not_interested = row["hist_video_not_interested"]

        n = len(hist_pids)
        sid_counts = Counter()
        liked_sid_counts = Counter()
        positive_sid_counts = Counter()

        for i in range(n):
            pid = int(hist_pids[i])
            sid_tuple = tuple(pid2sid_video.get(pid, []))
            if not sid_tuple:
                continue
            sid_counts[sid_tuple] += 1
            is_positive = (
                (likes is not None and i < len(likes) and likes[i])
                or (follow is not None and i < len(follow) and follow[i])
                or (forward is not None and i < len(forward) and forward[i])
            )
            if is_positive:
                positive_sid_counts[sid_tuple] += 1
            if likes is not None and i < len(likes) and likes[i]:
                liked_sid_counts[sid_tuple] += 1

        longview_count = int(np.sum(longview)) if longview is not None and len(longview) > 0 else 0
        like_count = int(np.sum(likes)) if likes is not None and len(likes) > 0 else 0
        follow_count = int(np.sum(follow)) if follow is not None and len(follow) > 0 else 0
        forward_count = int(np.sum(forward)) if forward is not None and len(forward) > 0 else 0

        n_ad = len(row["hist_ad_pid"]) if row["hist_ad_pid"] is not None else 0
        n_goods = len(row["hist_goods_pid"]) if row["hist_goods_pid"] is not None else 0

        top_k = 20
        return {
            "top_sids": [list(s) for s, _ in sid_counts.most_common(top_k)],
            "top_positive_sids": [list(s) for s, _ in positive_sid_counts.most_common(top_k)],
            "interaction_stats": {
                "total_video": n,
                "longview_rate": longview_count / n if n > 0 else 0,
                "like_rate": like_count / n if n > 0 else 0,
                "follow_rate": follow_count / n if n > 0 else 0,
                "forward_rate": forward_count / n if n > 0 else 0,
            },
            "domain_distribution": {
                "video": n,
                "ad": n_ad,
                "product": n_goods,
            },
        }

    def query(self, uid: int) -> Optional[dict]:
        return self.profiles.get(uid)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.profiles, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("ProfileStore saved to %s (%.1f MB)", path, Path(path).stat().st_size / 1e6)

    @classmethod
    def load(cls, path: str) -> "ProfileStore":
        store = cls()
        with open(path, "rb") as f:
            store.profiles = pickle.load(f)
        logger.info("ProfileStore loaded: %d users from %s", len(store.profiles), path)
        return store


class LabelBehaviorStore:
    """Per-user histories grouped by interaction type.

    For each user, stores SID lists for: longview, like, follow, forward, not_interested.
    Used by LabelBehaviorTool to provide label-conditioned context.
    """

    def __init__(self):
        self.behaviors: dict[int, dict[str, list]] = {}

    def build(self, data_path: str, pid2sid_video: dict):
        import pyarrow.parquet as pq

        logger.info("Building LabelBehaviorStore from %s ...", data_path)
        pf = pq.ParquetFile(data_path)
        cols = [
            "uid", "hist_video_pid", "hist_video_like", "hist_video_longview",
            "hist_video_follow", "hist_video_forward", "hist_video_not_interested",
        ]
        label_keys = ["longview", "like", "follow", "forward", "not_interested"]
        label_cols = [f"hist_video_{k}" for k in label_keys]

        for batch in pf.iter_batches(batch_size=5000, columns=cols):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                uid = int(row["uid"])
                hist_pids = row["hist_video_pid"]
                if hist_pids is None:
                    continue
                n = len(hist_pids)

                user_behaviors = {}
                for key, col in zip(label_keys, label_cols):
                    labels = row[col]
                    if labels is None or len(labels) == 0:
                        user_behaviors[key] = []
                        continue

                    sids = []
                    for i in range(min(n, len(labels))):
                        if labels[i]:
                            pid = int(hist_pids[i])
                            sid = pid2sid_video.get(pid)
                            if sid:
                                sids.append(sid)
                    user_behaviors[key] = sids

                self.behaviors[uid] = user_behaviors

        logger.info("LabelBehaviorStore built: %d users", len(self.behaviors))

    def query(self, uid: int, label: Optional[str] = None) -> Optional[dict | list]:
        user_data = self.behaviors.get(uid)
        if user_data is None:
            return None
        if label is not None:
            return user_data.get(label, [])
        return user_data

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.behaviors, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("LabelBehaviorStore saved to %s (%.1f MB)", path, Path(path).stat().st_size / 1e6)

    @classmethod
    def load(cls, path: str) -> "LabelBehaviorStore":
        store = cls()
        with open(path, "rb") as f:
            store.behaviors = pickle.load(f)
        logger.info("LabelBehaviorStore loaded: %d users from %s", len(store.behaviors), path)
        return store


class CollaborativeStore:
    """Item-item co-occurrence matrix with time decay.

    Computes co-occurrence within sliding windows of user history sequences.
    Items that appear closer together in the sequence get higher weights.
    Stored as a scipy sparse matrix for efficient neighbor lookup.
    """

    def __init__(self):
        self.sid2idx: dict[tuple, int] = {}
        self.idx2sid: list[tuple] = []
        self.cooc_matrix = None  # scipy.sparse.csr_matrix

    def build(
        self,
        data_path: str,
        pid2sid_video: dict,
        window_size: int = 20,
        decay_factor: float = 0.9,
        min_freq: int = 3,
        max_vocab: int = 500000,
    ):
        import pyarrow.parquet as pq
        from scipy.sparse import csr_matrix, lil_matrix

        logger.info(
            "Building CollaborativeStore (window=%d, decay=%.2f, min_freq=%d, max_vocab=%d) ...",
            window_size, decay_factor, min_freq, max_vocab,
        )
        pf = pq.ParquetFile(data_path)

        sid_freq = Counter()
        raw_seqs = []

        for batch in pf.iter_batches(batch_size=5000, columns=["hist_video_pid"]):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                hist_pids = row["hist_video_pid"]
                if hist_pids is None:
                    continue
                sid_seq = []
                for pid in hist_pids:
                    pid = int(pid)
                    sid = pid2sid_video.get(pid)
                    if sid:
                        sid_tuple = tuple(sid)
                        sid_freq[sid_tuple] += 1
                        sid_seq.append(sid_tuple)
                if len(sid_seq) >= 2:
                    raw_seqs.append(sid_seq)

        vocab = {
            s for s, c in sid_freq.most_common(max_vocab) if c >= min_freq
        }
        logger.info(
            "Vocabulary after filtering: %d / %d SIDs (min_freq=%d)",
            len(vocab), len(sid_freq), min_freq,
        )

        self.idx2sid = sorted(vocab)
        self.sid2idx = {s: i for i, s in enumerate(self.idx2sid)}
        n = len(self.idx2sid)

        cooc = lil_matrix((n, n), dtype=np.float32)
        progress_interval = max(1, len(raw_seqs) // 10)
        for seq_idx, seq in enumerate(raw_seqs):
            if seq_idx % progress_interval == 0:
                logger.info("  Processing sequence %d / %d ...", seq_idx, len(raw_seqs))
            filtered = [(i, s) for i, s in enumerate(seq) if s in vocab]
            for pos, (orig_i, sid_i) in enumerate(filtered):
                idx_i = self.sid2idx[sid_i]
                for prev_pos in range(max(0, pos - window_size), pos):
                    orig_j, sid_j = filtered[prev_pos]
                    idx_j = self.sid2idx[sid_j]
                    if idx_i == idx_j:
                        continue
                    distance = orig_i - orig_j
                    weight = decay_factor ** distance
                    cooc[idx_i, idx_j] += weight
                    cooc[idx_j, idx_i] += weight

        self.cooc_matrix = csr_matrix(cooc)
        logger.info("CollaborativeStore built: %d items, %d nnz", n, self.cooc_matrix.nnz)

    def query_neighbors(self, sid: list[int] | tuple, top_k: int = 10) -> list[tuple[tuple, float]]:
        sid_tuple = tuple(sid)
        idx = self.sid2idx.get(sid_tuple)
        if idx is None:
            return []

        row = self.cooc_matrix.getrow(idx).toarray().flatten()
        if row.sum() == 0:
            return []

        top_indices = np.argsort(row)[::-1][:top_k]
        results = []
        for i in top_indices:
            if row[i] > 0:
                results.append((self.idx2sid[i], float(row[i])))
        return results

    def save(self, path: str):
        from scipy.sparse import save_npz

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        base = Path(path)
        save_npz(str(base.with_suffix(".npz")), self.cooc_matrix)
        with open(str(base.with_suffix(".meta.pkl")), "wb") as f:
            pickle.dump({
                "sid2idx": self.sid2idx,
                "idx2sid": self.idx2sid,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        total_size = (
            base.with_suffix(".npz").stat().st_size
            + base.with_suffix(".meta.pkl").stat().st_size
        )
        logger.info("CollaborativeStore saved to %s (%.1f MB total)", path, total_size / 1e6)

    @classmethod
    def load(cls, path: str) -> "CollaborativeStore":
        from scipy.sparse import load_npz

        store = cls()
        base = Path(path)
        store.cooc_matrix = load_npz(str(base.with_suffix(".npz")))
        with open(str(base.with_suffix(".meta.pkl")), "rb") as f:
            meta = pickle.load(f)
        store.sid2idx = meta["sid2idx"]
        store.idx2sid = meta["idx2sid"]
        logger.info(
            "CollaborativeStore loaded: %d items, %d nnz from %s",
            len(store.idx2sid), store.cooc_matrix.nnz, path,
        )
        return store


class ItemTextStore:
    """BM25 index over item text descriptions.

    Sources:
      - reco_gsu_caption / reco_target_caption from main data
      - item_understand benchmark (pid → caption answer)

    Only stores PID→SID mappings for items that actually have text.
    Uses rank_bm25 + jieba for Chinese text retrieval.
    """

    def __init__(self):
        self.pid2text: dict[int, str] = {}
        self.pid2sid: dict[int, list[int]] = {}
        self.bm25 = None
        self._corpus_pids: list[int] = []

    def build(
        self,
        data_path: str,
        pid2sid_video: dict,
        pid2sid_product: dict,
        benchmark_dir: Optional[str] = None,
    ):
        import pyarrow.parquet as pq

        logger.info("Building ItemTextStore from %s ...", data_path)
        pf = pq.ParquetFile(data_path)

        for batch in pf.iter_batches(
            batch_size=5000,
            columns=["reco_gsu_caption", "reco_target_caption",
                      "target_video_pid", "hist_video_pid"],
        ):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                captions = row["reco_gsu_caption"]
                target_cap = row["reco_target_caption"]
                target_pids = row["target_video_pid"]

                if captions is not None and target_pids is not None:
                    for i, cap in enumerate(captions):
                        if cap and isinstance(cap, str) and len(cap) > 10:
                            if i < len(target_pids):
                                pid = int(target_pids[i])
                                if pid not in self.pid2text:
                                    self.pid2text[pid] = cap

                if (
                    target_cap
                    and isinstance(target_cap, str)
                    and len(target_cap) > 10
                    and target_pids is not None
                    and len(target_pids) > 0
                ):
                    pid = int(target_pids[0])
                    if pid not in self.pid2text:
                        self.pid2text[pid] = target_cap

        if benchmark_dir:
            iu_path = Path(benchmark_dir) / "item_understand" / "item_understand_test.parquet"
            if iu_path.exists():
                df_iu = pd.read_parquet(str(iu_path))
                for _, row in df_iu.iterrows():
                    meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
                    pid = meta.get("pid")
                    answer = meta.get("answer", "")
                    if pid and answer and len(answer) > 10:
                        self.pid2text[int(pid)] = answer
                logger.info("  Added %d captions from item_understand", len(df_iu))

        all_pid2sid = {**pid2sid_video, **pid2sid_product}
        self.pid2sid = {
            pid: all_pid2sid[pid]
            for pid in self.pid2text
            if pid in all_pid2sid
        }

        logger.info("ItemTextStore: %d items with text, %d with SID mapping", len(self.pid2text), len(self.pid2sid))
        self._build_bm25_index()

    def _build_bm25_index(self):
        if not self.pid2text:
            logger.warning("No text captions available, BM25 index empty")
            return

        import jieba
        from rank_bm25 import BM25Okapi

        self._corpus_pids = list(self.pid2text.keys())
        corpus = []
        for pid in self._corpus_pids:
            text = self.pid2text[pid]
            tokens = list(jieba.cut(text))
            corpus.append(tokens)

        self.bm25 = BM25Okapi(corpus)
        logger.info("BM25 index built over %d documents", len(corpus))

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []

        import jieba
        tokens = list(jieba.cut(query))

        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._corpus_pids[idx], float(scores[idx])))
        return results

    def get_text(self, pid: int) -> Optional[str]:
        return self.pid2text.get(pid)

    def get_sid(self, pid: int) -> Optional[list[int]]:
        return self.pid2sid.get(pid)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "pid2text": self.pid2text,
                "pid2sid": self.pid2sid,
                "corpus_pids": self._corpus_pids,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("ItemTextStore saved to %s (%.1f MB)", path, Path(path).stat().st_size / 1e6)

    @classmethod
    def load(cls, path: str) -> "ItemTextStore":
        store = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        store.pid2text = data["pid2text"]
        store.pid2sid = data["pid2sid"]
        store._corpus_pids = data["corpus_pids"]
        store._build_bm25_index()
        logger.info(
            "ItemTextStore loaded: %d texts, %d pid->sid from %s",
            len(store.pid2text), len(store.pid2sid), path,
        )
        return store
