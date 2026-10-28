"""Lesson 3 — Vector databases. Exact vs ANN: measure the recall/latency curve.
ANN recall@10 = overlap between ANN top-10 and exact top-10 (not relevance — that's lesson 4).
If DATABASE_URL is set and `make db` is running, pgvector is benchmarked too.
"""
import os
import time

import numpy as np

from rag.stores import FaissStore

from ._common import data, default_chunks, embedder, table


def ann_recall(exact, approx, k=10):
    return float(np.mean([len({i for i, _ in a[:k]} & {i for i, _ in e[:k]}) / k
                          for a, e in zip(approx, exact)]))


def bench(store, Q, k=10):
    t = time.perf_counter()
    res = store.search_batch(Q, k)
    return res, (time.perf_counter() - t) / len(Q) * 1000


def main():
    _, queries, _ = data()
    emb = embedder()
    chunks = default_chunks()
    X = emb.encode_docs([c.text for c in chunks], cache_key="nochunk")
    ids = [c.id for c in chunks]
    Q = emb.encode_queries(list(queries.values()))

    flat = FaissStore(emb.dim, "flat")
    flat.add(ids, X)
    exact, ms = bench(flat, Q)
    rows = [{"config": "faiss_flat (exact)", "ann_recall@10": 1.0, "ms/query": round(ms, 3)}]

    t = time.perf_counter()
    hnsw = FaissStore(emb.dim, "hnsw", M=32)
    hnsw.add(ids, X)
    print(f"HNSW build: {time.perf_counter() - t:.2f}s for {len(ids)} vectors")
    for ef in (8, 16, 32, 64, 128):
        hnsw.set_ef_search(ef)
        res, ms = bench(hnsw, Q)
        rows.append({"config": f"faiss_hnsw ef={ef}", "ann_recall@10": round(ann_recall(exact, res), 4),
                     "ms/query": round(ms, 3)})

    if os.getenv("DATABASE_URL"):
        try:
            from rag.stores import PgVectorStore
            pg = PgVectorStore(emb.dim)
            pg.add(ids, X)
            pg.build_index()
            for ef in (16, 40, 100):
                pg.set_ef_search(ef)
                res, ms = bench(pg, Q)
                rows.append({"config": f"pgvector hnsw ef={ef}",
                             "ann_recall@10": round(ann_recall(exact, res), 4), "ms/query": round(ms, 3)})
        except Exception as e:  # noqa: BLE001
            print(f"pgvector skipped: {e}")
    table(rows)
    print("At 5k vectors exact search is already fast; ANN matters at 10^6+. "
          "Try X = np.tile(X, (200, 1)) + noise to feel it.")


if __name__ == "__main__":
    main()
