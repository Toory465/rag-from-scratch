"""Lesson 2b — Is a chunking gain real? Paired bootstrap on per-query differences.
Both configs answer the same 300 queries, so compare them query by query: resampling the
*differences* cancels out per-query difficulty and gives a much tighter CI than two separate ones.
Usage: python -m lessons.l02b_significance   (A=no_chunking, B=sentence_64 by default)
"""
import os

import numpy as np

from rag import chunking as C
from rag.evaluation import ndcg_at_k, recall_at_k

from ._common import data, dense, embedder, run_retriever

CONFIGS = {
    "no_chunking": (C.no_chunking, {}),
    "sentence_64": (C.sentence_window, {"max_words": 64}),
    "sentence_128": (C.sentence_window, {"max_words": 128}),
    "fixed_64_o16": (C.fixed_size, {"size": 64, "overlap": 16}),
}


def paired_bootstrap(diff: np.ndarray, n: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())  # two-sided
    return float(diff.mean()), float(lo), float(hi), float(p)


def show(qid, queries, qrels, titles, runs, label):
    print(f"  [{label}] top-3:")
    for rank, d in enumerate(runs[qid][:3], 1):
        mark = "CORRECT" if d in qrels[qid] else "       "
        print(f"    {rank}. {mark} {d}: {titles[d][:80]}")
    ranks = [runs[qid].index(d) + 1 for d in qrels[qid] if d in runs[qid]]
    print(f"    correct doc rank: {ranks or 'not in top 100'}")


def main():
    a_name, b_name = os.getenv("A", "no_chunking"), os.getenv("B", "sentence_64")
    docs, queries, qrels = data()
    titles = {d.id: d.title for d in docs}
    emb = embedder()

    runs = {}
    for name in (a_name, b_name):
        fn, kw = CONFIGS[name]
        r = dense(C.chunk_corpus(docs, fn, **kw), emb, cache_key=name)
        runs[name] = run_retriever(r.retrieve, queries)

    qids = list(queries)
    for metric, f in (("ndcg@10", lambda r, q: ndcg_at_k(r, qrels[q], 10)),
                      ("recall@100", lambda r, q: recall_at_k(r, qrels[q], 100))):
        a = np.array([f(runs[a_name][q], q) for q in qids])
        b = np.array([f(runs[b_name][q], q) for q in qids])
        d = b - a
        m, lo, hi, p = paired_bootstrap(d)
        verdict = "SIGNIFICANT" if lo > 0 or hi < 0 else "not significant (CI includes 0)"
        print(f"\n{metric}: {a_name}={a.mean():.4f}  {b_name}={b.mean():.4f}")
        print(f"  diff {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}  -> {verdict}")
        print(f"  per query: {b_name} better {int((d > 0).sum())}, worse {int((d < 0).sum())}, "
              f"same {int((d == 0).sum())}")
        if metric == "ndcg@10":
            nd = d

    order = np.argsort(nd)
    for label, i in (("biggest gain", order[-1]), ("biggest loss", order[0])):
        q = qids[i]
        print(f"\n=== {label} for {b_name} (ndcg diff {nd[i]:+.3f}) ===\nQ: {queries[q]}")
        show(q, queries, qrels, titles, runs[a_name], a_name)
        show(q, queries, qrels, titles, runs[b_name], b_name)


if __name__ == "__main__":
    main()
