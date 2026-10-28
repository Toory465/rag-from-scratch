"""Lesson 4 — Retrieval pipelines: BM25 vs dense vs hybrid (RRF), with bootstrap CIs."""
from rag.evaluation import bootstrap_ci, ndcg_at_k
from rag.retrieval import HybridRetriever

from ._common import bm25, data, default_chunks, dense, run_retriever, score, table


def main():
    _, queries, qrels = data()
    chunks = default_chunks()
    sparse, dns = bm25(chunks), dense(chunks)
    retrievers = {
        "bm25": sparse,
        "dense": dns,
        "hybrid_rrf": HybridRetriever([sparse, dns]),
        "hybrid_rrf_dense2x": HybridRetriever([sparse, dns], weights=[1.0, 2.0]),
    }
    rows, per_q = [], {}
    for name, r in retrievers.items():
        runs = run_retriever(r.retrieve, queries)
        rows.append(score(name, runs, qrels))
        per_q[name] = {q: ndcg_at_k(runs[q], qrels[q], 10) for q in runs}
    table(rows)

    for name, v in per_q.items():
        m, lo, hi = bootstrap_ci(list(v.values()))
        print(f"{name:22s} nDCG@10 {m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    wins = sum(per_q["bm25"][q] > per_q["dense"][q] for q in queries)
    print(f"\nBM25 beats dense on {wins}/{len(queries)} queries — read some of them.")


if __name__ == "__main__":
    main()
