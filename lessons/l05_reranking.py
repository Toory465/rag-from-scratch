"""Lesson 5 — Re-ranking: cheap first stage (top-N) -> cross-encoder -> top-10.
Watch two things: nDCG@10 gain, and that reranking can never fix recall@N of stage 1.
"""
import time

from rag.rerank import CrossEncoderReranker
from rag.retrieval import HybridRetriever, to_doc_ranking

from ._common import bm25, data, default_chunks, dense, run_retriever, score, table


def main():
    _, queries, qrels = data()
    chunks = default_chunks()
    hybrid = HybridRetriever([bm25(chunks), dense(chunks)])
    reranker = CrossEncoderReranker()

    rows = [score("hybrid (no rerank)", run_retriever(hybrid.retrieve, queries), qrels)]
    for n in (20, 50, 100):
        t = time.perf_counter()
        runs = {qid: to_doc_ranking(reranker.rerank(q, hybrid.retrieve(q, n)))
                for qid, q in queries.items()}
        ms = (time.perf_counter() - t) / len(queries) * 1000
        rows.append({**score(f"hybrid top{n} -> rerank", runs, qrels), "ms/query": round(ms)})
    table(rows)


if __name__ == "__main__":
    main()
