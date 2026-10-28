"""Lesson 7 — Evaluate the full system.
Retrieval metrics on all 300 queries; LLM-judge generation metrics on a sample (N, default 30).
Writes results/eval_<timestamp>.csv so you can diff configurations over time.
"""
import os
import time
from pathlib import Path

import pandas as pd

from rag.evaluation import context_precision, judge_faithfulness, judge_relevance
from rag.generate import Generator, format_context
from rag.pipeline import RAGPipeline
from rag.rerank import CrossEncoderReranker
from rag.retrieval import HybridRetriever, to_doc_ranking

from ._common import bm25, data, default_chunks, dense, score, table


def main():
    n = int(os.getenv("N", "30"))
    _, queries, qrels = data()
    chunks = default_chunks()
    gen = Generator()
    pipe = RAGPipeline(HybridRetriever([bm25(chunks), dense(chunks)]),
                       CrossEncoderReranker(), gen, k_retrieve=50, k_final=5)

    runs = {qid: to_doc_ranking(pipe.retrieve(q)[1]) for qid, q in queries.items()}
    table([score("pipeline final top-5 (doc-level)", runs, qrels)])

    rows = []
    for qid, q in list(queries.items())[:n]:
        res = pipe.run(q)
        ctx = format_context(res.contexts)
        f = judge_faithfulness(gen, q, res.answer, ctx)
        rows.append({
            "qid": qid, "question": q, "answer": res.answer,
            "context_precision": context_precision([h.doc_id for h in res.contexts], qrels[qid]),
            "hit@5": int(any(h.doc_id in qrels[qid] for h in res.contexts)),
            **f, "answer_relevance": judge_relevance(gen, q, res.answer),
        })
    df = pd.DataFrame(rows)
    cols = ["context_precision", "hit@5", "faithfulness", "answer_relevance", "abstained"]
    print(df[cols].mean(numeric_only=True).round(3).to_string())
    print("\nAbstained despite gold doc in context (retrieval OK, generation too cautious):",
          int(((df.abstained) & (df["hit@5"] == 1)).sum()))
    print("Answered without gold doc in context (hallucination risk):",
          int(((~df.abstained) & (df["hit@5"] == 0)).sum()))

    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}. Hand-label ~20 rows and check judge agreement before trusting it.")


if __name__ == "__main__":
    main()
