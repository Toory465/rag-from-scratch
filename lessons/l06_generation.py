"""Lesson 6 — Close the loop: retrieve -> rerank -> generate with citations."""
import itertools

from rag.generate import Generator
from rag.pipeline import RAGPipeline
from rag.rerank import CrossEncoderReranker
from rag.retrieval import HybridRetriever

from ._common import bm25, data, default_chunks, dense


def main():
    _, queries, qrels = data()
    chunks = default_chunks()
    pipe = RAGPipeline(HybridRetriever([bm25(chunks), dense(chunks)]),
                       CrossEncoderReranker(), Generator(), k_retrieve=50, k_final=5)
    extra = ["What is the boiling point of tungsten?"]  # should abstain
    for qid, q in itertools.chain(itertools.islice(queries.items(), 3), [("ood", extra[0])]):
        res = pipe.run(q)
        print("=" * 90 + f"\nQ: {q}\nretrieved docs: {[h.doc_id for h in res.contexts]}")
        print(f"gold docs:      {list(qrels.get(qid, {}))}\n\n{res.answer}\n")


if __name__ == "__main__":
    main()
