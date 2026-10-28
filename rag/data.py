"""BEIR SciFact: 5,183 abstracts, 300 test queries with human relevance labels (qrels).

Small enough to iterate on a laptop, labelled enough to measure retrieval properly.
"""
from collections import defaultdict

from datasets import load_dataset

from .schema import Doc


def load_scifact(split: str = "test"):
    corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/scifact-qrels", split=split)

    docs = [Doc(id=str(r["_id"]), title=r["title"], text=r["text"]) for r in corpus_ds]

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for r in qrels_ds:
        qrels[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])

    queries = {str(r["_id"]): r["text"] for r in queries_ds if str(r["_id"]) in qrels}
    return docs, queries, dict(qrels)
