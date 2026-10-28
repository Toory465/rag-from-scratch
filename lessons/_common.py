import time
from contextlib import contextmanager
from functools import lru_cache

import pandas as pd

from rag.chunking import chunk_corpus, no_chunking
from rag.data import load_scifact
from rag.embeddings import Embedder
from rag.evaluation import evaluate_rankings
from rag.retrieval import BM25Retriever, DenseRetriever, to_doc_ranking
from rag.stores import NumpyStore


@lru_cache
def data():
    return load_scifact("test")


@lru_cache
def embedder(name: str | None = None):
    return Embedder(name)


def dense(chunks, emb=None, cache_key="nochunk", store=None):
    emb = emb or embedder()
    return DenseRetriever.build(chunks, emb, store or NumpyStore(), cache_key=cache_key)


def bm25(chunks):
    return BM25Retriever(chunks)


def default_chunks():
    docs, _, _ = data()
    return chunk_corpus(docs, no_chunking)


def run_retriever(retrieve_fn, queries: dict[str, str], k: int = 100) -> dict[str, list[str]]:
    """Fetch extra chunks so that after doc-dedup we still have ~k docs."""
    return {qid: to_doc_ranking(retrieve_fn(q, k * 3))[:k] for qid, q in queries.items()}


def score(name, runs, qrels) -> dict:
    return {"config": name, **evaluate_rankings(runs, qrels)}


def table(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index("config")
    print("\n" + df.to_string() + "\n")
    return df


@contextmanager
def timer(label: str):
    t = time.perf_counter()
    yield
    print(f"[{label}] {time.perf_counter() - t:.2f}s")
