"""Retrievers: sparse (BM25), dense (embeddings + vector store), hybrid (RRF).

BM25 wins on exact terms (gene names, IDs, acronyms); dense wins on paraphrase.
Reciprocal Rank Fusion combines *ranks*, so you don't have to calibrate score scales.
"""
import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from .schema import Chunk, Hit

_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "which with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


class _Base:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}

    def _hits(self, pairs) -> list[Hit]:
        return [Hit(cid, self.by_id[cid].doc_id, self.by_id[cid].text, s) for cid, s in pairs]


class BM25Retriever(_Base):
    def __init__(self, chunks, k1: float = 1.2, b: float = 0.75):
        super().__init__(chunks)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks], k1=k1, b=b)

    def retrieve(self, query: str, k: int = 10) -> list[Hit]:
        s = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-s)[:k]
        return self._hits((self.chunks[i].id, float(s[i])) for i in top)


class DenseRetriever(_Base):
    def __init__(self, chunks, embedder, store):
        super().__init__(chunks)
        self.embedder, self.store = embedder, store

    @classmethod
    def build(cls, chunks, embedder, store, cache_key: str | None = None):
        vecs = embedder.encode_docs([c.text for c in chunks], cache_key=cache_key)
        store.add([c.id for c in chunks], vecs)
        return cls(chunks, embedder, store)

    def retrieve(self, query: str, k: int = 10) -> list[Hit]:
        q = self.embedder.encode_queries([query])[0]
        return self._hits(self.store.search(q, k))


class HybridRetriever:
    def __init__(self, retrievers: list, weights: list[float] | None = None, rrf_k: int = 60,
                 depth: int = 100):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.rrf_k, self.depth = rrf_k, depth

    def retrieve(self, query: str, k: int = 10) -> list[Hit]:
        return rrf([r.retrieve(query, self.depth) for r in self.retrievers],
                   self.weights, self.rrf_k)[:k]


def rrf(rankings: list[list[Hit]], weights: list[float] | None = None, k: int = 60) -> list[Hit]:
    weights = weights or [1.0] * len(rankings)
    score: dict[str, float] = defaultdict(float)
    keep: dict[str, Hit] = {}
    for w, ranking in zip(weights, rankings):
        for rank, h in enumerate(ranking, start=1):
            score[h.chunk_id] += w / (k + rank)
            keep.setdefault(h.chunk_id, h)
    order = sorted(score, key=score.get, reverse=True)
    return [Hit(c, keep[c].doc_id, keep[c].text, score[c]) for c in order]


def to_doc_ranking(hits: list[Hit]) -> list[str]:
    """Chunk hits -> doc ids (first occurrence = max-score aggregation). qrels are doc-level."""
    seen, out = set(), []
    for h in hits:
        if h.doc_id not in seen:
            seen.add(h.doc_id)
            out.append(h.doc_id)
    return out
