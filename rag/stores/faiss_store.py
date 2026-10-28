"""FAISS: 'flat' = exact; 'hnsw' = graph ANN (what pgvector/Qdrant/etc. also use).

HNSW knobs: M (graph degree, memory/recall), efConstruction (build quality),
efSearch (query-time recall/latency trade-off). Sweep efSearch in lesson 3.
"""
import faiss
import numpy as np


class FaissStore:
    def __init__(self, dim: int, kind: str = "flat", M: int = 32, ef_construction: int = 200,
                 ef_search: int = 64):
        if kind == "flat":
            self.index = faiss.IndexFlatIP(dim)
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        else:
            raise ValueError(kind)
        self.kind = kind
        self.ids: list[str] = []

    def set_ef_search(self, ef: int):
        if self.kind == "hnsw":
            self.index.hnsw.efSearch = ef

    def add(self, ids, vectors):
        self.ids.extend(ids)
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search_batch(self, queries, k):
        D, I = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return [[(self.ids[i], float(d)) for d, i in zip(dr, ir) if i != -1] for dr, ir in zip(D, I)]

    def search(self, query, k):
        return self.search_batch(query[None, :], k)[0]
