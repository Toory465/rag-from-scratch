"""Exact brute-force search. O(N*d) per query; the ground truth for ANN recall."""
import numpy as np


class NumpyStore:
    def __init__(self):
        self.ids: list[str] = []
        self.mat: np.ndarray | None = None

    def add(self, ids, vectors):
        self.ids.extend(ids)
        v = vectors.astype(np.float32)
        self.mat = v if self.mat is None else np.vstack([self.mat, v])

    def search(self, query, k):
        scores = self.mat @ query
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]

    def search_batch(self, queries, k):
        return [self.search(q, k) for q in queries]
