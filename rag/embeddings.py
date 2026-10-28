"""Embeddings: map text -> dense vector so that semantic similarity ~ cosine similarity.

Key ideas to verify in lesson 1:
- Normalise vectors -> inner product == cosine, which is what most ANN indexes assume.
- Many retrieval models are *asymmetric*: queries get an instruction prefix, docs don't.
- Max sequence length silently truncates long chunks (bge-small: 512 tokens).
"""
import hashlib
import os
from pathlib import Path

import numpy as np

QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-small-v2": "query: ",
    "intfloat/e5-base-v2": "query: ",
}
DOC_PREFIXES = {"intfloat/e5-small-v2": "passage: ", "intfloat/e5-base-v2": "passage: "}

CACHE = Path(".cache/embeddings")


class Embedder:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        self.model = SentenceTransformer(self.model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.q_prefix = QUERY_PREFIXES.get(self.model_name, "")
        self.d_prefix = DOC_PREFIXES.get(self.model_name, "")

    def _encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        v = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=len(texts) > 500, convert_to_numpy=True,
        )
        return v.astype(np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.q_prefix + t for t in texts])

    def encode_docs(self, texts: list[str], cache_key: str | None = None) -> np.ndarray:
        """Disk-cached: re-embedding 5k docs on every run kills iteration speed."""
        if cache_key:
            h = hashlib.md5(f"{self.model_name}|{cache_key}|{len(texts)}".encode()).hexdigest()
            path = CACHE / f"{h}.npy"
            if path.exists():
                return np.load(path)
        v = self._encode([self.d_prefix + t for t in texts])
        if cache_key:
            CACHE.mkdir(parents=True, exist_ok=True)
            np.save(path, v)
        return v


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T
