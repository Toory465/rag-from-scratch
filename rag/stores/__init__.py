from .base import VectorStore
from .faiss_store import FaissStore
from .numpy_store import NumpyStore

__all__ = ["FaissStore", "NumpyStore", "PgVectorStore", "VectorStore"]


def __getattr__(name):  # lazy: psycopg is an optional dependency
    if name == "PgVectorStore":
        from .pgvector_store import PgVectorStore
        return PgVectorStore
    raise AttributeError(name)
