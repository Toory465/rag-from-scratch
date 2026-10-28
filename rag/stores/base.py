from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """query: (dim,) normalised. Returns [(id, cosine_score)] best first."""
        ...
