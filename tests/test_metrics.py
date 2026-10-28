import math

import numpy as np

from rag.evaluation import mrr_at_k, ndcg_at_k, recall_at_k
from rag.retrieval import BM25Retriever, rrf, to_doc_ranking
from rag.schema import Chunk, Hit
from rag.stores import FaissStore, NumpyStore

REL = {"a": 1, "c": 1}


def test_recall_mrr():
    assert recall_at_k(["x", "a", "y"], REL, 2) == 0.5
    assert mrr_at_k(["x", "a"], REL, 10) == 0.5
    assert mrr_at_k(["x", "y"], REL, 10) == 0.0


def test_ndcg_perfect_and_partial():
    assert ndcg_at_k(["a", "c", "x"], REL, 10) == 1.0
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert abs(ndcg_at_k(["x", "a"], REL, 10) - expected) < 1e-9


def test_rrf_prefers_consensus():
    h = lambda c: Hit(c, c, c, 0.0)
    fused = rrf([[h("a"), h("b"), h("c")], [h("c"), h("a"), h("d")]])
    assert fused[0].chunk_id == "a"


def test_doc_ranking_dedup():
    hits = [Hit("d1::0", "d1", "", 1), Hit("d2::0", "d2", "", 1), Hit("d1::1", "d1", "", 1)]
    assert to_doc_ranking(hits) == ["d1", "d2"]


def test_stores_agree():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 32)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ids = [str(i) for i in range(500)]
    a, b = NumpyStore(), FaissStore(32, "flat")
    a.add(ids, X)
    b.add(ids, X)
    assert [i for i, _ in a.search(X[7], 5)] == [i for i, _ in b.search(X[7], 5)]
    assert a.search(X[7], 1)[0][0] == "7"


def test_bm25_exact_term():
    chunks = [Chunk("1", "1", "insulin resistance in muscle"), Chunk("2", "2", "hotel pricing"),
              Chunk("3", "3", "tumour growth in mice")]
    assert BM25Retriever(chunks).retrieve("insulin", 1)[0].doc_id == "1"
