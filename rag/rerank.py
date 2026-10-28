"""Cross-encoder re-ranking. A bi-encoder embeds query and doc separately (fast, indexable);
a cross-encoder reads them *together* (slow, much more accurate). Standard pattern:
retrieve top-50..100 cheaply, re-rank with the cross-encoder, keep top-5 for the LLM.
"""
import os

from .schema import Hit


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
        self.model = CrossEncoder(self.model_name, device=device, max_length=512)

    def rerank(self, query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, h.text) for h in hits], batch_size=32)
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        out = [Hit(h.chunk_id, h.doc_id, h.text, float(s)) for h, s in ranked]
        return out[:top_k] if top_k else out
