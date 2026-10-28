"""Chunking strategies. Each returns list[Chunk]; chunk ids are '{doc_id}::{i}'.

Trade-off you are measuring: small chunks -> precise embeddings, but lose context;
large chunks -> context preserved, but the vector averages over several topics.
Lengths are in whitespace words (a cheap token proxy: ~1.3 tokens/word for English).
"""
import re
from collections.abc import Callable

import numpy as np

from .schema import Chunk, Doc

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(\[])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


def _make(doc: Doc, pieces: list[str], with_title: bool) -> list[Chunk]:
    prefix = f"{doc.title}\n" if with_title and doc.title else ""
    return [
        Chunk(id=f"{doc.id}::{i}", doc_id=doc.id, text=prefix + p, meta={"pos": i})
        for i, p in enumerate(pieces) if p.strip()
    ]


def no_chunking(doc: Doc, with_title: bool = True) -> list[Chunk]:
    return _make(doc, [doc.text], with_title)


def fixed_size(doc: Doc, size: int = 128, overlap: int = 32, with_title: bool = True) -> list[Chunk]:
    assert 0 <= overlap < size
    words = doc.text.split()
    step = size - overlap
    pieces = [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]
    return _make(doc, pieces, with_title)


def sentence_window(doc: Doc, max_words: int = 128, overlap_sents: int = 1,
                    with_title: bool = True) -> list[Chunk]:
    """Pack whole sentences up to max_words; carry `overlap_sents` into the next chunk."""
    sents = split_sentences(doc.text)
    pieces, cur = [], []
    for s in sents:
        if cur and sum(len(x.split()) for x in cur) + len(s.split()) > max_words:
            pieces.append(" ".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents else []
        cur.append(s)
    if cur:
        pieces.append(" ".join(cur))
    return _make(doc, pieces, with_title)


def recursive(doc: Doc, max_words: int = 128, separators=("\n\n", "\n", ". ", " "),
              with_title: bool = True) -> list[Chunk]:
    """LangChain-style: split on the coarsest separator that gets pieces under the limit."""

    def _split(text: str, seps: tuple[str, ...]) -> list[str]:
        if len(text.split()) <= max_words or not seps:
            return [text]
        sep, rest = seps[0], seps[1:]
        parts = [p for p in text.split(sep) if p.strip()]
        if len(parts) == 1:
            return _split(text, rest)
        out, buf = [], ""
        for p in parts:
            cand = f"{buf}{sep}{p}" if buf else p
            if len(cand.split()) <= max_words:
                buf = cand
            else:
                if buf:
                    out.append(buf)
                buf = p if len(p.split()) <= max_words else ""
                if not buf:
                    out.extend(_split(p, rest))
        if buf:
            out.append(buf)
        return out

    return _make(doc, _split(doc.text, tuple(separators)), with_title)


def semantic(doc: Doc, embed_fn: Callable[[list[str]], np.ndarray], percentile: float = 90,
             max_words: int = 256, with_title: bool = True) -> list[Chunk]:
    """Break where adjacent-sentence cosine distance is in the top (100-percentile)%."""
    sents = split_sentences(doc.text)
    if len(sents) <= 2:
        return _make(doc, [" ".join(sents)], with_title)
    v = embed_fn(sents)
    dist = 1 - np.sum(v[:-1] * v[1:], axis=1)  # assumes normalised vectors
    thr = np.percentile(dist, percentile)
    pieces, cur = [], [sents[0]]
    for i, s in enumerate(sents[1:]):
        too_long = sum(len(x.split()) for x in cur) + len(s.split()) > max_words
        if dist[i] >= thr or too_long:
            pieces.append(" ".join(cur))
            cur = []
        cur.append(s)
    pieces.append(" ".join(cur))
    return _make(doc, pieces, with_title)


def chunk_corpus(docs: list[Doc], strategy: Callable[..., list[Chunk]], **kw) -> list[Chunk]:
    return [c for d in docs for c in strategy(d, **kw)]
