"""Lesson 2 — Chunking. Same embedder + exact search; only the chunker changes.
SciFact abstracts are short (~200 words), so 'no chunking' is a strong baseline here.
To see chunking matter, also try a long-document corpus (e.g. your own PDFs / internal docs).
Set SEMANTIC=1 to include semantic chunking (embeds every sentence; a few minutes on CPU).
"""
import os

from rag import chunking as C

from ._common import data, dense, embedder, run_retriever, score, table


def main():
    docs, queries, qrels = data()
    emb = embedder()
    configs = {
        "no_chunking": (C.no_chunking, {}),
        "no_chunking_no_title": (C.no_chunking, {"with_title": False}),
        "fixed_64_o16": (C.fixed_size, {"size": 64, "overlap": 16}),
        "fixed_128_o32": (C.fixed_size, {"size": 128, "overlap": 32}),
        "sentence_64": (C.sentence_window, {"max_words": 64}),
        "sentence_128": (C.sentence_window, {"max_words": 128}),
        "recursive_96": (C.recursive, {"max_words": 96}),
    }
    if os.getenv("SEMANTIC"):
        configs["semantic_p90"] = (C.semantic, {"embed_fn": emb.encode_docs, "percentile": 90})

    rows = []
    for name, (fn, kw) in configs.items():
        chunks = C.chunk_corpus(docs, fn, **kw)
        r = dense(chunks, emb, cache_key=name)
        runs = run_retriever(r.retrieve, queries)
        avg = sum(len(c.text.split()) for c in chunks) / len(chunks)
        rows.append({**score(name, runs, qrels), "n_chunks": len(chunks), "avg_words": round(avg)})
        print(f"done {name}")
    table(rows)


if __name__ == "__main__":
    main()
