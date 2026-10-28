"""Lesson 1 — Embeddings.
Questions to answer:
  * Does cosine similarity track meaning? Where does it fail (negation, numbers)?
  * What does the query instruction prefix buy you?
  * Anisotropy: what's the *baseline* cosine between unrelated texts? (Not 0.)
"""
import numpy as np

from rag.embeddings import cosine

from ._common import data, embedder

PAIRS = [
    ("The drug reduced tumour growth in mice.", "Tumour size decreased in treated mice."),
    ("The drug reduced tumour growth in mice.", "The drug did not reduce tumour growth in mice."),
    ("Dosage was 5 mg per day.", "Dosage was 50 mg per day."),
    ("Hotel occupancy rose in July.", "Tumour size decreased in treated mice."),
]


def main():
    emb = embedder()
    print(f"model={emb.model_name} dim={emb.dim} max_seq={emb.model.max_seq_length}")

    for a, b in PAIRS:
        va, vb = emb.encode_docs([a, b])
        print(f"{float(va @ vb):.3f}  | {a}  <->  {b}")
    print("-> negation and numbers barely move the vector. Remember this when BM25 wins.\n")

    docs, queries, qrels = data()
    rng = np.random.default_rng(0)
    sample = [docs[i].text for i in rng.choice(len(docs), 300, replace=False)]
    v = emb.encode_docs(sample)
    sims = v @ v.T
    off = sims[~np.eye(len(v), dtype=bool)]
    print(f"Unrelated-doc cosine: mean={off.mean():.3f} p5={np.quantile(off, .05):.3f} "
          f"p95={np.quantile(off, .95):.3f}  (anisotropy baseline)\n")

    qid = next(iter(queries))
    q, rel_doc = queries[qid], next(iter(qrels[qid]))
    doc = next(d for d in docs if d.id == rel_doc)
    with_p = float(emb.encode_queries([q])[0] @ emb.encode_docs([doc.text])[0])
    without_p = float(emb.encode_docs([q])[0] @ emb.encode_docs([doc.text])[0])
    print(f"Query: {q}\nrelevant-doc cosine with prefix={with_p:.3f} without={without_p:.3f}")

    lens = [len(emb.model.tokenizer(d.text)["input_ids"]) for d in docs[:1000]]
    print(f"Token length p50={np.median(lens):.0f} p95={np.quantile(lens, .95):.0f} "
          f"truncated={np.mean(np.array(lens) > emb.model.max_seq_length):.1%}")
    _ = cosine  # handy for your own experiments


if __name__ == "__main__":
    main()
