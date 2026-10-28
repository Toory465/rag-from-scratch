# rag-from-scratch

Learning Retrieval-Augmented Generation by building every layer and **measuring** it.
No LangChain/LlamaIndex: each component is ~50–100 lines so the mechanics stay visible.

**Dataset:** [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) — 5,183 scientific abstracts,
300 test claims with human relevance labels. Labels are what make this a learning project
rather than a demo: every design choice gets a number.

```
query ─► [BM25] ─┐
                 ├─► RRF fusion ─► top-50 ─► cross-encoder rerank ─► top-5 ─► Claude (cited answer)
query ─► [dense]─┘                                                              │
          ▲                                                                     ▼
   embeddings + vector store (numpy / FAISS / pgvector)            eval: retrieval + LLM-judge
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
make install                 # pip install -e ".[pg,dev]"
cp .env.example .env         # add ANTHROPIC_API_KEY (lessons 6–7 only)
make db                      # optional: pgvector in Docker on :5433
make test
```

## Curriculum

| # | Concept | Run | Module |
|---|---------|-----|--------|
| 1 | Embeddings | `make l1` | `rag/embeddings.py` |
| 2 | Chunking strategies | `make l2` | `rag/chunking.py` |
| 3 | Vector databases | `make l3` | `rag/stores/` |
| 4 | Retrieval pipelines | `make l4` | `rag/retrieval.py` |
| 5 | Re-ranking | `make l5` | `rag/rerank.py` |
| 6 | Generation | `make l6` | `rag/generate.py`, `rag/pipeline.py` |
| 7 | Evaluation | `make l7` | `rag/evaluation.py` |

### 0. What RAG is
An LLM answers from its weights (stale, unverifiable). RAG retrieves evidence at query time and
conditions generation on it: **answer quality ≤ retrieval quality**. Most RAG failures are
retrieval failures, which is why lessons 1–5 are about retrieval and only 6 is about the LLM.

### 1. Embeddings
Bi-encoder maps text → normalised vector; relevance ≈ cosine.
- [ ] Measure the anisotropy baseline (cosine between *unrelated* docs is not 0).
- [ ] Show negation/number blindness (`"5 mg"` vs `"50 mg"`).
- [ ] Query instruction prefix on/off; check how many docs exceed `max_seq_length`.
- [ ] Swap `EMBED_MODEL` (bge-small → bge-base, e5-base-v2) and re-run lesson 4.

### 2. Chunking
Fixed-size, sentence-window, recursive, semantic (break at embedding-distance spikes).
- [ ] Compare nDCG@10 / Recall@100 vs chunk size and overlap.
- [ ] Title prepended vs not ("contextual" chunks).
- [ ] SciFact abstracts are short, so `no_chunking` is strong. Repeat on a long-document corpus
      of your own — that's where chunking choices actually separate.

### 3. Vector databases
Exact (numpy, FAISS Flat) vs ANN (FAISS HNSW, pgvector HNSW).
- [ ] Plot ANN-recall@10 vs ms/query while sweeping `efSearch`.
- [ ] Scale to ~1M vectors (tile + noise) to see where exact search breaks.
- [ ] pgvector: add a metadata filter (`WHERE year > …`) and observe filtered-HNSW recall loss.

### 4. Retrieval pipelines
BM25 (lexical), dense (semantic), hybrid via Reciprocal Rank Fusion.
- [ ] Bootstrap CIs before declaring a winner (300 queries is small).
- [ ] Read queries where BM25 beats dense; categorise the failure.
- [ ] Tune RRF weights; try query rewriting / HyDE as an extension.

### 5. Re-ranking
Cross-encoder scores (query, passage) jointly over the top-N candidates.
- [ ] nDCG@10 gain vs N (20/50/100) and latency cost.
- [ ] Note: rerank can't recover what stage 1 missed → Recall@N of stage 1 is the ceiling.

### 6. Generation
Numbered context, mandatory citations, explicit abstention.
- [ ] Out-of-domain question should abstain.
- [ ] Vary `k_final` (3/5/10): more context ≠ better answers.

### 7. Evaluation
Two layers:
- **Retrieval** (labelled, deterministic): Recall@k, MRR@10, nDCG@10.
- **Generation** (LLM-as-judge): faithfulness (claim-level support), answer relevance,
  context precision, abstention behaviour.
- [ ] Hand-label 20 answers; compute judge agreement before trusting the judge.
- [ ] Build a small labelled set on your own domain (synthetic Q generation + human review).

## Layout
```
rag/            library: data, embeddings, chunking, stores/, retrieval, rerank, generate, evaluation
lessons/        one runnable script per concept (python -m lessons.l04_retrieval)
tests/          unit tests for chunkers, metrics, fusion, stores
results/        eval CSVs (git-ignored)
```

## Results log
Record your numbers here as you go (config → nDCG@10, Recall@100, latency).

| Date | Config | nDCG@10 | Recall@100 | Notes |
|------|--------|---------|------------|-------|
|      |        |         |            |       |
