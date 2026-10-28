"""RAG evaluation in two layers.

1. Retrieval (needs labelled qrels, cheap, deterministic): Recall@k, MRR@k, nDCG@k.
   Tune chunking / embeddings / hybrid weights / reranker here first.
2. Generation (LLM-as-judge, noisy, costs money): faithfulness (is every claim supported by
   the retrieved context?) and answer relevance. Spot-check the judge against your own labels.
"""
import json
import math
import re

import numpy as np
import pandas as pd


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    relevant = {d for d, g in rel.items() if g > 0}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def mrr_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k], 1):
        if rel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_rankings(runs: dict[str, list[str]], qrels: dict[str, dict[str, int]],
                      ks=(1, 5, 10, 100)) -> dict[str, float]:
    """runs: query_id -> ranked doc ids."""
    rows = []
    for qid, ranked in runs.items():
        rel = qrels.get(qid, {})
        r = {}
        for k in ks:
            r[f"recall@{k}"] = recall_at_k(ranked, rel, k)
        r["mrr@10"] = mrr_at_k(ranked, rel, 10)
        r["ndcg@10"] = ndcg_at_k(ranked, rel, 10)
        rows.append(r)
    return pd.DataFrame(rows).mean().round(4).to_dict()


def bootstrap_ci(per_query: list[float], n: int = 2000, alpha: float = 0.05, seed: int = 0):
    """300 queries is not many: report CIs before claiming one config beats another."""
    rng = np.random.default_rng(seed)
    a = np.asarray(per_query)
    means = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return float(a.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# ---------- LLM-as-judge ----------

FAITH_SYS = """You are a strict evaluator. Given CONTEXT and an ANSWER, split the answer into atomic
factual claims and mark each as supported or unsupported by the context alone.
Respond with JSON only: {"claims": [{"claim": str, "supported": bool}], "abstained": bool}
Set abstained=true if the answer says it doesn't know."""

REL_SYS = """Rate how directly the ANSWER addresses the QUESTION, from 1 (irrelevant) to 5 (fully
answers it). Ignore whether it is factually correct. Respond with JSON only: {"score": int}"""


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


def judge_faithfulness(gen, question: str, answer: str, context: str) -> dict:
    out = _json(gen.complete(FAITH_SYS, f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"))
    claims = out.get("claims", [])
    score = (sum(c.get("supported", False) for c in claims) / len(claims)) if claims else None
    return {"faithfulness": score, "abstained": out.get("abstained", False), "n_claims": len(claims)}


def judge_relevance(gen, question: str, answer: str) -> float | None:
    s = _json(gen.complete(REL_SYS, f"QUESTION:\n{question}\n\nANSWER:\n{answer}")).get("score")
    return (s - 1) / 4 if isinstance(s, (int, float)) else None


def context_precision(final_doc_ids: list[str], rel: dict[str, int]) -> float:
    """Fraction of passages sent to the LLM that are actually relevant (noise in the prompt)."""
    return sum(rel.get(d, 0) > 0 for d in final_doc_ids) / len(final_doc_ids) if final_doc_ids else 0.0
