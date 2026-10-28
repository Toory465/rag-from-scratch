from dataclasses import dataclass, field

from .schema import Hit


@dataclass
class RAGResult:
    question: str
    answer: str | None
    contexts: list[Hit]
    candidates: list[Hit] = field(default_factory=list)


class RAGPipeline:
    """retrieve(k_retrieve) -> [rerank] -> top k_final -> [generate]"""

    def __init__(self, retriever, reranker=None, generator=None, k_retrieve: int = 50,
                 k_final: int = 5):
        self.retriever, self.reranker, self.generator = retriever, reranker, generator
        self.k_retrieve, self.k_final = k_retrieve, k_final

    def retrieve(self, question: str) -> tuple[list[Hit], list[Hit]]:
        cands = self.retriever.retrieve(question, self.k_retrieve)
        final = (self.reranker.rerank(question, cands, self.k_final) if self.reranker
                 else cands[: self.k_final])
        return cands, final

    def run(self, question: str) -> RAGResult:
        cands, final = self.retrieve(question)
        ans = self.generator.answer(question, final) if self.generator else None
        return RAGResult(question, ans, final, cands)
