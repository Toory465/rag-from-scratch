"""Generation: put retrieved chunks in the prompt, force citations, allow abstention."""
import os

from anthropic import Anthropic

from .schema import Hit

SYSTEM = """You answer questions using ONLY the numbered context passages provided.
Rules:
- Cite every factual sentence with passage numbers in brackets, e.g. [1] or [2][3].
- If the passages do not contain the answer, reply exactly: "I don't know based on the provided context."
- Be concise. Do not use outside knowledge."""


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] (doc {h.doc_id})\n{h.text}" for i, h in enumerate(hits, 1))


class Generator:
    def __init__(self, model: str | None = None, max_tokens: int = 512):
        self.client = Anthropic()
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    def answer(self, question: str, hits: list[Hit]) -> str:
        user = f"Context:\n{format_context(hits)}\n\nQuestion: {question}"
        return self.complete(SYSTEM, user)
