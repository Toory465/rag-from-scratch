from dataclasses import dataclass, field


@dataclass
class Doc:
    id: str
    text: str
    title: str = ""


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float
