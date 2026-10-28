import numpy as np

from rag import chunking as C
from rag.schema import Doc

TEXT = " ".join(f"Sentence number {i} talks about topic {i % 3}." for i in range(40))
DOC = Doc(id="d1", title="T", text=TEXT)


def test_fixed_size_overlap_and_coverage():
    ch = C.fixed_size(DOC, size=50, overlap=10, with_title=False)
    words = TEXT.split()
    assert ch[0].text.split() == words[:50]
    assert ch[1].text.split()[:10] == words[40:50]
    assert ch[-1].text.split()[-1] == words[-1]


def test_sentence_window_respects_limit_and_sentences():
    ch = C.sentence_window(DOC, max_words=30, overlap_sents=0, with_title=False)
    assert all(len(c.text.split()) <= 30 for c in ch)
    assert all(c.text.endswith(".") for c in ch)


def test_recursive_limit():
    ch = C.recursive(DOC, max_words=25, with_title=False)
    assert all(len(c.text.split()) <= 25 for c in ch)
    assert " ".join(c.text for c in ch).count("Sentence") == 40


def test_semantic_splits_on_topic_shift():
    def fake_embed(sents):
        v = np.array([[1.0, 0.0] if "topic 0" in s else [0.0, 1.0] for s in sents])
        return v
    ch = C.semantic(DOC, fake_embed, percentile=50, with_title=False)
    assert len(ch) > 1


def test_ids_and_title():
    ch = C.fixed_size(DOC, size=100, overlap=0)
    assert ch[0].id == "d1::0" and ch[0].text.startswith("T\n")
