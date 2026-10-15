"""Tiny fake RAG apps, each with one known bug, so the tests know what Kensa should find."""
import random
import re

from kensa import Output

STOP = {"the", "a", "an", "is", "of", "what", "how", "many", "much", "in", "to", "for", "are", "me", "tell",
        "quick", "question", "number", "amount", "does", "do", "at", "which", "it", "and", "on"}


def words(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP}


def best_sentence(question, docs):
    sents = [s.strip() for d in docs for s in re.split(r"(?<=[.!?])\s+", d) if s.strip()]
    scored = sorted(sents, key=lambda s: -len(words(s) & words(question)))
    if not scored or len(words(scored[0]) & words(question)) < 2:
        return None
    return scored[0]


def good_app(question, docs):
    """Extractive: returns the sentence that matches the question best, or refuses."""
    s = best_sentence(question, docs)
    return Output("I don't know based on these documents.", refused=True) if s is None else Output(s)


MEMORY = {"vacation": "Employees get 24 days of paid vacation per year."}


def memory_app(question, docs):
    """Answers some questions from 'training data' instead of the documents."""
    for key, fact in MEMORY.items():
        if key in question.lower():
            return Output(fact)
    return good_app(question, docs)


def first_doc_app(question, docs):
    """Only reads the first document (a 'lost in the middle' bug, exaggerated)."""
    return good_app(question, docs[:1])


def make_flaky_app(p_bad=0.5, seed=1):
    rng = random.Random(seed)

    def app(question, docs):
        out = good_app(question, docs)
        if not out.refused and rng.random() < p_bad:
            return Output(out.answer + " This policy was introduced in 2019.")
        return out

    return app
