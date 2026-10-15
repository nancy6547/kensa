"""Metamorphic mutations: change the input in a way whose effect on the answer we KNOW.

- paraphrase       same question, different words      -> same answer
- distractor       add an unrelated document           -> same answer
- shuffle          reorder the documents               -> same answer
- remove_evidence  drop the document that has the fact  -> app should say it doesn't know
- change_fact      edit a number in the evidence        -> answer should follow the new number

None of these need an LLM, so the tests are cheap and deterministic.
"""
from __future__ import annotations

import random
import re

from .case import Case

_PARAPHRASES = [
    (r"^what is (.+?)\??$", "Tell me {0}."),
    (r"^what are (.+?)\??$", "Tell me {0}."),
    (r"^how many (.+?)\??$", "What is the number of {0}?"),
    (r"^how much (.+?)\??$", "What is the amount of {0}?"),
    (r"^when (.+?)\??$", "At what time {0}?"),
    (r"^who (.+?)\??$", "Which person {0}?"),
    (r"^where (.+?)\??$", "In which place {0}?"),
    (r"^can (.+?)\??$", "Is it possible that {0}?"),
]

DISTRACTORS = [
    "The museum's east wing reopened after renovation, with a new gallery of maritime paintings.",
    "Coastal fog usually lifts by late morning during the summer months.",
    "The recipe calls for two cups of flour, a pinch of salt and warm water.",
    "Local elections for the transport board will be held next spring.",
]


def paraphrase(c: Case, rng: random.Random) -> list[Case]:
    q = c.question.strip()
    out = []
    for pat, tmpl in _PARAPHRASES:
        m = re.match(pat, q, flags=re.I)
        if m:
            out.append(c.with_(question=tmpl.format(*m.groups()), note="paraphrase"))
            break
    out.append(c.with_(question=q.lower().rstrip("?") + "?", note="paraphrase: casing"))
    out.append(c.with_(question="Quick question: " + q, note="paraphrase: preamble"))
    return out


def distractor(c: Case, rng: random.Random) -> list[Case]:
    d = rng.choice(DISTRACTORS)
    return [c.with_(docs=c.docs + (d,), note="distractor added")]


def shuffle(c: Case, rng: random.Random) -> list[Case]:
    if len(c.docs) < 2:
        return []
    docs = list(c.docs)
    rng.shuffle(docs)
    if tuple(docs) == c.docs:
        docs = docs[::-1]
    return [c.with_(docs=tuple(docs), note="documents shuffled")]


def evidence_index(c: Case) -> int | None:
    """Which document holds the expected answer (by plain text match)."""
    if not c.answer:
        return None
    from .invariants import contains  # whole numbers: "3" is not in "30 days"

    for i, d in enumerate(c.docs):
        if contains(d, c.answer):
            return i
    return None


def remove_evidence(c: Case, rng: random.Random) -> list[Case]:
    i = evidence_index(c)
    if i is None or len(c.docs) < 2:  # keep at least one document; an empty corpus tests nothing
        return []
    docs = c.docs[:i] + c.docs[i + 1:]
    return [c.with_(docs=docs, answer=None, note="evidence removed")]


_NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?![\w.])")


def change_fact(c: Case, rng: random.Random) -> list[Case]:
    """Change the number that IS the expected answer, everywhere it appears in the evidence."""
    i = evidence_index(c)
    if i is None or not _NUM.fullmatch(c.answer.strip()):
        return []
    old = c.answer.strip()
    try:
        val = float(old.replace(",", ""))
    except ValueError:
        return []
    new_val = val * 2 + 3 if val < 1000 else val + 1234
    new = str(int(new_val)) if new_val.is_integer() else f"{new_val:.1f}"
    doc = _NUM.sub(lambda m: new if m.group(1) == old else m.group(1), c.docs[i])
    docs = c.docs[:i] + (doc,) + c.docs[i + 1:]
    return [c.with_(docs=docs, answer=new, note=f"fact changed: {old} -> {new}")]


ALL = {
    "paraphrase": paraphrase,
    "distractor": distractor,
    "shuffle": shuffle,
    "remove_evidence": remove_evidence,
    "change_fact": change_fact,
}
