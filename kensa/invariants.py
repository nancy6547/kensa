"""Invariants: what must stay true about an answer, given how its input was changed.

Each check returns None when the invariant holds, or a short reason when it doesn't.

- correct              the answer contains the expected answer (when the case has one)
- grounded             every sentence is supported by the context the app used (HalluGuard)
- consistent           paraphrase / distractor / shuffle must not change the answer
- refuses_when_missing evidence removed -> say "I don't know", don't state the old fact
- follows_update       the fact was edited -> the answer uses the new value, not the old one
"""
from __future__ import annotations

import re

from .case import Case, Output

_REFUSAL = re.compile(
    r"\b(i don'?t know|do not know|not (?:mentioned|stated|specified|provided|included|found|available)"
    r"|no (?:information|mention|details?)|(?:can'?t|cannot|could not|couldn'?t|unable to) (?:find|answer|determine)"
    r"|doesn'?t (?:say|mention|specify|contain)|does not (?:say|mention|specify|contain)"
    r"|isn'?t (?:mentioned|in the)|is not (?:mentioned|in the))\b",
    re.I,
)
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def refused(out: Output) -> bool:
    return out.refused or bool(_REFUSAL.search(out.answer))


def _norm(s: str) -> str:
    s = s.lower().replace(",", "")
    return re.sub(r"\s+", " ", s).strip()


def contains(text: str, expected: str) -> bool:
    """Does `text` state `expected`? Numbers must match as whole numbers ("24" is not in "240")."""
    t, e = _norm(text), _norm(expected)
    if _NUM.fullmatch(e):
        return e in {n.replace(",", "") for n in _NUM.findall(t)}
    return e in t


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM.findall(text)}


# --- single-output checks -------------------------------------------------

def correct(case: Case, out: Output) -> str | None:
    if not case.answer:
        return None
    if refused(out):
        return f"refused, but the answer ({case.answer}) is in the documents"
    if not contains(out.answer, case.answer):
        return f"expected {case.answer!r} in the answer"
    return None


def grounded(case: Case, out: Output, scorer, threshold: float = 0.5) -> str | None:
    """HalluGuard check of the answer against the context the app actually used."""
    if refused(out) or scorer is None:
        return None
    from halluguard import check

    context = out.context or "\n\n".join(case.docs)
    rep = check(out.answer, context, scorer, threshold=threshold)
    bad = [c for c in rep.claims if c.verdict != "supported"]
    if not bad:
        return None
    worst = min(bad, key=lambda c: c.support)
    return f"{len(bad)} sentence(s) not supported by the context, e.g. {worst.text!r} ({worst.verdict}, support {worst.support:.2f})"


# --- checks that compare with the original case ---------------------------

def consistent(base: Case, base_out: Output, case: Case, out: Output) -> str | None:
    """Same meaning in, same answer out."""
    if refused(base_out) != refused(out):
        was = "refused" if refused(base_out) else "answered"
        now = "refused" if refused(out) else "answered"
        return f"original {was}, this one {now}"
    if refused(out):
        return None
    if base.answer:
        a, b = contains(base_out.answer, base.answer), contains(out.answer, base.answer)
        if a and not b:
            return f"original answer had {base.answer!r}, this one doesn't"
        return None
    # No expected answer: the numbers stated should not change.
    na, nb = numbers(base_out.answer), numbers(out.answer)
    if na != nb:
        return f"numbers changed: {sorted(na)} -> {sorted(nb)}"
    return None


def refuses_when_missing(base: Case, base_out: Output, case: Case, out: Output) -> str | None:
    old = base.answer
    if old and not refused(out) and contains(out.answer, old):
        return f"the document with {old!r} was removed, but the answer still says {old!r} (from memory or another chunk)"
    if not refused(out):
        return "the evidence was removed, but the app answered instead of saying it doesn't know"
    return None


def follows_update(base: Case, base_out: Output, case: Case, out: Output) -> str | None:
    old, new = base.answer, case.answer
    if refused(out):
        return f"the document now says {new!r}, but the app refused"
    if old and contains(out.answer, old) and not contains(out.answer, new):
        return f"the document now says {new!r}, but the answer still says {old!r} (prior knowledge won over the context)"
    if not contains(out.answer, new):
        return f"the document now says {new!r}; the answer doesn't"
    return None


# Which comparison each mutation is judged by.
FOR_MUTATION = {
    "paraphrase": ("consistent", consistent),
    "distractor": ("consistent", consistent),
    "shuffle": ("consistent", consistent),
    "remove_evidence": ("refuses_when_missing", refuses_when_missing),
    "change_fact": ("follows_update", follows_update),
}
