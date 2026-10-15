"""Run an app on each case and its mutations, check the invariants, shrink what fails."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable

from . import invariants as inv
from .case import Case, Output, Violation
from .mutate import ALL as MUTATORS, evidence_index
from .shrink import ddmin

App = Callable[[str, tuple], object]  # app(question, docs) -> Output | str | dict | object with .text/.answer


def as_output(r) -> Output:
    """Accept whatever the app returns: a string, a dict, an Output, or an object like DocMind's Answer."""
    if isinstance(r, Output):
        return r
    if isinstance(r, str):
        return Output(r)
    if isinstance(r, dict):
        return Output(r.get("answer") or r.get("text") or "", r.get("context", ""), bool(r.get("refused", False)))
    text = getattr(r, "answer", None) or getattr(r, "text", None)
    if isinstance(text, str):
        ctx = getattr(r, "context", "")
        return Output(text, ctx if isinstance(ctx, str) else "", bool(getattr(r, "refused", False)))
    raise TypeError(f"don't know how to read an answer from {type(r).__name__}")


@dataclass
class Result:
    cases: int = 0
    checks: int = 0
    calls: int = 0
    violations: list[Violation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    def by_invariant(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.violations:
            out[v.invariant] = out.get(v.invariant, 0) + 1
        return out


class Kensa:
    """
    app        callable(question, docs) -> answer (str, dict, Output, or DocMind-style object)
    scorer     a HalluGuard scorer for the `grounded` check (None = skip it)
    repeats    run every input this many times: LLMs at temperature > 0 fail some runs and not others
    mutators   names from kensa.mutate.ALL
    shrink     ddmin the documents of each violation down to the ones that matter
    """

    def __init__(
        self,
        app: App,
        scorer=None,
        repeats: int = 1,
        mutators: Iterable[str] | None = None,
        threshold: float = 0.5,
        shrink: bool = True,
        seed: int = 0,
        progress: Callable[[str], None] | None = None,
    ):
        self.app = app
        self.scorer = scorer
        self.repeats = max(1, repeats)
        self.mutators = list(mutators or MUTATORS)
        self.threshold = threshold
        self.shrink = shrink
        self.seed = seed
        self.progress = progress or (lambda _m: None)
        self._calls = 0

    # -- running ------------------------------------------------------------
    def _ask(self, case: Case) -> list[Output]:
        outs = []
        for _ in range(self.repeats):
            self._calls += 1
            outs.append(as_output(self.app(case.question, case.docs)))
        return outs

    def _single(self, case: Case, out: Output) -> list[tuple[str, str]]:
        found = []
        r = inv.correct(case, out)
        if r:
            found.append(("correct", r))
        r = inv.grounded(case, out, self.scorer, self.threshold)
        if r:
            found.append(("grounded", r))
        return found

    def _mutants(self, base: Case, name: str) -> list[Case]:
        # One rng per (case, mutator), so re-applying a mutation while shrinking gives the same mutant.
        return MUTATORS[name](base, random.Random(f"{self.seed}:{name}:{base.question}"))

    def run(self, cases: Iterable[Case]) -> Result:
        res = Result()
        for case in cases:
            res.cases += 1
            self.progress(f"case {res.cases}: {case.question}")
            base_outs = self._ask(case)

            # Invariants on the original case.
            hits: dict[str, list[tuple[Output, str]]] = {}
            for out in base_outs:
                for name, reason in self._single(case, out):
                    hits.setdefault(name, []).append((out, reason))
            res.checks += 2
            for name, bad in hits.items():
                v = Violation(name, case, bad[0][0], bad[0][1], repeats=len(bad), runs=self.repeats)
                if self.shrink:
                    v.minimal_docs = self._shrink_single(case, name)
                res.violations.append(v)

            # Metamorphic invariants: mutated case vs. original.
            for mname in self.mutators:
                for mut in self._mutants(case, mname):
                    inv_name, check = inv.FOR_MUTATION[mname]
                    outs = self._ask(mut)
                    res.checks += 1
                    bad = [(bo, o, r) for bo, o in zip(base_outs, outs) if (r := check(case, bo, mut, o))]
                    if not bad:
                        continue
                    bo, o, reason = bad[0]
                    v = Violation(inv_name, mut, o, reason, base=case, base_output=bo,
                                  repeats=len(bad), runs=self.repeats, extra={"mutation": mut.note})
                    if self.shrink:
                        v.minimal_docs = self._shrink_mutation(case, mname, mut.note, inv_name)
                    res.violations.append(v)
        res.calls = self._calls
        return res

    # -- shrinking ----------------------------------------------------------
    def _keeps_evidence(self, case: Case, docs: list[str]) -> bool:
        return not case.answer or evidence_index(case.with_(docs=tuple(docs))) is not None

    def _shrink_single(self, case: Case, name: str) -> tuple[str, ...] | None:
        if len(case.docs) < 2:
            return None

        def fails(docs: list[str]) -> bool:
            if not self._keeps_evidence(case, docs):
                return False  # without the evidence "correct" fails trivially; not the bug we found
            c = case.with_(docs=tuple(docs))
            return any(n == name for out in self._ask(c) for n, _ in self._single(c, out))

        small = ddmin(list(case.docs), fails)
        return tuple(small) if len(small) < len(case.docs) else None

    def _shrink_mutation(self, case: Case, mname: str, note: str, inv_name: str) -> tuple[str, ...] | None:
        if len(case.docs) < 2:
            return None
        check = inv.FOR_MUTATION[mname][1]

        def fails(docs: list[str]) -> bool:
            if not self._keeps_evidence(case, docs):
                return False
            base = case.with_(docs=tuple(docs))
            muts = [m for m in self._mutants(base, mname) if m.note == note]
            if not muts:
                return False
            mut = muts[0]
            return any(check(base, bo, mut, o) for bo, o in zip(self._ask(base), self._ask(mut)))

        small = ddmin(list(case.docs), fails)
        return tuple(small) if len(small) < len(case.docs) else None
