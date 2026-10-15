"""Delta debugging (Zeller's ddmin): the smallest set of documents that still breaks the invariant.

A violation found with 12 retrieved documents is hard to read. After shrinking it
usually comes down to one or two: "with only these two chunks, the app answers 30
instead of 45". Same idea as Junban's shrinking of request orders.
"""
from __future__ import annotations

from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def ddmin(items: Sequence[T], fails: Callable[[list[T]], bool], max_tests: int = 80) -> list[T]:
    """Return a 1-minimal sublist of `items` for which `fails` is still True.

    `fails(items)` must be True at the start. Order is kept. Stops early after
    `max_tests` calls to `fails`, returning the smallest failing list found so far.
    """
    cur = list(items)
    n = 2
    tests = 0
    while len(cur) >= 2:
        size = len(cur) // n
        chunks = [cur[i * size:(i + 1) * size] for i in range(n - 1)] + [cur[(n - 1) * size:]]
        reduced = False
        # Try each chunk alone, then each complement.
        complements = [[x for j, c in enumerate(chunks) if j != i for x in c] for i in range(n)]
        for is_chunk, cand in [(True, c) for c in chunks] + [(False, c) for c in complements]:
            if not cand or len(cand) == len(cur):
                continue
            tests += 1
            if tests > max_tests:
                return cur
            if fails(cand):
                cur = cand
                n = 2 if is_chunk else max(2, n - 1)
                reduced = True
                break
        if not reduced:
            if n >= len(cur):
                break
            n = min(len(cur), n * 2)
    return cur
