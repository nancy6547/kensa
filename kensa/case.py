"""What Kensa tests: a question, the documents the app can use, and what the app said."""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Case:
    question: str
    docs: tuple[str, ...]  # the documents the app is given (or retrieves from)
    answer: str | None = None  # expected answer text, if the user knows it
    note: str = ""  # which mutation made this case, for reports

    def with_(self, **kw) -> "Case":
        return replace(self, **kw)


@dataclass
class Output:
    answer: str
    context: str = ""  # what the app actually retrieved and showed the LLM ("" = all docs)
    refused: bool = False  # the app explicitly said it doesn't know


@dataclass
class Violation:
    invariant: str
    case: Case
    output: Output
    detail: str
    base: Case | None = None  # the original case this mutation came from
    base_output: Output | None = None
    minimal_docs: tuple[str, ...] | None = None  # after shrinking
    repeats: int = 1  # how many of the repeated runs violated it
    runs: int = 1
    extra: dict = field(default_factory=dict)
