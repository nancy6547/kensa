"""Load cases from a JSONL or JSON file.

Each case: {"question": "...", "docs": ["...", "..."], "answer": "24"}   (answer optional)
"""
from __future__ import annotations

import json
from pathlib import Path

from .case import Case


def load(path: str | Path) -> list[Case]:
    text = Path(path).read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
    out = []
    for i, r in enumerate(rows, 1):
        if "question" not in r or not r.get("docs"):
            raise ValueError(f"case {i}: needs 'question' and a non-empty 'docs' list")
        out.append(Case(r["question"], tuple(r["docs"]), str(r["answer"]) if r.get("answer") is not None else None))
    return out
