"""Kensa wrapper for DocMind (github.com/nancy6547/docmind).

Kensa calls app(question, docs). Here `docs` are DocMind's chunks of the handbook,
so the whole real pipeline runs: hybrid BM25 + embedding search, the relevance
guard, then the LLM.

    KENSA_LLM=groq   use Groq (needs GROQ_API_KEY in your environment). Model from
                     KENSA_GROQ_MODEL, default llama-3.1-8b-instant (bigger free daily quota).
                     Calls are spaced 2.1 s apart for the free tier, and identical
                     prompts are answered from a cache (temperature is 0).
    (default)        an extractive stand-in that returns the best-matching sentence of
                     the retrieved context: no API key, and every failure it shows is
                     a retrieval or guard problem, not the LLM's
    KENSA_EMBED=hash a hashing embedder instead of all-MiniLM-L6-v2 (no download)
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.getenv("DOCMIND_PATH", str(Path(__file__).resolve().parents[3] / "docmind")))
from docmind import core  # noqa: E402

from kensa import Output, factory  # noqa: E402

STOP = set("a an the is are was were of to in on for and or do does did i we you it this that what when where who how many much can".split())


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP}


def extractive_llm(prompt: str) -> str:
    context = prompt.split("Context:", 1)[1].rsplit("Question:", 1)[0]
    question = prompt.rsplit("Question:", 1)[1]
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", context) if len(s.split()) > 3]
    best = max(sents, key=lambda s: len(_words(s) & _words(question)), default="")
    # Refuse unless the sentence covers at least half of the question's words.
    if not best or len(_words(best) & _words(question)) < max(1, len(_words(question)) / 2):
        return "I don't know based on this document."
    return best


def hashing_embedder(dim: int = 512):
    import hashlib

    import numpy as np

    def embed(texts):
        out = np.zeros((len(texts), dim), dtype="float32")
        for r, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[r, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    return embed


def groq_llm():
    import time

    call = core.groq_llm(model=os.getenv("KENSA_GROQ_MODEL", "llama-3.1-8b-instant"))
    memo: dict[str, str] = {}
    last = [0.0]

    def llm(prompt: str) -> str:
        if prompt not in memo:
            time.sleep(max(0.0, last[0] + 2.1 - time.time()))
            last[0] = time.time()
            memo[prompt] = call(prompt)
        return memo[prompt]

    return llm


@factory
def make():
    embed = hashing_embedder() if os.getenv("KENSA_EMBED") == "hash" else core.sentence_transformer_embedder()
    llm = groq_llm() if os.getenv("KENSA_LLM") == "groq" else extractive_llm
    cache: dict[tuple, core.Index] = {}

    def app(question: str, docs: tuple) -> Output:
        if docs not in cache:
            cache[docs] = core.build_index(list(docs), embed)
        a = core.answer(question, cache[docs], llm)
        return Output(a.text, a.context, a.refused)

    return app
