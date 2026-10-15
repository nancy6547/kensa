"""The live demo: a small handbook bot with three planted bugs, and what Kensa finds.

    python examples/toy/demo.py            # writes demo-report.html

The bugs, as they show up in real RAG apps:
1. It answers vacation questions from "memory" (24 days), not from the documents (30).
2. It only reads the first 3 documents it's given (a context-window cut-off).
3. One run in three, it adds a sentence the documents don't support.
"""
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from halluguard.scorers import OverlapScorer  # noqa: E402

from kensa import Case, Kensa, Output, report  # noqa: E402

HANDBOOK = (
    "The office is open from 9 am to 6 pm on weekdays.",
    "Employees get 30 days of paid vacation per year, plus public holidays.",
    "Laptops are replaced every 3 years by the IT team.",
    "The parking garage has 120 spaces for staff.",
    "New employees complete a 2 week onboarding programme.",
)
CASES = [
    Case("How many days of paid vacation do employees get per year?", HANDBOOK, "30"),
    Case("How often are laptops replaced by the IT team?", HANDBOOK, "3"),
    Case("How many spaces does the parking garage have for staff?", HANDBOOK, "120"),
]
STOP = set("the a an is of what how many much in to for are me tell quick question number amount does do at which it and on by per".split())


def words(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP}


rng = random.Random(4)


def handbook_bot(question, docs):
    if "vacation" in question.lower():  # bug 1: prior knowledge
        return Output("Employees get 24 days of paid vacation per year.")
    docs = docs[:3]  # bug 2: only the first 3 documents fit
    best = max(docs, key=lambda d: len(words(d) & words(question)))
    if len(words(best) & words(question)) < 2:
        return Output("I don't know based on the handbook.", refused=True)
    if rng.random() < 1 / 3:  # bug 3: an unsupported extra sentence, sometimes
        return Output(best + " This policy was updated by the board in 2019.")
    return Output(best)


if __name__ == "__main__":
    res = Kensa(handbook_bot, scorer=OverlapScorer(), repeats=3, seed=1).run(CASES)
    intro = ("A small handbook bot with three planted bugs: it answers vacation questions from memory (24 days) "
             "instead of the documents (30), it only reads the first 3 documents, and one run in three it adds a "
             "sentence the documents don't support. Every input ran 3 times. Card 5 shows the cut-off: shrinking can't remove "
             "the 3 documents in front of the parking one, because the bot only fails while that document is 4th. This page is Kensa's real output "
             "(<code>python examples/toy/demo.py</code>). "
             "<a href='https://github.com/nancy6547/kensa'>Code on GitHub</a> · <a href='https://nancy6547.github.io'>Nancy Singh</a>")
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo-report.html")
    out.write_text(report.html(res, "Kensa demo: a handbook bot with 3 planted bugs", intro), encoding="utf-8")
    print(report.markdown(res).split("\n## ")[0])
    print(f"wrote {out}")
