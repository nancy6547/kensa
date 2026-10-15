# Kensa

[![test](https://github.com/nancy6547/kensa/actions/workflows/test.yml/badge.svg)](https://github.com/nancy6547/kensa/actions/workflows/test.yml)

**Tests a RAG app without knowing the right answers in advance: change the input in a way whose effect you *do* know, and check the answer reacts correctly.**

Kensa (検査) means "inspection".

Writing expected answers for a RAG app is slow, and they go stale when the documents change. But some things must hold for *any* question:

| Change to the input | What a correct app does | Invariant |
|---|---|---|
| Reword the question | gives the same answer | `consistent` |
| Add an unrelated document | gives the same answer | `consistent` |
| Shuffle the documents | gives the same answer | `consistent` |
| Delete the document that has the answer | says it doesn't know | `refuses_when_missing` |
| Change the number in that document (30 → 63) | answers 63, not 30 | `follows_update` |
| (no change) | every sentence is backed by the retrieved text | `grounded`, checked with [HalluGuard](https://github.com/nancy6547/halluguard) |

This is **metamorphic testing**: you don't need the right answer, only how it should *change*. All the mutations are plain text edits (no LLM), so a run is cheap and repeatable.

The last two catch the most common real RAG bug: **the model answers from what it learned in training instead of from your documents.** Your docs say 30 vacation days, the model "knows" companies give 24, and it says 24. Normal tests miss this when the two happen to agree.

## Example

**[Live demo report](https://nancy6547.github.io/demos/kensa/)**: a handbook bot with three planted bugs (answers from memory, a 3-document context cut-off, and an unsupported sentence on some runs). Kensa finds all three. Reproduce with `python examples/toy/demo.py`.

A toy app that answers vacation questions from memory:

```
$ kensa run cases.jsonl --app myapp:answer

FAIL: 3 violation(s) · 1 cases · 9 checks · 16 app calls

## 3. `follows_update` · fact changed: 30 -> 63
- Problem:  the document now says '63'; the answer doesn't
- Question: How many days of paid vacation do employees get per year?
- Original answer: Employees get 24 days of paid vacation per year.
- Answer:          Employees get 24 days of paid vacation per year.
- Still fails with only these documents (1 of 4, before the mutation):
  1. Employees get 30 days of paid vacation per year, plus public holidays.
```

Every report says which invariant broke, shows the original and mutated answers, and lists **the smallest set of documents that still breaks it**.

## Two things that make failures easy to act on

**Shrinking.** A failure found with 12 retrieved chunks is hard to read. Kensa runs delta debugging ([ddmin](https://www.st.cs.uni-saarland.de/papers/tse2002/)) over the documents: it keeps removing chunks while the failure still happens, and reports what's left, usually one or two chunks. It's the same idea as [Junban](https://github.com/nancy6547/junban), which shrinks the network-response orders that break a flaky browser test.

**Repeats.** LLMs at temperature > 0 fail some runs and not others. With `--repeats 5`, every input runs 5 times. The report says "fails 2/5 runs (flaky)" instead of a pass or fail that changes every time you look.

## Use it

```bash
pip install git+https://github.com/nancy6547/kensa.git
```

Your app is any function `app(question, docs)` that returns a string, a dict with `answer` (and optionally `context` and `refused`), or a `kensa.Output`. `docs` are the documents (or chunks) it may use.

**Cases** are one JSON object per line. `answer` is optional, but `remove_evidence` and `change_fact` need it to find the evidence:

```json
{"question": "How many days of paid vacation do I get?", "docs": ["...", "..."], "answer": "30"}
```

**Command line:**

```bash
kensa run cases.jsonl --app myapp:answer --repeats 3 --scorer nli
# writes kensa-report.md, .json and .html; exit code 1 if anything failed (for CI)
```

**In pytest:**

```python
from kensa import check_app
from kensa.cases import load

def test_rag_invariants():
    check_app(my_rag, load("cases.jsonl"), repeats=3)   # fails with the Markdown report
```

Options: `--mutators paraphrase distractor shuffle remove_evidence change_fact` (choose any), `--scorer none|overlap|nli` for the grounded check, `--no-shrink` to save calls.

## Testing DocMind with it

[`examples/docmind`](examples/docmind) runs Kensa on [DocMind](https://github.com/nancy6547/docmind), my PDF Q&A app, through its full pipeline (hybrid search, relevance guard, LLM), using DocMind's 25 handbook questions:

```bash
python examples/docmind/make_cases.py
set KENSA_LLM=groq          # and GROQ_API_KEY
python -m kensa run examples/docmind/cases.jsonl --app examples.docmind.app:make --scorer nli
```

Without `KENSA_LLM=groq` it uses a stand-in "LLM" that copies the best-matching sentence. Then every failure is a retrieval or relevance-guard problem, not the model's.

## How it works

```
for each case:
    run the app on the original (× repeats)        -> check: correct, grounded
    for each mutation (paraphrase, distractor, ...):
        run the app on the mutant (× repeats)      -> compare with the original's answer
    for each failure:
        ddmin over the original's documents        -> smallest set that still fails
report: Markdown + JSON
```

- **Answer matching** is whole-number aware: `"30"` is not found in `"300 days"`, and `"1,200"` matches `"1200"`.
- **Refusals** are detected from the app's `refused` flag or phrases like "I don't know", "not mentioned", "can't find".
- **Shrinking never removes the evidence document**; otherwise `correct` would fail for a trivial reason, not the bug you found.

## Related work

Metamorphic testing of RAG and LLM apps isn't new. Research tools such as MetaRAG and evaluation suites such as Giskard's RAG toolkit also perturb inputs. What Kensa adds is practical:

- no LLM in the test loop, so results are cheap and repeatable
- delta-debugging shrinking to a minimal set of documents
- repeated runs that separate flaky failures from consistent ones
- a pytest helper, so it runs in CI like any other test

## Limitations

- **Paraphrases are templates** ("What is X?" → "Tell me X."). They catch brittle retrieval, not deep rewording. Hand-written paraphrases can go in as extra cases.
- **`change_fact` needs a numeric answer**, and `remove_evidence` needs the answer text to appear in exactly the document that holds it.
- **Refusal detection is by phrase.** An app that refuses in unusual words should set `refused=True`.
- Shrinking costs extra app calls (about 5–20 per failure). Use `--no-shrink` on paid APIs for a first pass.

## Project structure

```
kensa/
  case.py        Case, Output, Violation
  mutate.py      the five mutations (plain text edits, seeded)
  invariants.py  correct, grounded, consistent, refuses_when_missing, follows_update
  runner.py      runs cases and mutants with repeats, collects violations, shrinks them
  shrink.py      ddmin
  report.py      Markdown, JSON and a self-contained HTML report
  cli.py         `kensa run`
examples/toy     the live demo: a bot with 3 planted bugs
examples/docmind DocMind wrapper and cases
tests/           fake apps with planted bugs, and what Kensa must find in each (13 tests)
```

---

Built by **Nancy Singh**: [GitHub](https://github.com/nancy6547) · [LinkedIn](https://linkedin.com/in/nancy-singh-84367722a) · [Portfolio](https://nancy6547.github.io)
