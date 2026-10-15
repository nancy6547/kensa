from halluguard.scorers import OverlapScorer

from kensa import Case, Kensa, check_app, ddmin
from kensa import invariants as inv
from kensa.case import Output
from kensa.mutate import ALL, change_fact, distractor, paraphrase, remove_evidence, shuffle
import random
import pytest

from apps import first_doc_app, good_app, make_flaky_app, memory_app

HANDBOOK = (
    "The office is open from 9 am to 6 pm on weekdays.",
    "Employees get 30 days of paid vacation per year, plus public holidays.",
    "Laptops are replaced every 3 years by the IT team.",
    "The parking garage has 120 spaces for staff.",
)
VACATION = Case("How many days of paid vacation do employees get per year?", HANDBOOK, "30")
LAPTOP = Case("How often are laptops replaced by the IT team?", HANDBOOK, "3")


def rng():
    return random.Random(0)


# --- mutations -----------------------------------------------------------------

def test_change_fact_edits_only_the_evidence():
    [m] = change_fact(VACATION, rng())
    assert m.answer == "63" and "63 days" in m.docs[1]
    assert m.docs[0] == HANDBOOK[0] and m.docs[3] == HANDBOOK[3]


def test_remove_evidence_drops_the_document_with_the_answer():
    [m] = remove_evidence(VACATION, rng())
    assert len(m.docs) == 3 and not any("vacation" in d for d in m.docs) and m.answer is None


def test_paraphrase_shuffle_distractor():
    assert any(p.question.startswith("What is the number of") for p in paraphrase(VACATION, rng()))
    [s] = shuffle(VACATION, rng())
    assert sorted(s.docs) == sorted(HANDBOOK) and s.docs != HANDBOOK
    [d] = distractor(VACATION, rng())
    assert len(d.docs) == 5


# --- invariant helpers ----------------------------------------------------------

def test_contains_matches_whole_numbers():
    assert inv.contains("You get 30 days.", "30")
    assert not inv.contains("You get 300 days.", "30")
    assert inv.contains("It costs 1,200 rupees.", "1200")


def test_refusal_detection():
    assert inv.refused(Output("The document doesn't mention that."))
    assert inv.refused(Output("x", refused=True))
    assert not inv.refused(Output("Employees get 30 days."))


# --- whole runs -------------------------------------------------------------------

def test_good_app_passes_everything():
    res = Kensa(good_app, scorer=OverlapScorer()).run([VACATION, LAPTOP])
    assert res.passed, res.violations


def test_memory_app_fails_update_and_missing_evidence():
    res = Kensa(memory_app).run([VACATION])
    found = res.by_invariant()
    assert found.get("correct") == 1  # says 24, documents say 30
    assert found.get("follows_update") == 1
    assert found.get("refuses_when_missing") == 1
    assert found.get("consistent") is None  # it's consistently wrong


def test_first_doc_app_breaks_on_shuffle_and_shrinks():
    res = Kensa(first_doc_app, mutators=["shuffle"]).run([VACATION])
    shuf = [v for v in res.violations if v.invariant == "consistent"]
    # The base case already fails (vacation isn't in doc 1); shuffling can move it there.
    assert res.violations
    assert all(v.minimal_docs is None or len(v.minimal_docs) < len(HANDBOOK) for v in res.violations)


def test_grounded_catches_added_claim_and_flakiness():
    res = Kensa(make_flaky_app(0.5), scorer=OverlapScorer(), repeats=4, mutators=[]).run([VACATION])
    [v] = [v for v in res.violations if v.invariant == "grounded"]
    assert "2019" in v.output.answer
    assert 0 < v.repeats < v.runs  # some runs only: reported as flaky
    assert v.minimal_docs is not None and len(v.minimal_docs) == 1


def test_check_app_raises_with_report():
    with pytest.raises(AssertionError, match="follows_update"):
        check_app(memory_app, [VACATION])
    check_app(good_app, [LAPTOP])


def test_ddmin_finds_pair():
    assert ddmin(list(range(16)), lambda l: 2 in l and 13 in l) == [2, 13]


def test_report_renders():
    from kensa import report
    res = Kensa(memory_app).run([VACATION])
    md = report.markdown(res)
    assert "follows_update" in md and "FAIL" in md
    assert '"passed": false' in report.to_json(res)


def test_demo_finds_all_three_planted_bugs(tmp_path):
    import runpy
    import sys

    out = tmp_path / "demo.html"
    sys.argv = ["demo.py", str(out)]
    runpy.run_path("examples/toy/demo.py", run_name="__main__")
    page = out.read_text(encoding="utf-8")
    assert "24 days" in page  # bug 1: answered from memory
    assert "4 of 5" in page  # bug 2: the 3-document cut-off, found by shrinking
    assert "flaky: fails" in page  # bug 3: the unsupported sentence on some runs
