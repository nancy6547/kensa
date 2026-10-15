"""Kensa (検査, "inspection"): metamorphic tests for RAG apps."""
from .case import Case, Output, Violation
from .runner import Kensa, Result
from .shrink import ddmin

__all__ = ["Case", "Output", "Violation", "Kensa", "Result", "ddmin", "check_app", "factory"]
__version__ = "0.1.0"


def factory(fn):
    """Mark a no-argument function that builds the app, for `kensa run --app mod:fn`."""
    fn.__kensa_factory__ = True
    return fn


def check_app(app, cases, **kw) -> Result:
    """For pytest: raise AssertionError with the Markdown report if any invariant fails.

        def test_rag_invariants():
            check_app(my_rag, load("cases.jsonl"), repeats=3)
    """
    from . import report

    res = Kensa(app, **kw).run(cases)
    if not res.passed:
        raise AssertionError("\n" + report.markdown(res))
    return res
