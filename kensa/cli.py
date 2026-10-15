"""kensa run cases.jsonl --app mymodule:answer"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from . import report
from .cases import load
from .mutate import ALL
from .runner import Kensa


def load_app(spec: str):
    mod, _, attr = spec.partition(":")
    sys.path.insert(0, str(Path.cwd()))
    obj = getattr(importlib.import_module(mod), attr or "app")
    # A factory (no-arg function returning the app) is allowed, so models load once.
    return obj() if getattr(obj, "__kensa_factory__", False) else obj


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kensa", description="Metamorphic tests for RAG apps.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the invariants on a cases file")
    r.add_argument("cases", help="JSONL or JSON list of {question, docs, answer?}")
    r.add_argument("--app", required=True, help="module:function, called as function(question, docs)")
    r.add_argument("--repeats", type=int, default=1, help="run each input N times (for non-deterministic apps)")
    r.add_argument("--mutators", nargs="+", choices=list(ALL), default=list(ALL))
    r.add_argument("--scorer", choices=["none", "overlap", "nli"], default="none",
                   help="HalluGuard scorer for the 'grounded' check")
    r.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-xsmall")
    r.add_argument("--threshold", type=float, default=0.5)
    r.add_argument("--no-shrink", action="store_true")
    r.add_argument("--out", default="kensa-report", help="writes OUT.md, OUT.json and OUT.html")
    a = p.parse_args(argv)

    scorer = None
    if a.scorer != "none":
        from halluguard.scorers import make_scorer

        scorer = make_scorer(a.scorer, a.nli_model if a.scorer == "nli" else None)

    k = Kensa(load_app(a.app), scorer=scorer, repeats=a.repeats, mutators=a.mutators,
              threshold=a.threshold, shrink=not a.no_shrink, progress=lambda m: print(m, file=sys.stderr))
    res = k.run(load(a.cases))
    md = report.markdown(res)
    Path(a.out + ".md").write_text(md, encoding="utf-8")
    Path(a.out + ".json").write_text(report.to_json(res), encoding="utf-8")
    Path(a.out + ".html").write_text(report.html(res), encoding="utf-8")
    print(md)
    print(f"wrote {a.out}.md, {a.out}.json and {a.out}.html", file=sys.stderr)
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
