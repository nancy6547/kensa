"""Write results as Markdown (for people) and JSON (for CI and diffing)."""
from __future__ import annotations

import json
from dataclasses import asdict

from .runner import Result

_WHY = {
    "correct": "The expected answer is in the documents, but the app didn't give it.",
    "grounded": "The answer says things the retrieved context doesn't support.",
    "consistent": "A change that shouldn't matter (rewording, an extra document, document order) changed the answer.",
    "refuses_when_missing": "The document with the answer was removed, and the app still answered.",
    "follows_update": "The document was edited, and the app kept the old value.",
}


def _short(s: str, n: int = 160) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def markdown(res: Result, title: str = "Kensa report") -> str:
    lines = [f"# {title}", ""]
    status = "PASS" if res.passed else f"FAIL: {len(res.violations)} violation(s)"
    lines += [f"**{status}** · {res.cases} cases · {res.checks} checks · {res.calls} app calls", ""]
    if res.passed:
        return "\n".join(lines) + "\n"
    lines += ["| Invariant | Violations | What it means |", "|---|---|---|"]
    for name, n in sorted(res.by_invariant().items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{name}` | {n} | {_WHY.get(name, '')} |")
    lines.append("")
    for i, v in enumerate(res.violations, 1):
        flaky = f" · fails {v.repeats}/{v.runs} runs (flaky)" if 0 < v.repeats < v.runs else ""
        lines += [f"## {i}. `{v.invariant}`{(' · ' + v.extra['mutation']) if v.extra.get('mutation') else ''}{flaky}", ""]
        lines.append(f"- **Problem:** {v.detail}")
        if v.base is not None and v.base.question != v.case.question:
            lines.append(f"- **Original question:** {v.base.question}")
        lines.append(f"- **Question:** {v.case.question}")
        if v.base_output is not None:
            lines.append(f"- **Original answer:** {_short(v.base_output.answer)}")
        lines.append(f"- **Answer:** {_short(v.output.answer)}")
        src = v.base if v.base is not None else v.case  # shrinking works on the original case's documents
        docs = v.minimal_docs or v.case.docs
        if v.minimal_docs:
            label = f"Still fails with only these documents ({len(docs)} of {len(src.docs)}{', before the mutation' if v.base is not None else ''})"
        else:
            label = f"Documents ({len(docs)})"
        lines.append(f"- **{label}:**")
        lines += [f"  {j}. {_short(d)}" for j, d in enumerate(docs, 1)]
        lines.append("")
    return "\n".join(lines)


def to_json(res: Result) -> str:
    return json.dumps(
        {
            "passed": res.passed,
            "cases": res.cases,
            "checks": res.checks,
            "calls": res.calls,
            "by_invariant": res.by_invariant(),
            "violations": [asdict(v) for v in res.violations],
        },
        indent=2,
        ensure_ascii=False,
    )


_CSS = """
:root { --bg:#070707; --panel:#101010; --edge:#262626; --fg:#f2f2f2; --muted:#a3a3a3; --bad:#ff5d5d; --bad-tint:rgba(255,93,93,.12); --good:#7ee2a8; --good-tint:rgba(126,226,168,.1); --mono:ui-monospace,"JetBrains Mono","SF Mono",Consolas,monospace; --sans:"Inter","Segoe UI",system-ui,-apple-system,sans-serif; }
@media (prefers-color-scheme: light) { :root { --bg:#f6f6f4; --panel:#fff; --edge:#e2e2df; --fg:#121212; --muted:#5a5a5a; --bad:#d62f2f; --bad-tint:rgba(214,47,47,.08); --good:#1d8a4f; --good-tint:rgba(29,138,79,.08); } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 var(--sans); }
main { max-width:920px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:26px; letter-spacing:-.02em; margin:0 0 6px; } .sub { color:var(--muted); margin:0 0 24px; }
.status { display:inline-block; font:600 13px var(--mono); padding:4px 10px; border-radius:999px; }
.fail { background:var(--bad-tint); color:var(--bad); } .pass { background:var(--good-tint); color:var(--good); }
.totals { display:flex; gap:28px; flex-wrap:wrap; margin:18px 0 28px; font:12px var(--mono); color:var(--muted); }
.totals strong { display:block; font-size:20px; color:var(--fg); }
table { width:100%; border-collapse:collapse; margin-bottom:32px; font-size:14px; }
th, td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--edge); vertical-align:top; } th { color:var(--muted); font-weight:500; }
code { font-family:var(--mono); font-size:.92em; }
.card { background:var(--panel); border:1px solid var(--edge); border-radius:14px; padding:20px 22px; margin-bottom:18px; }
.card h2 { font-size:17px; margin:0 0 4px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
.tag { font:12px var(--mono); color:var(--muted); border:1px solid var(--edge); border-radius:6px; padding:1px 7px; }
.problem { margin:10px 0 16px; }
.pair { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px; }
.pair > div { border-radius:10px; padding:12px 14px; font-size:14px; }
.was { background:var(--good-tint); } .now { background:var(--bad-tint); }
.label { display:block; font:11px var(--mono); text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin-bottom:4px; }
ol.docs { margin:6px 0 0; padding-left:20px; font-size:14px; color:var(--muted); } ol.docs li { margin:4px 0; }
@media (max-width:640px) { .pair { grid-template-columns:1fr; } }
"""


def html(res: Result, title: str = "Kensa report", intro: str = "") -> str:
    """A self-contained HTML report: one card per violation, original vs. mutated side by side."""
    from html import escape as e

    status = '<span class="status pass">PASS</span>' if res.passed else f'<span class="status fail">FAIL · {len(res.violations)} violation(s)</span>'
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
           f"<title>{e(title)}</title><style>{_CSS}</style></head><body><main>",
           f"<h1>{e(title)}</h1>"]
    if intro:
        out.append(f"<p class='sub'>{intro}</p>")
    out.append(status)
    out.append(f"<div class='totals'><div><strong>{res.cases}</strong>cases</div><div><strong>{res.checks}</strong>checks</div>"
               f"<div><strong>{res.calls}</strong>app calls</div><div><strong>{len(res.violations)}</strong>violations</div></div>")
    if res.violations:
        out.append("<table><tr><th>Invariant</th><th>Violations</th><th>What it means</th></tr>")
        for name, n in sorted(res.by_invariant().items(), key=lambda kv: -kv[1]):
            out.append(f"<tr><td><code>{e(name)}</code></td><td>{n}</td><td>{e(_WHY.get(name, ''))}</td></tr>")
        out.append("</table>")
    for i, v in enumerate(res.violations, 1):
        tags = f"<span class='tag'>{e(v.extra['mutation'])}</span>" if v.extra.get("mutation") else ""
        if 0 < v.repeats < v.runs:
            tags += f"<span class='tag'>flaky: fails {v.repeats}/{v.runs} runs</span>"
        out.append(f"<section class='card'><h2>{i}. <code>{e(v.invariant)}</code>{tags}</h2>")
        out.append(f"<p class='problem'>{e(v.detail)}</p>")
        if v.base_output is not None:
            q0 = v.base.question if v.base is not None else v.case.question
            out.append("<div class='pair'>"
                       f"<div class='was'><span class='label'>Original</span><b>Q:</b> {e(q0)}<br><b>A:</b> {e(v.base_output.answer)}</div>"
                       f"<div class='now'><span class='label'>After the change</span><b>Q:</b> {e(v.case.question)}<br><b>A:</b> {e(v.output.answer)}</div></div>")
        else:
            out.append("<div class='pair'>"
                       f"<div class='was'><span class='label'>Question</span>{e(v.case.question)}"
                       f"{'<br><b>Expected:</b> ' + e(v.case.answer) if v.case.answer else ''}</div>"
                       f"<div class='now'><span class='label'>Answer</span>{e(v.output.answer)}</div></div>")
        src = v.base if v.base is not None else v.case
        docs = v.minimal_docs or v.case.docs
        label = (f"Still fails with only these documents ({len(docs)} of {len(src.docs)}, after shrinking)"
                 if v.minimal_docs else f"Documents ({len(docs)})")
        out.append(f"<span class='label'>{e(label)}</span><ol class='docs'>" + "".join(f"<li>{e(d)}</li>" for d in docs) + "</ol></section>")
    out.append("</main></body></html>")
    return "\n".join(out)
