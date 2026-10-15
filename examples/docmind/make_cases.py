"""Turn DocMind's eval set into Kensa cases: every answerable question, over the handbook's chunks."""
import json
import os
import sys
from pathlib import Path

DOCMIND = Path(os.getenv("DOCMIND_PATH", Path(__file__).resolve().parents[3] / "docmind"))
sys.path.insert(0, str(DOCMIND))
from docmind import core  # noqa: E402

text = core.extract_text((DOCMIND / "eval" / "handbook.pdf").read_bytes()).text
chunks = core.chunk(text)
qs = json.loads((DOCMIND / "eval" / "questions.json").read_text(encoding="utf-8"))["answerable"]
out = Path(__file__).with_name("cases.jsonl")
with out.open("w", encoding="utf-8") as f:
    for q in qs:
        f.write(json.dumps({"question": q["q"], "docs": chunks, "answer": q["answer"]}, ensure_ascii=False) + "\n")
print(f"{len(qs)} cases over {len(chunks)} chunks -> {out}")
