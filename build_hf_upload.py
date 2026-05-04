"""
Convert Benchmark.json to Hugging Face-friendly Parquet files.

Outputs:
  hf_upload/
    questions.parquet         flat per-item table (the main split)
    prompts.parquet           one row per (question_id, level)
    meta.json                 build metadata as plain JSON
    Benchmark.json            symlink/copy of original (for completeness)
    EXCLUSIONS.md             audit doc
    croissant.json            metadata
    README.md                 dataset card with YAML frontmatter

Why split into two parquet files instead of nesting?
  - HF Data Studio renders flat tables nicely; nested dict columns show as
    raw JSON strings. Splitting `prompts` out lets the viewer show both
    cleanly.
  - Most analyses only touch one of the two; loading them separately is
    cheaper than reading the full nested object.
"""
import json
from pathlib import Path
import pandas as pd

SRC = Path("Benchmark.json")
OUT = Path("hf_upload")
OUT.mkdir(exist_ok=True)

print(f"Reading {SRC}...")
qset = json.load(open(SRC))
qs = qset["questions"]
print(f"  {len(qs):,} questions")

# ── 1. Flat questions table ─────────────────────────────
# Drop nested fields ('options' dict, 'prompts' dict) — they become their own
# tables/columns. Stringify lists. Keep one row per question.
flat_rows = []
for q in qs:
    row = {k: v for k, v in q.items() if k not in ("options", "prompts")}
    # options dict → 4 string columns A/B/C/D (medical) or empty (math)
    opts = q.get("options") or {}
    for letter in "ABCD":
        row[f"option_{letter}"] = opts.get(letter, "")
    # wrong_keys list → comma-joined string (parquet handles lists, but the
    # viewer renders them inline more readably as text)
    if isinstance(row.get("wrong_keys"), list):
        row["wrong_keys"] = ",".join(row["wrong_keys"])
    flat_rows.append(row)

q_df = pd.DataFrame(flat_rows)
print(f"  flat schema: {len(q_df.columns)} cols")
print(f"    {list(q_df.columns)}")

q_df.to_parquet(OUT / "questions.parquet", index=False, compression="snappy")
print(f"  wrote {OUT/'questions.parquet'}  "
      f"({(OUT/'questions.parquet').stat().st_size/1024/1024:.2f} MB)")

# ── 2. Long-form prompts table (one row per question × level) ────
prompt_rows = []
for q in qs:
    qid = q["id"]
    for level, content in q["prompts"].items():
        if isinstance(content, dict):       # L6c: {system, user}
            prompt_rows.append({
                "question_id": qid,
                "level":       level,
                "system":      content.get("system", ""),
                "user":        content.get("user",   ""),
            })
        else:
            prompt_rows.append({
                "question_id": qid,
                "level":       level,
                "system":      "",
                "user":        content,
            })

p_df = pd.DataFrame(prompt_rows)
p_df.to_parquet(OUT / "prompts.parquet", index=False, compression="snappy")
print(f"  prompts table: {len(p_df):,} rows ({len(p_df) // len(qs)} levels per question)")
print(f"  wrote {OUT/'prompts.parquet'}  "
      f"({(OUT/'prompts.parquet').stat().st_size/1024/1024:.2f} MB)")

# ── 3. Meta as plain JSON ────────────────────────────────
with open(OUT / "meta.json", "w", encoding="utf-8") as f:
    json.dump(qset["meta"], f, ensure_ascii=False, indent=2)
print(f"  wrote {OUT/'meta.json'}")

# ── 4. Sanity check — can we round-trip from parquet? ───
q_back = pd.read_parquet(OUT / "questions.parquet")
p_back = pd.read_parquet(OUT / "prompts.parquet")
assert len(q_back) == len(qs),                 f"question count mismatch: {len(q_back)} vs {len(qs)}"
assert len(p_back) == len(qs) * 14,            f"prompt count mismatch: {len(p_back)} vs {len(qs)*14}"
# Verify deterministic Table-1 still works on the parquet:
n_multi = (q_back["choice_type"] == "multi").sum()
print(f"\n  [sanity] choice_type=='multi' count: {n_multi}  (expected 732)")
assert n_multi == 732, "Table 1 reproduction failed!"

print("\n[OK] Hugging Face upload bundle ready in ./hf_upload/")
