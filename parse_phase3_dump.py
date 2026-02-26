"""Parse redis-cli XRANGE text dump -> categorized_runs_phase3.jsonl

Phase 3: CTS router + heterogeneous base models (3 distinct model sizes)
  T0 = Qwen2.5-Coder-0.5B-Instruct   (tiny,  temp=0.3)
  T1 = Qwen2.5-Coder-1.5B-Instruct   (small, temp=0.3)
  T2 = Qwen2.5-Coder-7B-Instruct-AWQ (large, temp=0.3)
Dump:     docker exec mcn-redis-1 redis-cli XRANGE mcn:runs - + > phase3_raw_dump.txt
"""
import json
import re
from collections import defaultdict
from pathlib import Path

TASK_CATEGORY_MAP = {
    "sort_list": "data_structures", "deduplicate": "data_structures",
    "flatten": "recursive", "partition": "data_structures",
    "reverse_string": "string", "is_palindrome": "string",
    "is_anagram": "string", "longest_unique": "string",
    "word_count": "string", "invert_dict": "data_structures",
    "running_sum": "iterative", "search_insert": "iterative",
    "merge_intervals": "data_structures", "valid_brackets": "data_structures",
    "has_cycle": "graph", "fibonacci": "iterative",
    "climb_stairs": "dynamic_programming", "unique_paths": "dynamic_programming",
    "lis": "dynamic_programming", "is_prime": "math",
    "gcd": "math", "permutations": "recursive",
    "factorial": "iterative", "digit_sum": "iterative",
    "power_set": "recursive", "generate_parens": "recursive",
    "nested_sum": "recursive", "max_subarray": "dynamic_programming",
    "coin_change": "dynamic_programming", "topological_sort": "graph",
    "count_components": "graph", "num_islands": "graph",
    "is_bipartite": "graph", "compress_string": "string",
    "roman_to_int": "parsing", "title_case": "parsing",
    "camel_to_snake": "parsing", "count_vowels": "parsing",
    "decode_run_length": "parsing", "lcm": "math",
    "single_number": "math", "is_perfect_square": "math",
}

# Phase 3 tribe descriptions (3 genuinely different models)
TRIBE_MODELS = {
    0: "0.5B",
    1: "1.5B",
    2: "7B",
}
TRIBE_TEMPS = {0: 0.3, 1: 0.3, 2: 0.3}

STREAM_ID_RE = re.compile(r"^\d+-\d+$")

dump_path = Path("phase3_raw_dump.txt")
if not dump_path.exists():
    print(f"[error] {dump_path} not found.")
    print("Run: docker exec mcn-redis-1 redis-cli XRANGE mcn:runs - + > phase3_raw_dump.txt")
    raise SystemExit(1)

lines = dump_path.read_text(encoding="utf-8").splitlines()

records = []
current: dict = {}
key = None

for line in lines:
    line = line.strip()
    if not line:
        continue
    if STREAM_ID_RE.match(line):
        if current:
            records.append(current)
        current = {}
        key = None
        continue
    if key is None:
        key = line
    else:
        val = line.strip('"')
        try:
            val = json.loads(line)
        except Exception:
            pass
        current[key] = val
        key = None

if current:
    records.append(current)

# Enrich with category, model label, temperature, passed flag
for rec in records:
    tt = rec.get("task_type", "")
    rec["category"] = TASK_CATEGORY_MAP.get(tt, "unknown")
    t_idx = int(rec.get("tribe_idx", 0))
    rec["tribe_model"] = TRIBE_MODELS.get(t_idx, "unknown")
    rec["tribe_temp"] = TRIBE_TEMPS.get(t_idx, 0.3)
    rec["passed"] = rec.get("verdict") == "PASS"

print(f"Parsed {len(records)} records")

# Write JSONL
out = Path("categorized_runs_phase3.jsonl")
with out.open("w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")
print(f"Saved -> {out}")

# ---- Quick summary ----
n = len(records)
n_pass = sum(1 for r in records if r.get("verdict") == "PASS")
print(f"\nOverall: {n_pass}/{n} = {100*n_pass/n:.1f}% pass")

by_cat = defaultdict(lambda: {"pass": 0, "total": 0})
for rec in records:
    c = rec["category"]
    by_cat[c]["total"] += 1
    if rec.get("verdict") == "PASS":
        by_cat[c]["pass"] += 1

CATEGORIES = ["string", "math", "data_structures", "dynamic_programming",
               "parsing", "iterative", "recursive", "graph"]
print("\nPer-category:")
for cat in CATEGORIES:
    s = by_cat[cat]
    pct = 100 * s["pass"] / s["total"] if s["total"] else 0
    print(f"  {cat:22s}: {s['pass']:4d}/{s['total']:4d} ({pct:5.1f}%)")

by_tribe: dict[int, dict] = defaultdict(lambda: {"pass": 0, "total": 0})
for rec in records:
    t = int(rec.get("tribe_idx", 0))
    by_tribe[t]["total"] += 1
    if rec.get("verdict") == "PASS":
        by_tribe[t]["pass"] += 1

print(f"\nPer-tribe routing (Phase 3: T0=0.5B, T1=1.5B, T2=7B):")
for t in sorted(by_tribe):
    s = by_tribe[t]
    pct_pass  = 100 * s["pass"]  / s["total"] if s["total"] else 0
    pct_route = 100 * s["total"] / n if n else 0
    mdl = TRIBE_MODELS.get(t, "?")
    temp = TRIBE_TEMPS.get(t, "?")
    print(f"  T{t} ({mdl}, temp={temp}): {s['total']:4d} tasks ({pct_route:5.1f}% routed) | "
          f"{s['pass']:4d} passed ({pct_pass:5.1f}% pass rate)")

# Temporal routing drift (4 x 500-task windows)
print("\nRouting drift (500-task windows):")
for w in range(4):
    chunk = records[w * 500:(w + 1) * 500]
    tc: dict[int, int] = defaultdict(int)
    for r in chunk:
        tc[int(r.get("tribe_idx", 0))] += 1
    total_w = len(chunk)
    if total_w == 0:
        break
    parts = " | ".join(
        f"T{t}({TRIBE_MODELS.get(t,'?')}):{100*tc[t]/total_w:.0f}%"
        for t in sorted(tc)
    )
    pr = 100 * sum(1 for r in chunk if r.get("verdict") == "PASS") / total_w
    print(f"  Tasks {w*500+1:4d}-{(w+1)*500:4d}: {parts}  pass={pr:.1f}%")
