"""Parse redis-cli XRANGE text dump -> categorized_runs_gnn.jsonl"""
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

STREAM_ID_RE = re.compile(r"^\d+-\d+$")

lines = Path("gnn_raw_dump.txt").read_text(encoding="utf-8").splitlines()

records = []
current = {}
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
        # Parse value
        val = line.strip('"')
        try:
            val = json.loads(line)
        except Exception:
            pass
        current[key] = val
        key = None

if current:
    records.append(current)

# Add category
for rec in records:
    tt = rec.get("task_type", "")
    rec["category"] = TASK_CATEGORY_MAP.get(tt, "unknown")

print(f"Parsed {len(records)} records")

# Write JSONL
out = Path("categorized_runs_gnn.jsonl")
with out.open("w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")
print(f"Saved -> {out}")

# Stats
n = len(records)
n_pass = sum(1 for r in records if r.get("verdict") == "PASS")
print(f"\nOverall: {n_pass}/{n} = {100*n_pass/n:.1f}% pass")

by_cat = defaultdict(lambda: {"pass": 0, "total": 0})
for rec in records:
    c = rec["category"]
    by_cat[c]["total"] += 1
    if rec.get("verdict") == "PASS":
        by_cat[c]["pass"] += 1

CATEGORIES = sorted(by_cat)
print("\nPer-category:")
for cat in CATEGORIES:
    s = by_cat[cat]
    pct = 100 * s["pass"] / s["total"] if s["total"] else 0
    print(f"  {cat:22s}: {s['pass']:4d}/{s['total']:4d} ({pct:5.1f}%)")

by_tribe = defaultdict(lambda: {"pass": 0, "total": 0})
for rec in records:
    t = int(rec.get("tribe_idx", 0))
    by_tribe[t]["total"] += 1
    if rec.get("verdict") == "PASS":
        by_tribe[t]["pass"] += 1

print(f"\nPer-tribe routing:")
for t in sorted(by_tribe):
    s = by_tribe[t]
    pct_pass  = 100 * s["pass"]  / s["total"] if s["total"] else 0
    pct_route = 100 * s["total"] / n if n else 0
    print(f"  T{t}: {s['total']:4d} tasks ({pct_route:5.1f}% routed) | "
          f"{s['pass']:4d} passed ({pct_pass:5.1f}% pass rate)")

# Temporal routing drift (4 x 500-task windows)
print("\nRouting drift (500-task windows):")
for w in range(4):
    chunk = records[w*500:(w+1)*500]
    tc = defaultdict(int)
    for r in chunk:
        tc[int(r.get("tribe_idx", 0))] += 1
    total = len(chunk)
    parts = " | ".join(f"T{t}:{100*tc[t]/total:.0f}%" for t in sorted(tc))
    pr = 100 * sum(1 for r in chunk if r.get("verdict") == "PASS") / total
    print(f"  Tasks {w*500+1:4d}-{(w+1)*500:4d}: {parts}  pass={pr:.1f}%")
