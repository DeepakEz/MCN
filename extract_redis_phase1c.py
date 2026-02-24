"""Extract Phase 1C runs (mcn:runs, 2000 tasks) from Redis → categorized_runs_phase1c.jsonl"""
import redis
import json
import sys
from collections import defaultdict

sys.path.insert(0, "/app")

TASK_CATEGORY_MAP = {
    "sort_list": "data_structures", "deduplicate": "data_structures",
    "flatten": "recursive", "partition": "data_structures",
    "reverse_string": "string", "is_palindrome": "string",
    "is_anagram": "string", "longest_unique": "string",
    "word_count": "parsing", "invert_dict": "data_structures",
    "running_sum": "iterative", "search_insert": "iterative",
    "merge_intervals": "data_structures", "valid_brackets": "data_structures",
    "has_cycle": "graph", "fibonacci": "recursive",
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

r = redis.from_url("redis://redis:6379/0", decode_responses=True)

entries = r.xrange("mcn:runs")
print(f"Found {len(entries)} entries in mcn:runs stream")

records = []
for _id, fields in entries:
    rec = {k: json.loads(v) for k, v in fields.items()}
    task_type = rec.get("task_type", "")
    rec["category"] = TASK_CATEGORY_MAP.get(task_type, rec.get("category", "unknown"))
    rec["run"]      = rec.get("run_number", rec.get("run", 0))
    rec["tokens"]   = rec.get("generation_tokens", rec.get("tokens", 0))
    rec["exception"]= rec.get("exception_type") or rec.get("exception") or None
    records.append(rec)

out = "/results/categorized_runs_phase1c.jsonl"
with open(out, "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")
print(f"Saved {len(records)} records → {out}")

# ── Summary ──────────────────────────────────────────────────────────────────
n_pass = sum(1 for r in records if r.get("verdict") == "PASS")
print(f"\nOverall: {n_pass}/{len(records)} passed ({100*n_pass/len(records):.1f}%)")

# Per-category
by_cat = defaultdict(lambda: {"pass": 0, "total": 0})
for rec in records:
    c = rec["category"]
    by_cat[c]["total"] += 1
    if rec.get("verdict") == "PASS":
        by_cat[c]["pass"] += 1

print("\nPer-category:")
for cat in sorted(by_cat):
    s = by_cat[cat]
    pct = 100 * s["pass"] / s["total"] if s["total"] else 0
    print(f"  {cat:22s}: {s['pass']:4d}/{s['total']:4d} ({pct:5.1f}%)")

# Per-tribe routing and pass rate
by_tribe = defaultdict(lambda: {"pass": 0, "total": 0})
for rec in records:
    t = int(rec.get("tribe_idx", 0))
    by_tribe[t]["total"] += 1
    if rec.get("verdict") == "PASS":
        by_tribe[t]["pass"] += 1

total_routed = sum(s["total"] for s in by_tribe.values())
print(f"\nPer-tribe routing (total routed: {total_routed}):")
for t in sorted(by_tribe):
    s = by_tribe[t]
    pct_pass  = 100 * s["pass"] / s["total"] if s["total"] else 0
    pct_route = 100 * s["total"] / len(records) if records else 0
    print(f"  Tribe {t}: {s['total']:4d} tasks ({pct_route:4.1f}% of tasks) | "
          f"{s['pass']:4d} passed ({pct_pass:5.1f}%)")

# Exception breakdown
from collections import Counter
exc_counts = Counter(
    rec["exception"] for rec in records
    if rec.get("verdict") == "FAIL" and rec.get("exception")
)
print(f"\nTop failure exceptions ({sum(exc_counts.values())} total failures):")
for exc, n in exc_counts.most_common(8):
    print(f"  {exc:30s}: {n}")

# Temporal drift — compare first/second half routing
half = len(records) // 2
first_half  = records[:half]
second_half = records[half:]
print("\nRouting drift (first 1000 vs last 1000 tasks):")
for h_name, h_recs in [("First 1000", first_half), ("Last 1000", second_half)]:
    c = Counter(int(r.get("tribe_idx", 0)) for r in h_recs)
    total = sum(c.values())
    dist = " / ".join(f"T{t}={c[t]}({100*c[t]/total:.0f}%)" for t in sorted(c))
    n_p = sum(1 for r in h_recs if r.get("verdict") == "PASS")
    print(f"  {h_name}: {dist} | pass={100*n_p/len(h_recs):.1f}%")
