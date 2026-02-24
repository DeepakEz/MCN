"""Extract all MCN runs from Redis, enrich with categories, save to /results/categorized_runs.jsonl"""
import redis
import json
import sys

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
    # All values are JSON-encoded — deserialize each one
    rec = {k: json.loads(v) for k, v in fields.items()}
    task_type = rec.get("task_type", "")
    rec["category"] = TASK_CATEGORY_MAP.get(task_type, "unknown")
    rec["run"] = rec.get("run_number", 0)
    rec["tokens"] = rec.get("generation_tokens", 0)
    rec["exception"] = rec.get("exception_type") or None
    records.append(rec)

out = "/results/categorized_runs.jsonl"
with open(out, "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")
print(f"Saved {len(records)} records to {out}")

# Quick summary
from collections import defaultdict
n_pass = sum(1 for r in records if r.get("verdict") == "PASS")
print(f"\nOverall: {n_pass}/{len(records)} passed ({100*n_pass/len(records):.1f}%)")

by_cat = defaultdict(lambda: {"pass": 0, "total": 0})
for rec in records:
    c = rec["category"]
    by_cat[c]["total"] += 1
    if rec.get("verdict") == "PASS":
        by_cat[c]["pass"] += 1

print("\nPer-category:")
for cat in sorted(by_cat):
    s = by_cat[cat]
    print(f"  {cat:22s}: {s['pass']:3d}/{s['total']:3d} ({100*s['pass']/s['total']:5.1f}%)")

by_tribe = defaultdict(lambda: {"pass": 0, "total": 0})
for rec in records:
    t = int(rec.get("tribe_idx", 0))
    by_tribe[t]["total"] += 1
    if rec.get("verdict") == "PASS":
        by_tribe[t]["pass"] += 1

print("\nPer-tribe routing:")
for t in sorted(by_tribe):
    s = by_tribe[t]
    print(f"  Tribe {t}: {s['total']:3d} tasks, {s['pass']:3d} passed ({100*s['pass']/s['total']:5.1f}%)")
