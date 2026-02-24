"""MCN -- Experiment C: Routing Specialization Analysis.

Reads runs.jsonl produced by run_live_experiment.py and performs a rigorous
statistical analysis to determine whether routing specialization is occurring.

Analyses
--------
1. **Routing confusion matrix**: per-tribe x per-task-type assignment counts.
   If routing is uniform (no specialization), each cell ~ n_tasks / (n_tribes * n_types).
   If routing is specialised, certain tribes dominate certain task types.

2. **Per-tribe pass rates by task type**: Do different tribes solve different
   task categories better? This is the key evidence for specialization value.

3. **Chi-squared test of routing independence**: Tests H0: routing assignments
   are independent of task type (i.e., all task types are routed uniformly).
   A significant result (p < 0.05) means routing is NOT uniform -- the bandit
   has learned to route different task types differently.

4. **Temporal routing drift**: How routing evolves over time (first vs second
   half of the run) -- bandit convergence signal.

5. **Single-agent comparison**: If single_agent_runs.jsonl exists (from
   run_single_agent.py), compares per-task-type pass rates against MCN.

Usage:
    # After an MCN run:
    python analyze_routing.py --runs /results/runs.jsonl

    # With single-agent ablation for comparison:
    python analyze_routing.py \\
        --runs /results/runs.jsonl \\
        --single-agent /results/single_agent/single_agent_runs.jsonl

    # Docker Compose (volume mounted):
    docker run --rm -v mcn-results:/results mcn-runner \\
        python analyze_routing.py --runs /results/runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading + normalization
# ---------------------------------------------------------------------------

# Built from run_live_experiment.py LIVE_TASKS -- maps description prefix -> function_name.
# Needed to normalize old runs.jsonl files where task_type was stored as
# description[:50] instead of function_name (fixed in council.py, but old files
# still need this).
_DESC_PREFIX_TO_FNAME: dict[str, str] = {
    "Sort a list of integers": "sort_list",
    "Remove duplicate elements": "deduplicate",
    "Flatten a nested list": "flatten",
    "Partition a list into two lists": "partition",
    "Reverse a string": "reverse_string",
    "Check if a string is a palindrome": "is_palindrome",
    "Count the frequency of each word": "word_count",
    "Compute the nth Fibonacci number": "fibonacci",
    "Check if a positive integer is prime": "is_prime",
    "Compute the greatest common divisor": "gcd",
    "Invert a dictionary": "invert_dict",
    "Compute the running sum": "running_sum",
    "Given a sorted list of integers xs": "search_insert",
    "Find the length of the longest strictly increasing": "lis",
    "Given an integer n (n >= 0), compute the number of ways to climb": "climb_stairs",
    "Given a string, find the length of the longest sub": "longest_unique",
    "Given two strings s and t, return True if t is an": "is_anagram",
    "Given a string containing just brackets": "valid_brackets",
    "Return True if a positive integer n is a perfect": "is_perfect_square",
    "Given two non-negative integers m and n, compute C": "unique_paths",
    "Given an integer n (number of nodes": "has_cycle",
    "Given a list of integers, return all unique permut": "permutations",
}


def _normalize_task_type(tt: str) -> str:
    """Normalize task_type to function_name, handling both old (description)
    and new (function_name) formats in runs.jsonl."""
    # New format: already a short function name (no spaces, snake_case)
    if tt and " " not in tt and len(tt) < 30:
        return tt
    # Old format: description string -- look up by prefix
    for prefix, fname in _DESC_PREFIX_TO_FNAME.items():
        if tt.startswith(prefix):
            return fname
    # Unknown -- return as-is (truncated to 30 chars for display)
    return tt[:30] if len(tt) > 30 else tt


def load_runs(path: str) -> list[dict]:
    """Load a runs.jsonl file into a list of records.

    Normalizes task_type to function_name for consistent cross-file comparison,
    handling both old (description[:50]) and new (function_name) formats.
    """
    p = Path(path)
    if not p.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    records = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    # Normalize task_type to function_name
                    if "task_type" in rec:
                        rec["task_type"] = _normalize_task_type(rec["task_type"])
                    # Also normalize verdict field (council writes enum name, single-agent writes string)
                    if "verdict" in rec and isinstance(rec["verdict"], str):
                        rec["verdict"] = rec["verdict"].upper().replace("GATEVARDICT.", "").replace("GATEVERDICT.", "")
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
    return records


# ---------------------------------------------------------------------------
# Chi-squared test (scipy optional -- falls back to manual computation)
# ---------------------------------------------------------------------------

def chi_squared_test(observed: list[list[int]]) -> tuple[float, float, int]:
    """Compute chi-squared statistic and p-value for a contingency table.

    Args:
        observed: 2D list [row][col] of observed counts.

    Returns:
        (chi2_stat, p_value, degrees_of_freedom)
    """
    try:
        from scipy.stats import chi2_contingency
        import numpy as np
        chi2, p, dof, expected = chi2_contingency(np.array(observed))
        return float(chi2), float(p), int(dof)
    except ImportError:
        pass

    # Manual chi-squared computation (no scipy required)
    nrows = len(observed)
    ncols = len(observed[0])
    n = sum(sum(row) for row in observed)
    if n == 0:
        return 0.0, 1.0, (nrows - 1) * (ncols - 1)

    row_sums = [sum(observed[r]) for r in range(nrows)]
    col_sums = [sum(observed[r][c] for r in range(nrows)) for c in range(ncols)]

    chi2 = 0.0
    for r in range(nrows):
        for c in range(ncols):
            expected = row_sums[r] * col_sums[c] / n
            if expected > 0:
                chi2 += (observed[r][c] - expected) ** 2 / expected

    dof = (nrows - 1) * (ncols - 1)

    # p-value via chi2 CDF approximation (Wilson-Hilferty)
    try:
        import math
        # Use normal approximation for large dof
        z = (chi2 / dof) ** (1 / 3)
        mu = 1 - 2 / (9 * dof)
        sigma = (2 / (9 * dof)) ** 0.5
        z_score = (z - mu) / sigma
        # One-tailed p-value (right tail)
        p = 1 - 0.5 * (1 + math.erf(z_score / 2 ** 0.5))
    except Exception:
        p = float("nan")

    return chi2, p, dof


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def routing_confusion_matrix(
    records: list[dict],
    n_tribes: int | None = None,
) -> tuple[dict[str, dict[int, int]], list[str], list[int]]:
    """Build routing confusion matrix: task_type x tribe_idx -> count.

    Returns:
        (matrix, task_types, tribe_indices)
    """
    if n_tribes is None:
        n_tribes = max(r.get("tribe_idx", 0) for r in records) + 1

    # Collect all task types
    task_types = sorted(set(r.get("task_type", "unknown") for r in records))
    tribe_indices = list(range(n_tribes))

    # Build matrix: matrix[task_type][tribe_idx] = count
    matrix: dict[str, dict[int, int]] = {
        tt: {ti: 0 for ti in tribe_indices}
        for tt in task_types
    }

    for r in records:
        tt = r.get("task_type", "unknown")
        ti = r.get("tribe_idx", 0)
        if tt in matrix and ti in matrix[tt]:
            matrix[tt][ti] += 1

    return matrix, task_types, tribe_indices


def per_tribe_pass_rates(
    records: list[dict],
    n_tribes: int | None = None,
) -> dict[int, dict[str, dict[str, int]]]:
    """Compute per-tribe, per-task-type pass/fail counts.

    Returns:
        {tribe_idx: {task_type: {"pass": N, "fail": N, "total": N, "rate": float}}}
    """
    if n_tribes is None:
        n_tribes = max(r.get("tribe_idx", 0) for r in records) + 1

    result: dict[int, dict[str, dict]] = {
        ti: defaultdict(lambda: {"pass": 0, "fail": 0, "total": 0, "rate": 0.0})
        for ti in range(n_tribes)
    }

    for r in records:
        ti = r.get("tribe_idx", 0)
        tt = r.get("task_type", "unknown")
        verdict = r.get("verdict", "FAIL")
        if ti in result:
            result[ti][tt]["total"] += 1
            if verdict == "PASS":
                result[ti][tt]["pass"] += 1
            else:
                result[ti][tt]["fail"] += 1

    # Compute rates
    for ti in result:
        for tt in result[ti]:
            d = result[ti][tt]
            d["rate"] = d["pass"] / max(d["total"], 1)

    return result


def temporal_routing(
    records: list[dict],
    n_tribes: int | None = None,
) -> tuple[dict[int, int], dict[int, int]]:
    """Compare routing in first half vs second half.

    Returns:
        (first_half_counts, second_half_counts) -- dict[tribe_idx -> count]
    """
    if n_tribes is None:
        n_tribes = max(r.get("tribe_idx", 0) for r in records) + 1

    n = len(records)
    half = n // 2
    first_half = records[:half]
    second_half = records[half:]

    def count_tribes(recs: list[dict]) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for r in recs:
            counts[r.get("tribe_idx", 0)] += 1
        return dict(counts)

    return count_tribes(first_half), count_tribes(second_half)


# ---------------------------------------------------------------------------
# Phase 1C: Category-based utility metrics
# ---------------------------------------------------------------------------

def category_wise_delta(
    records: list[dict],
    ablation_records: list[dict] | None = None,
    n_tribes: int | None = None,
) -> dict[str, dict]:
    """Per-category delta: MCN pass rate minus best single-tribe pass rate.

    delta_C = pass_rate(MCN on category C) - pass_rate(best single tribe on C)

    When *ablation_records* are supplied (one record per task per tribe, from a
    per-tribe ablation sweep), the best-tribe rate comes from ablation data.
    Otherwise falls back to the empirically highest tribe pass rate seen inside
    the routing data (selection-biased proxy -- use with caution).

    Returns:
        {category: {
            "mcn_pass":        int,
            "mcn_total":       int,
            "mcn_rate":        float,
            "best_tribe":      int,
            "best_tribe_rate": float,
            "delta":           float,
            "source":          "ablation" | "internal",
        }}
    """
    if n_tribes is None:
        n_tribes = max((r.get("tribe_idx", 0) for r in records), default=0) + 1

    categories = sorted(set(r.get("category", "unknown") for r in records))

    # MCN pass rates per category
    mcn_cat: dict[str, dict] = {c: {"pass": 0, "total": 0} for c in categories}
    for r in records:
        c = r.get("category", "unknown")
        if c in mcn_cat:
            mcn_cat[c]["total"] += 1
            if r.get("verdict") == "PASS":
                mcn_cat[c]["pass"] += 1

    # Per-tribe pass rates per category -- ablation data or internal routing data
    ref_data = ablation_records if ablation_records else records
    source = "ablation" if ablation_records else "internal"
    ref_cat: dict[str, dict[int, dict]] = {
        c: {ti: {"pass": 0, "total": 0} for ti in range(n_tribes)}
        for c in categories
    }
    for r in ref_data:
        c = r.get("category", "unknown")
        ti = r.get("tribe_idx", 0)
        if c in ref_cat and ti in ref_cat[c]:
            ref_cat[c][ti]["total"] += 1
            if r.get("verdict") == "PASS":
                ref_cat[c][ti]["pass"] += 1

    result: dict[str, dict] = {}
    for c in categories:
        d = mcn_cat[c]
        mcn_rate = d["pass"] / max(d["total"], 1)

        best_ti, best_rate = 0, 0.0
        for ti in range(n_tribes):
            td = ref_cat[c][ti]
            if td["total"] > 0:
                rate = td["pass"] / td["total"]
                if rate > best_rate:
                    best_rate = rate
                    best_ti = ti

        result[c] = {
            "mcn_pass":        d["pass"],
            "mcn_total":       d["total"],
            "mcn_rate":        mcn_rate,
            "best_tribe":      best_ti,
            "best_tribe_rate": best_rate,
            "delta":           mcn_rate - best_rate,
            "source":          source,
        }
    return result


def tribe_solve_rate_when_assigned(
    records: list[dict],
    n_tribes: int | None = None,
) -> dict[int, dict[str, dict]]:
    """Per-tribe, per-category solve rate conditional on being assigned.

    solve_rate(T, C) = tasks_solved(T, C) / tasks_assigned_to_T_in_C

    Only counts attempts where tribe T was actually routed the task.  A tribe
    with a high solve rate when assigned suggests the router is sending it tasks
    it is genuinely good at.

    Returns:
        {tribe_idx: {category: {"pass": int, "total": int, "rate": float}}}
    """
    if n_tribes is None:
        n_tribes = max((r.get("tribe_idx", 0) for r in records), default=0) + 1

    result: dict[int, dict[str, dict]] = {
        ti: defaultdict(lambda: {"pass": 0, "total": 0, "rate": 0.0})
        for ti in range(n_tribes)
    }

    for r in records:
        ti = r.get("tribe_idx", 0)
        cat = r.get("category", "unknown")
        if ti in result:
            result[ti][cat]["total"] += 1
            if r.get("verdict") == "PASS":
                result[ti][cat]["pass"] += 1

    for ti in result:
        for cat in result[ti]:
            d = result[ti][cat]
            d["rate"] = d["pass"] / max(d["total"], 1)

    return result


def oracle_gap_decomposed(
    records: list[dict],
    n_tribes: int | None = None,
) -> dict:
    """Decompose the oracle gap into exploration cost and exploitation error.

    Definitions
    -----------
    oracle_rate : Weighted average of empirical best-tribe pass rate per
                  category.  Upper bound assuming perfect routing knowledge.
    actual_rate : Observed MCN pass rate across all records.
    total_gap   : oracle_rate - actual_rate.

    Quartile breakdown (records ordered by time / index)
    ---------------------------------------------------
    Q1 (first 25%)   -> exploration phase, bandit still learning
    Q2+Q3 (mid 50%)  -> mixed: partial convergence + tie-breaking noise
    Q4 (last 25%)    -> exploitation phase, bandit mostly converged

    exploration_cost    = oracle_gap(Q1)    x |Q1|    / N
    tie_breaking_noise  = oracle_gap(Q2+Q3) x |Q2+Q3| / N
    exploitation_error  = oracle_gap(Q4)    x |Q4|    / N

    Returns:
        {
            "oracle_rate":        float,
            "actual_rate":        float,
            "total_gap":          float,
            "exploration_cost":   float,
            "tie_breaking_noise": float,
            "exploitation_error": float,
            "categories": {cat: {
                "oracle_tribe": int,
                "oracle_rate":  float,
                "actual_rate":  float,
                "actual_count": int,
            }},
        }
    """
    if n_tribes is None:
        n_tribes = max((r.get("tribe_idx", 0) for r in records), default=0) + 1

    n = len(records)
    if n == 0:
        return {}

    cats = sorted(set(r.get("category", "unknown") for r in records))

    # Per-category, per-tribe counts
    cat_tribe: dict[str, dict[int, dict]] = {
        c: {ti: {"pass": 0, "total": 0} for ti in range(n_tribes)}
        for c in cats
    }
    for r in records:
        c = r.get("category", "unknown")
        ti = r.get("tribe_idx", 0)
        if c in cat_tribe and ti in cat_tribe[c]:
            cat_tribe[c][ti]["total"] += 1
            if r.get("verdict") == "PASS":
                cat_tribe[c][ti]["pass"] += 1

    # Determine oracle tribe and rates per category
    oracle_tribe: dict[str, int] = {}
    oracle_rate_per_cat: dict[str, float] = {}
    actual_rate_per_cat: dict[str, float] = {}
    actual_count_per_cat: dict[str, int] = {}

    for c in cats:
        best_ti, best_rate = 0, 0.0
        for ti in range(n_tribes):
            d = cat_tribe[c][ti]
            if d["total"] > 0:
                r_ti = d["pass"] / d["total"]
                if r_ti > best_rate:
                    best_rate = r_ti
                    best_ti = ti
        oracle_tribe[c] = best_ti
        oracle_rate_per_cat[c] = best_rate

        total_c = sum(cat_tribe[c][ti]["total"] for ti in range(n_tribes))
        pass_c  = sum(cat_tribe[c][ti]["pass"]  for ti in range(n_tribes))
        actual_rate_per_cat[c]  = pass_c / max(total_c, 1)
        actual_count_per_cat[c] = total_c

    total_n = sum(actual_count_per_cat.values()) or 1
    oracle_rate_overall = sum(
        oracle_rate_per_cat[c] * actual_count_per_cat[c] for c in cats
    ) / total_n
    actual_rate_overall = sum(1 for r in records if r.get("verdict") == "PASS") / n

    def _quartile_gap(slice_: list[dict]) -> float:
        """Oracle gap in a slice of records (positive = oracle > actual)."""
        if not slice_:
            return 0.0
        s_n = len(slice_)
        s_pass = sum(1 for r in slice_ if r.get("verdict") == "PASS")
        s_actual = s_pass / s_n
        s_cat_cnt: dict[str, int] = defaultdict(int)
        for r in slice_:
            s_cat_cnt[r.get("category", "unknown")] += 1
        s_total = sum(s_cat_cnt.values()) or 1
        s_oracle = sum(
            oracle_rate_per_cat.get(c, 0.0) * cnt
            for c, cnt in s_cat_cnt.items()
        ) / s_total
        return s_oracle - s_actual

    q1_end = n // 4
    q3_end = 3 * n // 4

    q1  = records[:q1_end]
    q23 = records[q1_end:q3_end]
    q4  = records[q3_end:]

    exploration_cost   = _quartile_gap(q1)  * (len(q1)  / n)
    tie_breaking_noise = _quartile_gap(q23) * (len(q23) / n)
    exploitation_error = _quartile_gap(q4)  * (len(q4)  / n)

    return {
        "oracle_rate":        oracle_rate_overall,
        "actual_rate":        actual_rate_overall,
        "total_gap":          oracle_rate_overall - actual_rate_overall,
        "exploration_cost":   exploration_cost,
        "tie_breaking_noise": tie_breaking_noise,
        "exploitation_error": exploitation_error,
        "categories": {
            c: {
                "oracle_tribe": oracle_tribe[c],
                "oracle_rate":  oracle_rate_per_cat[c],
                "actual_rate":  actual_rate_per_cat[c],
                "actual_count": actual_count_per_cat[c],
            }
            for c in cats
        },
    }


def learning_curve_by_category(
    records: list[dict],
    window: int = 10,
) -> dict[str, list[tuple[int, float, float]]]:
    """Rolling and cumulative pass rate per category over time.

    Args:
        records : All run records in chronological order.
        window  : Rolling window size (in category-specific attempts).

    Returns:
        {category: [(global_idx, cumulative_rate, rolling_rate), ...]}

        global_idx      -- position in the full records list (0-based)
        cumulative_rate -- pass rate over all category attempts seen so far
        rolling_rate    -- pass rate over last *window* category attempts
    """
    cats = sorted(set(r.get("category", "unknown") for r in records))
    result: dict[str, list[tuple[int, float, float]]] = {c: [] for c in cats}
    cat_hist: dict[str, list[int]] = {c: [] for c in cats}

    for idx, r in enumerate(records):
        c = r.get("category", "unknown")
        outcome = 1 if r.get("verdict") == "PASS" else 0
        cat_hist[c].append(outcome)

        hist = cat_hist[c]
        n_h = len(hist)
        cumulative = sum(hist) / n_h
        roll_win = hist[-window:] if n_h >= window else hist
        rolling = sum(roll_win) / len(roll_win)
        result[c].append((idx, cumulative, rolling))

    return result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _bar(value: float, max_val: float, width: int = 20) -> str:
    """Simple ASCII bar chart."""
    filled = int(round(width * value / max(max_val, 1)))
    return "#" * filled + "." * (width - filled)


def print_full_report(
    records: list[dict],
    single_records: list[dict] | None = None,
    ablation_records: list[dict] | None = None,
    n_tribes: int | None = None,
) -> None:
    """Print the full routing specialization analysis."""

    if not records:
        print("  ERROR: No records to analyze.")
        return

    n_tribes = n_tribes or (max(r.get("tribe_idx", 0) for r in records) + 1)
    n_total = len(records)
    n_pass = sum(1 for r in records if r.get("verdict") == "PASS")

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT C -- ROUTING SPECIALIZATION ANALYSIS")
    print(f"{'=' * 70}")
    print(f"\n  Dataset: {n_total} tasks, {n_pass} passed ({100*n_pass/max(n_total,1):.1f}%)")
    print(f"  Tribes:  {n_tribes}")

    # -----------------------------------------------------------------------
    # 1. Overall routing distribution
    # -----------------------------------------------------------------------
    print(f"\n  [1] OVERALL ROUTING DISTRIBUTION")
    print(f"  {'Tribe':>8}  {'Count':>6}  {'Pct':>6}  Bar")
    print(f"  {'-'*50}")

    tribe_total_counts: dict[int, int] = defaultdict(int)
    tribe_pass_counts: dict[int, int] = defaultdict(int)
    for r in records:
        ti = r.get("tribe_idx", 0)
        tribe_total_counts[ti] += 1
        if r.get("verdict") == "PASS":
            tribe_pass_counts[ti] += 1

    max_count = max(tribe_total_counts.values(), default=1)
    for ti in sorted(tribe_total_counts):
        cnt = tribe_total_counts[ti]
        pct = 100 * cnt / max(n_total, 1)
        passed = tribe_pass_counts.get(ti, 0)
        pass_rate = 100 * passed / max(cnt, 1)
        bar = _bar(cnt, max_count)
        print(f"  Tribe {ti:>2}:  {cnt:>6}  {pct:>5.1f}%  {bar}  ({pass_rate:.0f}% pass)")

    uniform_pct = 100 / n_tribes
    print(f"  Expected uniform: {uniform_pct:.1f}% per tribe")

    # -----------------------------------------------------------------------
    # 2. Routing confusion matrix (task_type x tribe)
    # -----------------------------------------------------------------------
    matrix, task_types, tribe_indices = routing_confusion_matrix(records, n_tribes)

    print(f"\n  [2] ROUTING CONFUSION MATRIX  (rows=task type, cols=tribe assigned)")
    col_header = f"  {'Task type':20s} | " + " | ".join(f"T{ti:>3}" for ti in tribe_indices)
    print(f"  {'-' * len(col_header)}")
    print(col_header)
    print(f"  {'-' * len(col_header)}")

    # For chi-squared test
    observed: list[list[int]] = []

    for tt in task_types:
        row = [matrix[tt][ti] for ti in tribe_indices]
        observed.append(row)
        row_total = sum(row)
        # Normalize row to show which tribe dominates for this task type
        dominant_tribe = max(tribe_indices, key=lambda ti: matrix[tt][ti])
        dominant_pct = 100 * matrix[tt][dominant_tribe] / max(row_total, 1)
        row_str = " | ".join(f"{v:>4}" for v in row)
        print(
            f"  {tt:20s} | {row_str}   "
            f"-> T{dominant_tribe} ({dominant_pct:.0f}%)"
        )

    print(f"  {'-' * len(col_header)}")

    # -----------------------------------------------------------------------
    # 3. Chi-squared test of routing independence
    # -----------------------------------------------------------------------
    print(f"\n  [3] STATISTICAL TEST: Is routing independent of task type?")
    print(f"      H0: routing assignments are uniform across task types (no specialization)")
    print(f"      H1: routing assignments are NOT uniform (specialization detected)")

    # Only run if we have enough data
    row_sums = [sum(row) for row in observed]
    col_sums = [sum(observed[r][c] for r in range(len(observed))) for c in range(n_tribes)]

    sufficient = all(s > 0 for s in row_sums) and all(s > 0 for s in col_sums)

    if len(task_types) < 2 or n_tribes < 2:
        print(f"      SKIP: need >= 2 task types and >= 2 tribes (got {len(task_types)} x {n_tribes})")
    elif not sufficient:
        print(f"      SKIP: some rows/columns are empty -- need more data")
    else:
        chi2, p, dof = chi_squared_test(observed)
        sig = "SIGNIFICANT" if (p < 0.05 and not (p != p)) else "NOT significant"
        print(f"      Chi2 = {chi2:.3f}, df = {dof}, p = {p:.4f}")
        print(f"      Result: {sig} (alpha=0.05)")
        if p < 0.05:
            print(f"      => Routing IS dependent on task type (specialization evidence)")
        else:
            print(f"      => Routing is NOT dependent on task type (no specialization)")

    # -----------------------------------------------------------------------
    # 4. Per-tribe, per-task-type pass rates
    # -----------------------------------------------------------------------
    tribe_rates = per_tribe_pass_rates(records, n_tribes)

    print(f"\n  [4] PER-TRIBE PASS RATES BY TASK TYPE")
    print(f"      (High variance across tribes = different strengths)")

    # Header
    header = f"  {'Task type':20s} | " + " | ".join(f"T{ti}pass" for ti in tribe_indices)
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for tt in task_types:
        rates = []
        for ti in tribe_indices:
            if tt in tribe_rates.get(ti, {}):
                d = tribe_rates[ti][tt]
                rate_str = f"{100*d['rate']:>5.0f}%"
                rates.append(rate_str)
            else:
                rates.append("   N/A")
        print(f"  {tt:20s} | " + " | ".join(f"{r:>6}" for r in rates))

    # -----------------------------------------------------------------------
    # 5. Temporal routing drift
    # -----------------------------------------------------------------------
    first_half, second_half = temporal_routing(records, n_tribes)

    print(f"\n  [5] TEMPORAL ROUTING DRIFT  (bandit convergence signal)")
    print(f"  {'Tribe':>8}  {'First half':>12}  {'Second half':>12}  Change")
    print(f"  {'-'*50}")
    for ti in sorted(set(list(first_half) + list(second_half))):
        f1 = first_half.get(ti, 0)
        f2 = second_half.get(ti, 0)
        n_half = len(records) // 2
        p1 = 100 * f1 / max(n_half, 1)
        p2 = 100 * f2 / max(n_half, 1)
        change = p2 - p1
        arrow = "^" if change > 3 else ("v" if change < -3 else "~")
        print(f"  Tribe {ti:>2}:  {p1:>10.1f}%  {p2:>11.1f}%  {arrow} {change:+.1f}pp")

    # -----------------------------------------------------------------------
    # 6. Single-agent comparison (if available)
    # -----------------------------------------------------------------------
    if single_records:
        print(f"\n  [6] SINGLE-AGENT ABLATION COMPARISON")
        single_pass = sum(1 for r in single_records if r.get("verdict") == "PASS")
        single_total = len(single_records)
        single_rate = 100 * single_pass / max(single_total, 1)
        mcn_rate = 100 * n_pass / max(n_total, 1)

        print(f"  {'System':20s}  {'Pass rate':>10}  {'Pass count':>12}")
        print(f"  {'-'*50}")
        print(f"  {'MCN (multi-tribe)':20s}  {mcn_rate:>9.1f}%  {n_pass:>8}/{n_total}")
        print(f"  {'Single agent':20s}  {single_rate:>9.1f}%  {single_pass:>8}/{single_total}")
        print()

        diff = mcn_rate - single_rate
        if abs(diff) < 2.0:
            verdict_str = "NO CLEAR BENEFIT (< 2pp difference -- within noise)"
        elif diff > 0:
            verdict_str = f"MCN BETTER by {diff:.1f}pp (routing adds value)"
        else:
            verdict_str = f"SINGLE AGENT BETTER by {-diff:.1f}pp (routing hurts)"
        print(f"  Verdict: {verdict_str}")
        print()
        print(f"  Per-task-type comparison:")
        single_by_type: dict[str, dict] = defaultdict(lambda: {"pass": 0, "total": 0})
        for r in single_records:
            tt = r.get("task_type", "unknown")
            single_by_type[tt]["total"] += 1
            if r.get("verdict") == "PASS":
                single_by_type[tt]["pass"] += 1

        mcn_by_type: dict[str, dict] = defaultdict(lambda: {"pass": 0, "total": 0})
        for r in records:
            tt = r.get("task_type", "unknown")
            mcn_by_type[tt]["total"] += 1
            if r.get("verdict") == "PASS":
                mcn_by_type[tt]["pass"] += 1

        all_types = sorted(set(list(single_by_type) + list(mcn_by_type)))
        print(f"  {'Task type':20s} | {'MCN':>8} | {'Single':>8} | {'Diff':>8}")
        print(f"  {'-'*56}")
        for tt in all_types:
            mcn_d = mcn_by_type.get(tt, {"pass": 0, "total": 0})
            sa_d = single_by_type.get(tt, {"pass": 0, "total": 0})
            mcn_r = 100 * mcn_d["pass"] / max(mcn_d["total"], 1)
            sa_r = 100 * sa_d["pass"] / max(sa_d["total"], 1)
            diff_r = mcn_r - sa_r
            sign = "+" if diff_r > 0 else ""
            print(f"  {tt:20s} | {mcn_r:>7.0f}% | {sa_r:>7.0f}% | {sign}{diff_r:>6.1f}pp")

    # -----------------------------------------------------------------------
    # 7-10. Phase 1C category metrics (only when records carry 'category')
    # -----------------------------------------------------------------------
    has_categories = sum(1 for r in records if r.get("category")) > len(records) * 0.5

    if has_categories:
        # --- 7. Category-wise delta ---
        cat_delta = category_wise_delta(
            records, ablation_records=ablation_records, n_tribes=n_tribes
        )
        src_note = list(cat_delta.values())[0]["source"] if cat_delta else "internal"
        print(f"\n  [7] CATEGORY-WISE DELTA  (MCN rate - best single-tribe rate)")
        print(f"      delta_C > 0: routing adds value  |  delta_C < 0: routing hurts")
        print(f"      Best-tribe source: {src_note}")
        print(f"\n  {'Category':20s} | {'MCN rate':>8} | {'Best T':>6} | {'Best rate':>9} | {'Delta':>9}")
        print(f"  {'-'*65}")
        for cat, d in sorted(cat_delta.items()):
            mcn_r  = 100 * d["mcn_rate"]
            best_r = 100 * d["best_tribe_rate"]
            delta  = 100 * d["delta"]
            sign   = "+" if delta >= 0 else ""
            n_note = f"n={d['mcn_total']}"
            print(
                f"  {cat:20s} | {mcn_r:>7.1f}% | "
                f"T{d['best_tribe']:>4} | {best_r:>8.1f}% | "
                f"{sign}{delta:>7.1f}pp  ({n_note})"
            )
        total_n_cat = sum(d["mcn_total"] for d in cat_delta.values()) or 1
        wtd_mcn  = sum(d["mcn_rate"]        * d["mcn_total"] for d in cat_delta.values()) / total_n_cat
        wtd_best = sum(d["best_tribe_rate"] * d["mcn_total"] for d in cat_delta.values()) / total_n_cat
        wtd_delta = wtd_mcn - wtd_best
        sign = "+" if wtd_delta >= 0 else ""
        print(f"  {'-'*65}")
        print(
            f"  {'Weighted overall':20s} | {100*wtd_mcn:>7.1f}% | "
            f"{'---':>6} | {100*wtd_best:>8.1f}% | {sign}{100*wtd_delta:>7.1f}pp"
        )

        # --- 8. Tribe solve rate when assigned ---
        solve_rates = tribe_solve_rate_when_assigned(records, n_tribes)
        all_cats_sr = sorted(set(cat for ti in solve_rates for cat in solve_rates[ti]))
        print(f"\n  [8] TRIBE SOLVE RATE WHEN ASSIGNED")
        print(f"      solve_rate(T,C) = tasks solved / tasks assigned by router")
        hdr_tribes = " | ".join(f"  T{ti}" for ti in sorted(solve_rates))
        print(f"\n  {'Category':20s} | {hdr_tribes}")
        print(f"  {'-'*65}")
        for cat in all_cats_sr:
            cells = []
            for ti in sorted(solve_rates):
                d = solve_rates[ti].get(cat, {"pass": 0, "total": 0, "rate": 0.0})
                n_t = d["total"]
                if n_t > 0:
                    cells.append(f"{100*d['rate']:>3.0f}% (n={n_t:>2})")
                else:
                    cells.append("--- (n= 0)")
            print(f"  {cat:20s} | " + " | ".join(cells))

        # --- 9. Oracle gap decomposition ---
        og = oracle_gap_decomposed(records, n_tribes)
        print(f"\n  [9] ORACLE GAP DECOMPOSITION")
        if og:
            oracle_r = 100 * og["oracle_rate"]
            actual_r = 100 * og["actual_rate"]
            total_g  = 100 * og["total_gap"]
            exp_c    = 100 * og["exploration_cost"]
            tbk_n    = 100 * og["tie_breaking_noise"]
            expl_e   = 100 * og["exploitation_error"]
            print(f"  Oracle rate (best-tribe routing): {oracle_r:>6.1f}%")
            print(f"  Actual MCN rate:                  {actual_r:>6.1f}%")
            print(f"  Total gap:                        {total_g:>+6.1f}pp")
            print()
            print(f"  Gap breakdown:")
            print(f"    Exploration cost    (Q1, first 25%):   {exp_c:>+6.2f}pp")
            print(f"    Tie-breaking noise  (Q2+Q3, mid 50%):  {tbk_n:>+6.2f}pp")
            print(f"    Exploitation error  (Q4, last 25%):    {expl_e:>+6.2f}pp")
            print()
            print(f"  Per-category oracle:")
            print(f"  {'Category':20s} | Oracle T | Oracle rate | MCN rate |    Gap")
            print(f"  {'-'*65}")
            for cat, cd in sorted(og["categories"].items()):
                gap_pp = 100 * (cd["oracle_rate"] - cd["actual_rate"])
                print(
                    f"  {cat:20s} | T{cd['oracle_tribe']:>6}  | "
                    f"{100*cd['oracle_rate']:>10.1f}% | "
                    f"{100*cd['actual_rate']:>7.1f}% | "
                    f"{gap_pp:>+5.1f}pp  (n={cd['actual_count']})"
                )

        # --- 10. Learning curves by category ---
        lc = learning_curve_by_category(records, window=10)
        print(f"\n  [10] LEARNING CURVES BY CATEGORY  (cumulative pass rate)")
        print(f"       Q1=first 25%, Q2=next 25%, Q3=next 25%, Q4=last 25% of category attempts")
        print()
        print(f"  {'Category':20s} | {'Q1':>7} | {'Q2':>7} | {'Q3':>7} | {'Q4':>7} | Trend (n)")
        print(f"  {'-'*70}")
        for cat in sorted(lc):
            pts = lc[cat]
            n_pts = len(pts)
            if n_pts == 0:
                continue
            q1_r = pts[max(0, n_pts // 4 - 1)][1]
            q2_r = pts[max(0, n_pts // 2 - 1)][1]
            q3_r = pts[max(0, 3 * n_pts // 4 - 1)][1]
            q4_r = pts[-1][1]
            trend = q4_r - q1_r
            sign  = "+" if trend >= 0 else ""
            arrow = "^" if trend > 0.03 else ("v" if trend < -0.03 else "~")
            print(
                f"  {cat:20s} | {100*q1_r:>6.1f}% | {100*q2_r:>6.1f}% | "
                f"{100*q3_r:>6.1f}% | {100*q4_r:>6.1f}% | "
                f"{arrow} {sign}{100*trend:.1f}pp  (n={n_pts})"
            )
    else:
        print(f"\n  [7-10] CATEGORY METRICS: SKIPPED")
        print(f"         (Records lack 'category' field -- re-run with Phase 1B task set)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")

    # Gini coefficient of routing distribution (0=uniform, 1=fully concentrated)
    counts = [tribe_total_counts.get(ti, 0) for ti in range(n_tribes)]
    total_c = sum(counts) or 1
    fracs = sorted(c / total_c for c in counts)
    n = len(fracs)
    if n > 1:
        gini = sum((2 * (i + 1) - n - 1) * fracs[i] for i in range(n)) / (n * sum(fracs))
    else:
        gini = 0.0

    print(f"  Routing Gini coefficient: {gini:.3f}  (0=uniform, 1=fully concentrated)")
    if gini < 0.1:
        print(f"  Routing is near-uniform -- no specialization detectable.")
    elif gini < 0.3:
        print(f"  Routing shows mild concentration.")
    else:
        print(f"  Routing shows strong concentration (potential specialization).")

    print(f"\n{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCN Experiment C: Routing Specialization Analysis",
    )
    parser.add_argument(
        "--runs", type=str, required=True,
        help="Path to runs.jsonl from an MCN experiment",
    )
    parser.add_argument(
        "--single-agent", type=str, default="",
        help="Path to single_agent_runs.jsonl (from run_single_agent.py)",
    )
    parser.add_argument(
        "--n-tribes", type=int, default=0,
        help="Number of tribes (auto-detected from data if 0)",
    )
    parser.add_argument(
        "--ablation", type=str, default="",
        help="Path to tribal ablation runs.jsonl for category-wise delta (optional)",
    )
    parser.add_argument(
        "--save", type=str, default="",
        help="Optional path to save the analysis as a JSON report",
    )
    args = parser.parse_args()

    print(f"\n  Loading MCN runs from: {args.runs}")
    records = load_runs(args.runs)
    print(f"  Loaded {len(records)} records")

    single_records = None
    if args.single_agent:
        print(f"  Loading single-agent runs from: {args.single_agent}")
        single_records = load_runs(args.single_agent)
        print(f"  Loaded {len(single_records)} single-agent records")

    ablation_records = None
    if args.ablation:
        print(f"  Loading tribal ablation runs from: {args.ablation}")
        ablation_records = load_runs(args.ablation)
        print(f"  Loaded {len(ablation_records)} ablation records")

    n_tribes = args.n_tribes if args.n_tribes > 0 else None

    print_full_report(
        records,
        single_records=single_records,
        ablation_records=ablation_records,
        n_tribes=n_tribes,
    )

    # Optional JSON dump for further analysis
    if args.save:
        matrix, task_types, tribe_indices = routing_confusion_matrix(records, n_tribes)
        tribe_rates = per_tribe_pass_rates(records, n_tribes)
        first_half, second_half = temporal_routing(records, n_tribes)

        report = {
            "n_records": len(records),
            "n_pass": sum(1 for r in records if r.get("verdict") == "PASS"),
            "n_tribes": n_tribes or (max(r.get("tribe_idx", 0) for r in records) + 1),
            "task_types": task_types,
            "routing_matrix": {
                tt: {str(ti): matrix[tt][ti] for ti in tribe_indices}
                for tt in task_types
            },
            "tribe_total_counts": {
                str(ti): sum(1 for r in records if r.get("tribe_idx") == ti)
                for ti in tribe_indices
            },
            "temporal": {
                "first_half": {str(k): v for k, v in first_half.items()},
                "second_half": {str(k): v for k, v in second_half.items()},
            },
        }

        # Phase 1C: category metrics (if available)
        has_cat = sum(1 for r in records if r.get("category")) > len(records) * 0.5
        if has_cat:
            cat_delta = category_wise_delta(records, ablation_records=ablation_records, n_tribes=n_tribes)
            og        = oracle_gap_decomposed(records, n_tribes)
            report["category_delta"] = {
                c: {k: (v if not isinstance(v, float) else round(v, 4)) for k, v in d.items()}
                for c, d in cat_delta.items()
            }
            report["oracle_gap"] = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in og.items()
                if k != "categories"
            }
            report["oracle_gap"]["categories"] = og.get("categories", {})

        with open(args.save, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Analysis saved to: {args.save}")


if __name__ == "__main__":
    main()
