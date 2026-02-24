"""MCN Phase 4 — Live LLM Experiment.

Runs the full MCN pipeline with real vLLM inference on local hardware.
Validates infrastructure, warms up, runs a diverse experiment, produces a report.

Prerequisites:
    - vLLM running at MCN_VLLM_URL (default: http://localhost:8000/v1)
    - Redis running at MCN_REDIS_URL (optional, for state persistence)
    - mcn-sandbox:v0.1 Docker image built
    - Docker socket accessible

Usage:
    # Inside docker compose (recommended):
    docker compose up --build

    # Standalone (start vLLM + Redis first):
    python run_live_experiment.py -n 30 --log-dir /results

    # With mock mode (no GPU needed, for testing the script itself):
    python run_live_experiment.py --mock -n 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

from mcn.config import MCNConfig
from mcn.protocol import Task


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    fmt = "[%(asctime)s] %(levelname)-7s %(name)-20s %(message)s"
    logging.basicConfig(
        level=level, format=fmt, datefmt="%H:%M:%S", stream=sys.stdout,
    )
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("docker").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Diverse task library (Phase 4 — real LLM tasks)
# ---------------------------------------------------------------------------

# Each task has: description, function_name, input_signature, unit_tests

LIVE_TASKS = [
    # --- List processing ---
    {
        "description": "Sort a list of integers in ascending order.",
        "function_name": "sort_list",
        "input_signature": "def sort_list(xs: list[int]) -> list[int]",
        "unit_tests": (
            "def test_sort_basic():\n"
            "    assert sort_list([3, 1, 2]) == [1, 2, 3]\n\n"
            "def test_sort_empty():\n"
            "    assert sort_list([]) == []\n\n"
            "def test_sort_single():\n"
            "    assert sort_list([5]) == [5]\n\n"
            "def test_sort_duplicates():\n"
            "    assert sort_list([3, 1, 2, 1]) == [1, 1, 2, 3]\n"
        ),
    },
    {
        "description": "Remove duplicate elements from a list while preserving order.",
        "function_name": "deduplicate",
        "input_signature": "def deduplicate(xs: list[int]) -> list[int]",
        "unit_tests": (
            "def test_dedup_basic():\n"
            "    assert deduplicate([1, 2, 2, 3, 1]) == [1, 2, 3]\n\n"
            "def test_dedup_empty():\n"
            "    assert deduplicate([]) == []\n\n"
            "def test_dedup_no_dupes():\n"
            "    assert deduplicate([1, 2, 3]) == [1, 2, 3]\n"
        ),
    },
    {
        "description": "Flatten a nested list of integers into a single flat list.",
        "function_name": "flatten",
        "input_signature": "def flatten(nested: list) -> list[int]",
        "unit_tests": (
            "def test_flatten_basic():\n"
            "    assert flatten([1, [2, 3], [4, [5]]]) == [1, 2, 3, 4, 5]\n\n"
            "def test_flatten_empty():\n"
            "    assert flatten([]) == []\n\n"
            "def test_flatten_flat():\n"
            "    assert flatten([1, 2, 3]) == [1, 2, 3]\n"
        ),
    },
    {
        "description": "Partition a list into two lists: elements less than pivot and elements greater or equal.",
        "function_name": "partition",
        "input_signature": "def partition(xs: list[int], pivot: int) -> tuple[list[int], list[int]]",
        "unit_tests": (
            "def test_partition_basic():\n"
            "    low, high = partition([3, 1, 4, 1, 5], 3)\n"
            "    assert sorted(low) == [1, 1]\n"
            "    assert sorted(high) == [3, 4, 5]\n\n"
            "def test_partition_empty():\n"
            "    assert partition([], 5) == ([], [])\n"
        ),
    },

    # --- String manipulation ---
    {
        "description": "Reverse a string.",
        "function_name": "reverse_string",
        "input_signature": "def reverse_string(s: str) -> str",
        "unit_tests": (
            "def test_reverse_basic():\n"
            "    assert reverse_string('hello') == 'olleh'\n\n"
            "def test_reverse_empty():\n"
            "    assert reverse_string('') == ''\n\n"
            "def test_reverse_single():\n"
            "    assert reverse_string('a') == 'a'\n"
        ),
    },
    {
        "description": "Check if a string is a palindrome (case-insensitive, ignoring non-alphanumeric).",
        "function_name": "is_palindrome",
        "input_signature": "def is_palindrome(s: str) -> bool",
        "unit_tests": (
            "def test_palindrome_true():\n"
            "    assert is_palindrome('racecar') == True\n\n"
            "def test_palindrome_case():\n"
            "    assert is_palindrome('RaceCar') == True\n\n"
            "def test_palindrome_false():\n"
            "    assert is_palindrome('hello') == False\n\n"
            "def test_palindrome_spaces():\n"
            "    assert is_palindrome('A man a plan a canal Panama') == True\n"
        ),
    },
    {
        "description": "Count the frequency of each word in a string (case-insensitive).",
        "function_name": "word_count",
        "input_signature": "def word_count(text: str) -> dict[str, int]",
        "unit_tests": (
            "def test_word_count_basic():\n"
            "    result = word_count('the cat and the dog')\n"
            "    assert result['the'] == 2\n"
            "    assert result['cat'] == 1\n\n"
            "def test_word_count_empty():\n"
            "    assert word_count('') == {}\n"
        ),
    },

    # --- Math ---
    {
        "description": (
            "Compute the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1). "
            "Return 0 for negative n. "
            "You MUST use an iterative loop — plain recursion will time out on large n."
        ),
        "function_name": "fibonacci",
        "input_signature": "def fibonacci(n: int) -> int",
        # Provide an explicit iterative template so small models don't fall back
        # to the naive O(2^n) recursive pattern that always times out on fib(30).
        "reference_solution": (
            "def fibonacci(n: int) -> int:\n"
            "    if n <= 0:\n"
            "        return 0\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n - 1):\n"
            "        a, b = b, a + b\n"
            "    return a"
        ),
        "unit_tests": (
            "def test_fib_zero():\n"
            "    assert fibonacci(0) == 0\n\n"
            "def test_fib_one():\n"
            "    assert fibonacci(1) == 1\n\n"
            "def test_fib_ten():\n"
            "    assert fibonacci(10) == 55\n\n"
            "def test_fib_small():\n"
            "    assert fibonacci(6) == 8\n\n"
            "def test_fib_thirty():\n"
            "    assert fibonacci(30) == 832040\n\n"
            "def test_fib_negative():\n"
            "    assert fibonacci(-1) == 0\n"
        ),
    },
    {
        "description": "Check if a positive integer is prime.",
        "function_name": "is_prime",
        "input_signature": "def is_prime(n: int) -> bool",
        "unit_tests": (
            "def test_prime_true():\n"
            "    assert is_prime(7) == True\n\n"
            "def test_prime_false():\n"
            "    assert is_prime(4) == False\n\n"
            "def test_prime_one():\n"
            "    assert is_prime(1) == False\n\n"
            "def test_prime_two():\n"
            "    assert is_prime(2) == True\n\n"
            "def test_prime_large():\n"
            "    assert is_prime(97) == True\n"
        ),
    },
    {
        "description": "Compute the greatest common divisor (GCD) of two positive integers.",
        "function_name": "gcd",
        "input_signature": "def gcd(a: int, b: int) -> int",
        "unit_tests": (
            "def test_gcd_basic():\n"
            "    assert gcd(12, 8) == 4\n\n"
            "def test_gcd_coprime():\n"
            "    assert gcd(7, 13) == 1\n\n"
            "def test_gcd_same():\n"
            "    assert gcd(5, 5) == 5\n"
        ),
    },

    # --- Data transformation ---
    {
        "description": (
            "Invert a dictionary, swapping keys and values. "
            "When all original values are unique, map each value directly to its key (value -> key). "
            "When multiple keys share the same value, map that value to a sorted list of those keys (value -> [key1, key2]). "
            "Example: {'a': 1, 'b': 2} -> {1: 'a', 2: 'b'}. "
            "Example: {'a': 1, 'b': 1} -> {1: ['a', 'b']}."
        ),
        "function_name": "invert_dict",
        "input_signature": "def invert_dict(d: dict) -> dict",
        "unit_tests": (
            "def test_invert_basic():\n"
            "    assert invert_dict({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}\n\n"
            "def test_invert_empty():\n"
            "    assert invert_dict({}) == {}\n\n"
            "def test_invert_duplicate_values():\n"
            "    result = invert_dict({'a': 1, 'b': 1})\n"
            "    assert result == {1: ['a', 'b']} or result == {1: ['b', 'a']}\n\n"
            "def test_invert_single():\n"
            "    assert invert_dict({'x': 99}) == {99: 'x'}\n"
        ),
    },
    {
        "description": "Compute the running sum of a list of integers.",
        "function_name": "running_sum",
        "input_signature": "def running_sum(xs: list[int]) -> list[int]",
        "unit_tests": (
            "def test_running_sum_basic():\n"
            "    assert running_sum([1, 2, 3, 4]) == [1, 3, 6, 10]\n\n"
            "def test_running_sum_empty():\n"
            "    assert running_sum([]) == []\n\n"
            "def test_running_sum_single():\n"
            "    assert running_sum([5]) == [5]\n"
        ),
    },

    # -----------------------------------------------------------------------
    # Experiment B: Harder tasks (HumanEval / MBPP style)
    # These tasks require non-trivial reasoning and are more discriminative.
    # -----------------------------------------------------------------------

    # --- Algorithmic / search ---
    {
        "description": (
            "Given a sorted list of integers xs and a target integer, return the index where target "
            "would be inserted to keep xs sorted (binary search / bisect_left semantics). "
            "If target is already in xs, return the index of its leftmost occurrence. "
            "Example: search_insert([1,3,5,6], 5) -> 2. search_insert([1,3,5,6], 2) -> 1."
        ),
        "function_name": "search_insert",
        "input_signature": "def search_insert(xs: list[int], target: int) -> int",
        "unit_tests": (
            "def test_si_found():\n"
            "    assert search_insert([1,3,5,6], 5) == 2\n\n"
            "def test_si_not_found():\n"
            "    assert search_insert([1,3,5,6], 2) == 1\n\n"
            "def test_si_before_all():\n"
            "    assert search_insert([1,3,5,6], 0) == 0\n\n"
            "def test_si_after_all():\n"
            "    assert search_insert([1,3,5,6], 7) == 4\n\n"
            "def test_si_empty():\n"
            "    assert search_insert([], 5) == 0\n\n"
            "def test_si_leftmost():\n"
            "    assert search_insert([1,2,2,2,3], 2) == 1\n"
        ),
    },
    {
        "description": (
            "Find the length of the longest strictly increasing subsequence (LIS) in a list of integers. "
            "A subsequence does not need to be contiguous. "
            "Example: lis([10,9,2,5,3,7,101,18]) -> 4 (subsequence [2,3,7,101])."
        ),
        "function_name": "lis",
        "input_signature": "def lis(xs: list[int]) -> int",
        "unit_tests": (
            "def test_lis_example():\n"
            "    assert lis([10,9,2,5,3,7,101,18]) == 4\n\n"
            "def test_lis_all_same():\n"
            "    assert lis([2,2,2,2]) == 1\n\n"
            "def test_lis_sorted():\n"
            "    assert lis([1,2,3,4,5]) == 5\n\n"
            "def test_lis_empty():\n"
            "    assert lis([]) == 0\n\n"
            "def test_lis_single():\n"
            "    assert lis([7]) == 1\n\n"
            "def test_lis_decreasing():\n"
            "    assert lis([5,4,3,2,1]) == 1\n"
        ),
    },
    {
        "description": (
            "Given an integer n (n >= 0), compute the number of ways to climb n stairs "
            "if you can take 1 or 2 steps at a time. "
            "Example: climb_stairs(3) -> 3 (1+1+1, 1+2, 2+1). "
            "climb_stairs(0) -> 1 (empty climb). climb_stairs(1) -> 1."
        ),
        "function_name": "climb_stairs",
        "input_signature": "def climb_stairs(n: int) -> int",
        "reference_solution": (
            "def climb_stairs(n: int) -> int:\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    a, b = 1, 1\n"
            "    for _ in range(n - 1):\n"
            "        a, b = b, a + b\n"
            "    return b"
        ),
        "unit_tests": (
            "def test_cs_zero():\n"
            "    assert climb_stairs(0) == 1\n\n"
            "def test_cs_one():\n"
            "    assert climb_stairs(1) == 1\n\n"
            "def test_cs_two():\n"
            "    assert climb_stairs(2) == 2\n\n"
            "def test_cs_three():\n"
            "    assert climb_stairs(3) == 3\n\n"
            "def test_cs_five():\n"
            "    assert climb_stairs(5) == 8\n\n"
            "def test_cs_ten():\n"
            "    assert climb_stairs(10) == 89\n"
        ),
    },

    # --- String algorithms ---
    {
        "description": (
            "Given a string, find the length of the longest substring without repeating characters. "
            "Example: longest_unique('abcabcbb') -> 3 ('abc'). "
            "longest_unique('bbbbb') -> 1. longest_unique('') -> 0."
        ),
        "function_name": "longest_unique",
        "input_signature": "def longest_unique(s: str) -> int",
        "unit_tests": (
            "def test_lu_basic():\n"
            "    assert longest_unique('abcabcbb') == 3\n\n"
            "def test_lu_all_same():\n"
            "    assert longest_unique('bbbbb') == 1\n\n"
            "def test_lu_empty():\n"
            "    assert longest_unique('') == 0\n\n"
            "def test_lu_all_unique():\n"
            "    assert longest_unique('abcde') == 5\n\n"
            "def test_lu_pwwkew():\n"
            "    assert longest_unique('pwwkew') == 3\n"
        ),
    },
    {
        "description": (
            "Given two strings s and t, return True if t is an anagram of s (uses exactly the same "
            "characters with the same frequencies), False otherwise. Case-sensitive. "
            "Example: is_anagram('anagram', 'nagaram') -> True. "
            "is_anagram('rat', 'car') -> False."
        ),
        "function_name": "is_anagram",
        "input_signature": "def is_anagram(s: str, t: str) -> bool",
        "unit_tests": (
            "def test_anagram_true():\n"
            "    assert is_anagram('anagram', 'nagaram') == True\n\n"
            "def test_anagram_false():\n"
            "    assert is_anagram('rat', 'car') == False\n\n"
            "def test_anagram_empty():\n"
            "    assert is_anagram('', '') == True\n\n"
            "def test_anagram_length_diff():\n"
            "    assert is_anagram('ab', 'abc') == False\n\n"
            "def test_anagram_case():\n"
            "    assert is_anagram('Ab', 'aB') == False\n"
        ),
    },
    {
        "description": (
            "Given a string containing just brackets '(', ')', '{', '}', '[', ']', "
            "return True if the bracket sequence is valid (properly opened and closed in order), "
            "False otherwise. "
            "Example: valid_brackets('()[]{}') -> True. valid_brackets('([)]') -> False."
        ),
        "function_name": "valid_brackets",
        "input_signature": "def valid_brackets(s: str) -> bool",
        "unit_tests": (
            "def test_vb_valid():\n"
            "    assert valid_brackets('()[]{}') == True\n\n"
            "def test_vb_nested():\n"
            "    assert valid_brackets('{[()]}') == True\n\n"
            "def test_vb_invalid_order():\n"
            "    assert valid_brackets('([)]') == False\n\n"
            "def test_vb_empty():\n"
            "    assert valid_brackets('') == True\n\n"
            "def test_vb_unclosed():\n"
            "    assert valid_brackets('(') == False\n\n"
            "def test_vb_extra_close():\n"
            "    assert valid_brackets(']') == False\n"
        ),
    },

    # --- Number theory / combinatorics ---
    {
        "description": (
            "Return True if a positive integer n is a perfect square, False otherwise. "
            "Do NOT use floating-point sqrt — use integer arithmetic only. "
            "Example: is_perfect_square(16) -> True. is_perfect_square(14) -> False."
        ),
        "function_name": "is_perfect_square",
        "input_signature": "def is_perfect_square(n: int) -> bool",
        "unit_tests": (
            "def test_ps_true():\n"
            "    assert is_perfect_square(16) == True\n\n"
            "def test_ps_false():\n"
            "    assert is_perfect_square(14) == False\n\n"
            "def test_ps_one():\n"
            "    assert is_perfect_square(1) == True\n\n"
            "def test_ps_large():\n"
            "    assert is_perfect_square(1000000) == True\n\n"
            "def test_ps_large_not():\n"
            "    assert is_perfect_square(999999) == False\n\n"
            "def test_ps_zero():\n"
            "    assert is_perfect_square(0) == True\n"
        ),
    },
    {
        "description": (
            "Given two non-negative integers m and n, compute C(m+n, m) — the number of "
            "unique paths in an m x n grid from the top-left to the bottom-right corner "
            "moving only right or down. "
            "Example: unique_paths(3, 7) -> 28. unique_paths(1, 1) -> 1."
        ),
        "function_name": "unique_paths",
        "input_signature": "def unique_paths(m: int, n: int) -> int",
        "reference_solution": (
            "def unique_paths(m: int, n: int) -> int:\n"
            "    # Guard: degenerate grid has no destination\n"
            "    if m <= 0 or n <= 0:\n"
            "        return 0\n"
            "    # C(m+n-2, m-1) unique monotone lattice paths\n"
            "    import math\n"
            "    return math.comb(m + n - 2, m - 1)\n"
        ),
        "unit_tests": (
            "def test_up_example():\n"
            "    assert unique_paths(3, 7) == 28\n\n"
            "def test_up_one_one():\n"
            "    assert unique_paths(1, 1) == 1\n\n"
            "def test_up_one_row():\n"
            "    assert unique_paths(1, 10) == 1\n\n"
            "def test_up_two_two():\n"
            "    assert unique_paths(2, 2) == 2\n\n"
            "def test_up_three_three():\n"
            "    assert unique_paths(3, 3) == 6\n"
        ),
    },

    # --- Graph / tree structure (lists-as-adjacency) ---
    {
        "description": (
            "Given an integer n (number of nodes, 0-indexed) and a list of directed edges "
            "[src, dst], detect if the graph has a cycle using DFS. Return True if cyclic. "
            "Example: has_cycle(4, [[0,1],[1,2],[2,0],[2,3]]) -> True (0->1->2->0). "
            "has_cycle(4, [[0,1],[1,2],[2,3]]) -> False."
        ),
        "function_name": "has_cycle",
        "input_signature": "def has_cycle(n: int, edges: list[list[int]]) -> bool",
        "reference_solution": (
            "def has_cycle(n: int, edges: list[list[int]]) -> bool:\n"
            "    # Kahn's algorithm: topological sort via in-degree.\n"
            "    # If all n nodes are processed the DAG has no cycle.\n"
            "    from collections import deque\n"
            "    in_degree = [0] * n\n"
            "    adj = [[] for _ in range(n)]\n"
            "    for src, dst in edges:\n"
            "        adj[src].append(dst)\n"
            "        in_degree[dst] += 1\n"
            "    queue = deque(i for i in range(n) if in_degree[i] == 0)\n"
            "    visited = 0\n"
            "    while queue:\n"
            "        node = queue.popleft()\n"
            "        visited += 1\n"
            "        for nb in adj[node]:\n"
            "            in_degree[nb] -= 1\n"
            "            if in_degree[nb] == 0:\n"
            "                queue.append(nb)\n"
            "    return visited != n\n"
        ),
        "unit_tests": (
            "def test_hc_cyclic():\n"
            "    assert has_cycle(4, [[0,1],[1,2],[2,0],[2,3]]) == True\n\n"
            "def test_hc_acyclic():\n"
            "    assert has_cycle(4, [[0,1],[1,2],[2,3]]) == False\n\n"
            "def test_hc_self_loop():\n"
            "    assert has_cycle(2, [[0,0]]) == True\n\n"
            "def test_hc_empty():\n"
            "    assert has_cycle(3, []) == False\n\n"
            "def test_hc_disconnected():\n"
            "    assert has_cycle(4, [[0,1],[2,3]]) == False\n"
        ),
    },
    {
        "description": (
            "Given a list of integers, return all unique permutations as a sorted list of lists. "
            "Example: permutations([1,1,2]) -> [[1,1,2],[1,2,1],[2,1,1]]. "
            "The output list must be lexicographically sorted."
        ),
        "function_name": "permutations",
        "input_signature": "def permutations(xs: list[int]) -> list[list[int]]",
        "reference_solution": (
            "def permutations(xs: list[int]) -> list[list[int]]:\n"
            "    import itertools\n"
            "    # itertools.permutations returns tuples; convert each to list\n"
            "    return [list(p) for p in sorted(set(itertools.permutations(xs)))]\n"
        ),
        "unit_tests": (
            "def test_perm_with_dupe():\n"
            "    assert permutations([1,1,2]) == [[1,1,2],[1,2,1],[2,1,1]]\n\n"
            "def test_perm_single():\n"
            "    assert permutations([1]) == [[1]]\n\n"
            "def test_perm_empty():\n"
            "    assert permutations([]) == [[]]\n\n"
            "def test_perm_three_unique():\n"
            "    result = permutations([1,2,3])\n"
            "    assert len(result) == 6\n"
            "    assert result == sorted(result)\n"
        ),
    },

    # -----------------------------------------------------------------------
    # Phase 1B: Additional tasks with explicit category tags
    # -----------------------------------------------------------------------

    # --- iterative (+2) ---
    {
        "description": "Compute the factorial of a non-negative integer n (n! = n*(n-1)*...*1). factorial(0) = 1.",
        "function_name": "factorial",
        "category": "iterative",
        "input_signature": "def factorial(n: int) -> int",
        "unit_tests": (
            "def test_fact_zero():\n"
            "    assert factorial(0) == 1\n\n"
            "def test_fact_one():\n"
            "    assert factorial(1) == 1\n\n"
            "def test_fact_five():\n"
            "    assert factorial(5) == 120\n\n"
            "def test_fact_ten():\n"
            "    assert factorial(10) == 3628800\n"
        ),
    },
    {
        "description": "Compute the sum of digits of a non-negative integer. digit_sum(123) -> 6. digit_sum(0) -> 0.",
        "function_name": "digit_sum",
        "category": "iterative",
        "input_signature": "def digit_sum(n: int) -> int",
        "unit_tests": (
            "def test_ds_zero():\n"
            "    assert digit_sum(0) == 0\n\n"
            "def test_ds_basic():\n"
            "    assert digit_sum(123) == 6\n\n"
            "def test_ds_single():\n"
            "    assert digit_sum(9) == 9\n\n"
            "def test_ds_large():\n"
            "    assert digit_sum(999) == 27\n"
        ),
    },

    # --- recursive (+3) ---
    {
        "description": (
            "Return the power set of a list of distinct integers as a list of lists (order does not matter). "
            "power_set([]) -> [[]]. power_set([1,2]) must contain [], [1], [2], [1,2]."
        ),
        "function_name": "power_set",
        "category": "recursive",
        "input_signature": "def power_set(xs: list[int]) -> list[list[int]]",
        "unit_tests": (
            "def test_pset_empty():\n"
            "    assert power_set([]) == [[]]\n\n"
            "def test_pset_single():\n"
            "    result = power_set([1])\n"
            "    assert sorted(result) == [[], [1]]\n\n"
            "def test_pset_two():\n"
            "    result = power_set([1, 2])\n"
            "    assert len(result) == 4\n"
            "    assert [] in result\n"
            "    assert [1, 2] in result or [2, 1] in result\n\n"
            "def test_pset_count():\n"
            "    assert len(power_set([1, 2, 3])) == 8\n"
        ),
    },
    {
        "description": (
            "Given n pairs of parentheses, generate all valid combinations of well-formed brackets. "
            "Return a sorted list of strings. "
            "generate_parens(1) -> ['()']. generate_parens(2) -> ['(())', '()()']."
        ),
        "function_name": "generate_parens",
        "category": "recursive",
        "input_signature": "def generate_parens(n: int) -> list[str]",
        "unit_tests": (
            "def test_gp_one():\n"
            "    assert generate_parens(1) == ['()']\n\n"
            "def test_gp_two():\n"
            "    assert generate_parens(2) == ['(())', '()()']\n\n"
            "def test_gp_three_count():\n"
            "    assert len(generate_parens(3)) == 5\n\n"
            "def test_gp_sorted():\n"
            "    result = generate_parens(3)\n"
            "    assert result == sorted(result)\n"
        ),
    },
    {
        "description": (
            "Compute the sum of all integers in a (possibly nested) list. "
            "nested_sum([1, [2, 3], [4, [5]]]) -> 15. nested_sum([]) -> 0."
        ),
        "function_name": "nested_sum",
        "category": "recursive",
        "input_signature": "def nested_sum(xs: list) -> int",
        "unit_tests": (
            "def test_nsum_basic():\n"
            "    assert nested_sum([1, [2, 3], [4, [5]]]) == 15\n\n"
            "def test_nsum_empty():\n"
            "    assert nested_sum([]) == 0\n\n"
            "def test_nsum_flat():\n"
            "    assert nested_sum([1, 2, 3]) == 6\n\n"
            "def test_nsum_deep():\n"
            "    assert nested_sum([[[[1]]]]) == 1\n"
        ),
    },

    # --- dynamic_programming (+2) ---
    {
        "description": (
            "Find the maximum sum of a contiguous subarray (Kadane's algorithm). "
            "max_subarray([-2,1,-3,4,-1,2,1,-5,4]) -> 6. "
            "For an array of all negatives, return the maximum (least-negative) element."
        ),
        "function_name": "max_subarray",
        "category": "dynamic_programming",
        "input_signature": "def max_subarray(nums: list[int]) -> int",
        "unit_tests": (
            "def test_msa_example():\n"
            "    assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6\n\n"
            "def test_msa_single():\n"
            "    assert max_subarray([1]) == 1\n\n"
            "def test_msa_all_neg():\n"
            "    assert max_subarray([-3,-1,-2]) == -1\n\n"
            "def test_msa_all_pos():\n"
            "    assert max_subarray([1,2,3]) == 6\n"
        ),
    },
    {
        "description": (
            "Return the fewest coins needed to make up the given amount using coins of the given "
            "denominations (each coin may be used unlimited times). Return -1 if impossible. "
            "coin_change([1,5,11], 15) -> 3 (5+5+5). coin_change([2], 3) -> -1."
        ),
        "function_name": "coin_change",
        "category": "dynamic_programming",
        "input_signature": "def coin_change(coins: list[int], amount: int) -> int",
        "unit_tests": (
            "def test_cc_example():\n"
            "    assert coin_change([1, 5, 11], 15) == 3\n\n"
            "def test_cc_impossible():\n"
            "    assert coin_change([2], 3) == -1\n\n"
            "def test_cc_zero():\n"
            "    assert coin_change([1, 2, 5], 0) == 0\n\n"
            "def test_cc_greedy_fails():\n"
            "    assert coin_change([1, 3, 4], 6) == 2\n"
        ),
    },

    # --- graph (+4) ---
    {
        "description": (
            "Given n nodes (0-indexed) and directed edges [[src, dst], ...], return a valid topological "
            "ordering as a list of node indices. If the graph has a cycle, return an empty list. "
            "Kahn's algorithm (in-degree / BFS) is recommended."
        ),
        "function_name": "topological_sort",
        "category": "graph",
        "input_signature": "def topological_sort(n: int, edges: list[list[int]]) -> list[int]",
        "reference_solution": (
            "def topological_sort(n: int, edges: list[list[int]]) -> list[int]:\n"
            "    from collections import deque\n"
            "    in_degree = [0] * n\n"
            "    adj = [[] for _ in range(n)]\n"
            "    for src, dst in edges:\n"
            "        adj[src].append(dst)\n"
            "        in_degree[dst] += 1\n"
            "    queue = deque(i for i in range(n) if in_degree[i] == 0)\n"
            "    result = []\n"
            "    while queue:\n"
            "        node = queue.popleft()\n"
            "        result.append(node)\n"
            "        for nb in adj[node]:\n"
            "            in_degree[nb] -= 1\n"
            "            if in_degree[nb] == 0:\n"
            "                queue.append(nb)\n"
            "    return result if len(result) == n else []\n"
        ),
        "unit_tests": (
            "def test_ts_linear():\n"
            "    assert topological_sort(4, [[0,1],[1,2],[2,3]]) == [0,1,2,3]\n\n"
            "def test_ts_cyclic():\n"
            "    assert topological_sort(3, [[0,1],[1,2],[2,0]]) == []\n\n"
            "def test_ts_no_edges():\n"
            "    result = topological_sort(3, [])\n"
            "    assert sorted(result) == [0, 1, 2]\n\n"
            "def test_ts_valid_order():\n"
            "    result = topological_sort(4, [[0,2],[1,2],[2,3]])\n"
            "    assert len(result) == 4\n"
            "    idx = {v: i for i, v in enumerate(result)}\n"
            "    assert idx[0] < idx[2] and idx[1] < idx[2] and idx[2] < idx[3]\n"
        ),
    },
    {
        "description": (
            "Count the number of connected components in an undirected graph with n nodes (0-indexed) "
            "and edges [[u, v], ...] (each edge is bidirectional). "
            "count_components(5, [[0,1],[1,2],[3,4]]) -> 2. count_components(4, []) -> 4."
        ),
        "function_name": "count_components",
        "category": "graph",
        "input_signature": "def count_components(n: int, edges: list[list[int]]) -> int",
        "unit_tests": (
            "def test_ccomp_two():\n"
            "    assert count_components(5, [[0,1],[1,2],[3,4]]) == 2\n\n"
            "def test_ccomp_isolated():\n"
            "    assert count_components(4, []) == 4\n\n"
            "def test_ccomp_all_one():\n"
            "    assert count_components(4, [[0,1],[1,2],[2,3]]) == 1\n\n"
            "def test_ccomp_single():\n"
            "    assert count_components(1, []) == 1\n"
        ),
    },
    {
        "description": (
            "Count the number of islands in a 2D grid of '1' (land) and '0' (water). "
            "An island is a group of '1's connected horizontally or vertically. "
            "num_islands([['1','1','0'],['0','1','0'],['0','0','1']]) -> 2."
        ),
        "function_name": "num_islands",
        "category": "graph",
        "input_signature": "def num_islands(grid: list[list[str]]) -> int",
        "unit_tests": (
            "def test_ni_basic():\n"
            "    g = [['1','1','0'],['0','1','0'],['0','0','1']]\n"
            "    assert num_islands(g) == 2\n\n"
            "def test_ni_empty():\n"
            "    assert num_islands([]) == 0\n\n"
            "def test_ni_all_water():\n"
            "    assert num_islands([['0','0'],['0','0']]) == 0\n\n"
            "def test_ni_one_island():\n"
            "    assert num_islands([['1','1'],['1','1']]) == 1\n"
        ),
    },
    {
        "description": (
            "Determine if an undirected graph with n nodes (0-indexed) and edges [[u,v],...] is bipartite "
            "(2-colorable: no two adjacent nodes share the same color). "
            "is_bipartite(4, [[0,1],[1,2],[2,3],[3,0]]) -> True (even cycle). "
            "is_bipartite(3, [[0,1],[1,2],[2,0]]) -> False (odd triangle)."
        ),
        "function_name": "is_bipartite",
        "category": "graph",
        "input_signature": "def is_bipartite(n: int, edges: list[list[int]]) -> bool",
        "unit_tests": (
            "def test_ibip_even_cycle():\n"
            "    assert is_bipartite(4, [[0,1],[1,2],[2,3],[3,0]]) == True\n\n"
            "def test_ibip_triangle():\n"
            "    assert is_bipartite(3, [[0,1],[1,2],[2,0]]) == False\n\n"
            "def test_ibip_empty():\n"
            "    assert is_bipartite(3, []) == True\n\n"
            "def test_ibip_linear():\n"
            "    assert is_bipartite(4, [[0,1],[1,2],[2,3]]) == True\n"
        ),
    },

    # --- string (+1) ---
    {
        "description": (
            "Run-length encode a string: replace each run of consecutive identical characters with "
            "the character followed by the run count. Omit the count when it is 1. "
            "compress_string('aabccc') -> 'a2bc3'. compress_string('abc') -> 'abc'. compress_string('') -> ''."
        ),
        "function_name": "compress_string",
        "category": "string",
        "input_signature": "def compress_string(s: str) -> str",
        "unit_tests": (
            "def test_cstr_basic():\n"
            "    assert compress_string('aabccc') == 'a2bc3'\n\n"
            "def test_cstr_no_repeat():\n"
            "    assert compress_string('abc') == 'abc'\n\n"
            "def test_cstr_empty():\n"
            "    assert compress_string('') == ''\n\n"
            "def test_cstr_all_same():\n"
            "    assert compress_string('aaaa') == 'a4'\n"
        ),
    },

    # --- data_structures (+1) ---
    {
        "description": (
            "Given a list of intervals [[start, end], ...], merge all overlapping intervals "
            "and return a sorted list of merged intervals. "
            "merge_intervals([[1,3],[2,6],[8,10],[15,18]]) -> [[1,6],[8,10],[15,18]]."
        ),
        "function_name": "merge_intervals",
        "category": "data_structures",
        "input_signature": "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]",
        "unit_tests": (
            "def test_mi_basic():\n"
            "    assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n\n"
            "def test_mi_empty():\n"
            "    assert merge_intervals([]) == []\n\n"
            "def test_mi_no_overlap():\n"
            "    assert merge_intervals([[1,2],[3,4]]) == [[1,2],[3,4]]\n\n"
            "def test_mi_all_overlap():\n"
            "    assert merge_intervals([[1,4],[2,5],[3,6]]) == [[1,6]]\n"
        ),
    },

    # --- parsing (+5) ---
    {
        "description": (
            "Convert a Roman numeral string to an integer. "
            "Symbols: I=1, V=5, X=10, L=50, C=100, D=500, M=1000. "
            "Subtraction rules: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900. "
            "roman_to_int('III') -> 3. roman_to_int('MCMXCIV') -> 1994."
        ),
        "function_name": "roman_to_int",
        "category": "parsing",
        "input_signature": "def roman_to_int(s: str) -> int",
        "unit_tests": (
            "def test_rti_simple():\n"
            "    assert roman_to_int('III') == 3\n\n"
            "def test_rti_subtraction():\n"
            "    assert roman_to_int('IX') == 9\n\n"
            "def test_rti_complex():\n"
            "    assert roman_to_int('MCMXCIV') == 1994\n\n"
            "def test_rti_lviii():\n"
            "    assert roman_to_int('LVIII') == 58\n\n"
            "def test_rti_iv():\n"
            "    assert roman_to_int('IV') == 4\n"
        ),
    },
    {
        "description": (
            "Convert a string to title case: capitalize the first letter of each word, "
            "lowercase the rest. Words are separated by spaces. "
            "title_case('the quick brown fox') -> 'The Quick Brown Fox'. title_case('') -> ''."
        ),
        "function_name": "title_case",
        "category": "parsing",
        "input_signature": "def title_case(s: str) -> str",
        "unit_tests": (
            "def test_tc_basic():\n"
            "    assert title_case('the quick brown fox') == 'The Quick Brown Fox'\n\n"
            "def test_tc_empty():\n"
            "    assert title_case('') == ''\n\n"
            "def test_tc_single():\n"
            "    assert title_case('hello') == 'Hello'\n\n"
            "def test_tc_mixed():\n"
            "    assert title_case('hELLO wORLD') == 'Hello World'\n"
        ),
    },
    {
        "description": (
            "Convert a camelCase or PascalCase string to snake_case by inserting an underscore "
            "before each uppercase letter (except the first), then lowercasing everything. "
            "camel_to_snake('camelCase') -> 'camel_case'. camel_to_snake('myVariableName') -> 'my_variable_name'."
        ),
        "function_name": "camel_to_snake",
        "category": "parsing",
        "input_signature": "def camel_to_snake(s: str) -> str",
        "unit_tests": (
            "def test_cts_basic():\n"
            "    assert camel_to_snake('camelCase') == 'camel_case'\n\n"
            "def test_cts_multi():\n"
            "    assert camel_to_snake('myVariableName') == 'my_variable_name'\n\n"
            "def test_cts_empty():\n"
            "    assert camel_to_snake('') == ''\n\n"
            "def test_cts_already_lower():\n"
            "    assert camel_to_snake('lowercase') == 'lowercase'\n"
        ),
    },
    {
        "description": (
            "Count the number of vowels (a, e, i, o, u — case-insensitive) in a string. "
            "count_vowels('Hello World') -> 3. count_vowels('bcd') -> 0."
        ),
        "function_name": "count_vowels",
        "category": "parsing",
        "input_signature": "def count_vowels(s: str) -> int",
        "unit_tests": (
            "def test_cvow_basic():\n"
            "    assert count_vowels('Hello World') == 3\n\n"
            "def test_cvow_empty():\n"
            "    assert count_vowels('') == 0\n\n"
            "def test_cvow_all():\n"
            "    assert count_vowels('aeiou') == 5\n\n"
            "def test_cvow_none():\n"
            "    assert count_vowels('bcd') == 0\n\n"
            "def test_cvow_upper():\n"
            "    assert count_vowels('AEIOU') == 5\n"
        ),
    },
    {
        "description": (
            "Decode a run-length encoded string. Each character is optionally followed by a single "
            "digit (1-9) giving its repeat count; if no digit follows, the character appears once. "
            "decode_run_length('a2bc3') -> 'aabccc'. decode_run_length('abc') -> 'abc'."
        ),
        "function_name": "decode_run_length",
        "category": "parsing",
        "input_signature": "def decode_run_length(s: str) -> str",
        "unit_tests": (
            "def test_drl_basic():\n"
            "    assert decode_run_length('a2bc3') == 'aabccc'\n\n"
            "def test_drl_no_repeat():\n"
            "    assert decode_run_length('abc') == 'abc'\n\n"
            "def test_drl_empty():\n"
            "    assert decode_run_length('') == ''\n\n"
            "def test_drl_counts():\n"
            "    assert decode_run_length('a3b2') == 'aaabb'\n"
        ),
    },

    # --- math (+2) ---
    {
        "description": "Compute the least common multiple (LCM) of two positive integers. lcm(4, 6) -> 12.",
        "function_name": "lcm",
        "category": "math",
        "input_signature": "def lcm(a: int, b: int) -> int",
        "unit_tests": (
            "def test_lcm_basic():\n"
            "    assert lcm(4, 6) == 12\n\n"
            "def test_lcm_coprime():\n"
            "    assert lcm(3, 5) == 15\n\n"
            "def test_lcm_same():\n"
            "    assert lcm(7, 7) == 7\n\n"
            "def test_lcm_one():\n"
            "    assert lcm(1, 8) == 8\n"
        ),
    },
    {
        "description": (
            "Given a non-empty list of integers where every element appears exactly twice except for one, "
            "find that single non-duplicate element. "
            "single_number([2,2,1]) -> 1. single_number([4,1,2,1,2]) -> 4."
        ),
        "function_name": "single_number",
        "category": "math",
        "input_signature": "def single_number(nums: list[int]) -> int",
        "unit_tests": (
            "def test_snum_basic():\n"
            "    assert single_number([2, 2, 1]) == 1\n\n"
            "def test_snum_longer():\n"
            "    assert single_number([4, 1, 2, 1, 2]) == 4\n\n"
            "def test_snum_single():\n"
            "    assert single_number([7]) == 7\n\n"
            "def test_snum_negative():\n"
            "    assert single_number([-1, -1, 0]) == 0\n"
        ),
    },
]


# ---------------------------------------------------------------------------
# Category map — covers all tasks (existing + Phase 1B)
# ---------------------------------------------------------------------------

# Maps function_name -> category for tasks that don't carry a "category" field.
TASK_CATEGORY_MAP: dict[str, str] = {
    # Original tasks (no "category" key in spec)
    "sort_list":          "data_structures",
    "deduplicate":        "data_structures",
    "flatten":            "recursive",
    "partition":          "data_structures",
    "reverse_string":     "string",
    "is_palindrome":      "string",
    "word_count":         "string",
    "fibonacci":          "iterative",
    "is_prime":           "math",
    "gcd":                "math",
    "invert_dict":        "data_structures",
    "running_sum":        "iterative",
    "search_insert":      "iterative",
    "lis":                "dynamic_programming",
    "climb_stairs":       "dynamic_programming",
    "longest_unique":     "string",
    "is_anagram":         "string",
    "valid_brackets":     "data_structures",
    "is_perfect_square":  "math",
    "unique_paths":       "dynamic_programming",
    "has_cycle":          "graph",
    "permutations":       "recursive",
    # Phase 1B tasks (also carry "category" in spec — map included for completeness)
    "factorial":          "iterative",
    "digit_sum":          "iterative",
    "power_set":          "recursive",
    "generate_parens":    "recursive",
    "nested_sum":         "recursive",
    "max_subarray":       "dynamic_programming",
    "coin_change":        "dynamic_programming",
    "topological_sort":   "graph",
    "count_components":   "graph",
    "num_islands":        "graph",
    "is_bipartite":       "graph",
    "compress_string":    "string",
    "merge_intervals":    "data_structures",
    "roman_to_int":       "parsing",
    "title_case":         "parsing",
    "camel_to_snake":     "parsing",
    "count_vowels":       "parsing",
    "decode_run_length":  "parsing",
    "lcm":                "math",
    "single_number":      "math",
}


def _task_category(spec: dict) -> str:
    """Return the category for a task spec, falling back to TASK_CATEGORY_MAP."""
    return spec.get("category", TASK_CATEGORY_MAP.get(spec["function_name"], "unknown"))


def make_live_task(spec: dict) -> Task:
    """Create a Task from a spec dictionary."""
    return Task(
        description=spec["description"],
        function_name=spec["function_name"],
        input_signature=spec["input_signature"],
        unit_tests=spec["unit_tests"],
        reference_solution=spec.get("reference_solution"),
        timeout_seconds=MCNConfig.SANDBOX_TIMEOUT,
    )


def make_live_task_sequence(n: int) -> list[tuple[str, str, Task]]:
    """Generate a round-robin sequence of n tasks.

    Returns:
        List of (task_type, category, Task) 3-tuples.
    """
    seq = []
    for i in range(n):
        spec = LIVE_TASKS[i % len(LIVE_TASKS)]
        task = make_live_task(spec)
        task_type = spec["function_name"]
        category  = _task_category(spec)
        seq.append((task_type, category, task))
    return seq


def make_stratified_task_sequence(tasks_per_category: int) -> list[tuple[str, str, Task]]:
    """Generate a stratified task sequence with tasks_per_category per category.

    Tasks are sampled round-robin within each category, then the full sequence is
    shuffled so tasks from different categories are interleaved (prevents the bandit
    from only seeing one category at a time during learning).

    Args:
        tasks_per_category: Number of task attempts per category.

    Returns:
        List of (task_type, category, Task) 3-tuples, shuffled.
    """
    import random

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for spec in LIVE_TASKS:
        cat = _task_category(spec)
        by_cat[cat].append(spec)

    seq: list[tuple[str, str, Task]] = []
    for cat in sorted(by_cat):
        specs = by_cat[cat]
        for i in range(tasks_per_category):
            spec     = specs[i % len(specs)]
            task     = make_live_task(spec)
            seq.append((spec["function_name"], cat, task))

    random.shuffle(seq)
    return seq


# ---------------------------------------------------------------------------
# Pure-Python mock simulation (no Ray — works on Python 3.13+)
# ---------------------------------------------------------------------------

def _run_mock_simulation(
    task_seq: list[tuple[str, str, "Task"]],
    n_tribes: int = 3,
    log_dir: str = "",
) -> tuple[list[dict], dict, dict]:
    """Simulate MCN routing without Ray actors.

    Uses a UCB1 bandit per (category, tribe) pair and a fixed capability
    matrix to generate realistic pass/fail outcomes.  Returns the same
    record schema as the live Ray path so that ``print_report`` and
    ``analyze_routing.py`` work unchanged.

    Returns:
        (all_results, stats_dict, routing_totals_dict)
    """
    import math
    import random

    # Capability matrix — P[tribe][category] = base solve probability.
    # Mirrors the matrix in run_oracle_experiment.py.
    CAP: list[dict[str, float]] = [
        # T0: strong at iterative, string, data_structures, math
        {"iterative": 0.90, "recursive": 0.60, "dynamic_programming": 0.65,
         "graph": 0.35, "string": 0.95, "data_structures": 0.95,
         "parsing": 0.85, "math": 0.85},
        # T1: strong at parsing, string, data_structures
        {"iterative": 0.85, "recursive": 0.70, "dynamic_programming": 0.75,
         "graph": 0.45, "string": 0.95, "data_structures": 0.93,
         "parsing": 0.88, "math": 0.82},
        # T2: strong at dynamic_programming, recursive, graph
        {"iterative": 0.75, "recursive": 0.80, "dynamic_programming": 0.85,
         "graph": 0.50, "string": 0.90, "data_structures": 0.92,
         "parsing": 0.85, "math": 0.75},
    ]
    DEFAULT_CAP = 0.75

    CATEGORIES = [
        "iterative", "recursive", "dynamic_programming", "graph",
        "string", "data_structures", "parsing", "math",
    ]

    # UCB1 accumulators: counts[ci][tribe], wins[ci][tribe]
    n_cats = len(CATEGORIES)
    counts = [[0] * n_tribes for _ in range(n_cats)]
    wins   = [[0] * n_tribes for _ in range(n_cats)]

    def _ucb1_route(category: str, step: int) -> int:
        ci = CATEGORIES.index(category) if category in CATEGORIES else 0
        # Explore any arm not yet tried in this category first
        for t in range(n_tribes):
            if counts[ci][t] == 0:
                return t
        best_t, best_score = 0, -1.0
        for t in range(n_tribes):
            mu  = wins[ci][t] / counts[ci][t]
            ucb = mu + math.sqrt(2.0 * math.log(step + 1) / counts[ci][t])
            if ucb > best_score:
                best_score, best_t = ucb, t
        return best_t

    all_results: list[dict] = []
    routing_totals: dict[int, int] = {i: 0 for i in range(n_tribes)}
    n_total = len(task_seq)

    for i, (task_type, category, _task) in enumerate(task_seq):
        tribe_idx = _ucb1_route(category, i + 1)
        cap       = CAP[tribe_idx].get(category, DEFAULT_CAP)
        passed    = random.random() < cap

        # Update UCB accumulators
        ci = CATEGORIES.index(category) if category in CATEGORIES else 0
        counts[ci][tribe_idx] += 1
        if passed:
            wins[ci][tribe_idx] += 1

        routing_totals[tribe_idx] += 1
        reward = 1.0 if passed else -0.5
        tokens = random.randint(80, 260) if passed else random.randint(120, 380)

        record: dict = {
            "run":       i + 1,
            "task_type": task_type,
            "category":  category,
            "tribe_idx": tribe_idx,
            "tribe_id":  f"tribe_{tribe_idx}",
            "verdict":   "PASS" if passed else "FAIL",
            "reward":    reward,
            "exception": None if passed else "AssertionError",
            "tokens":    tokens,
        }
        all_results.append(record)

        v = "+" if passed else "-"
        exc_str = "  AssertionError" if not passed else ""
        print(
            f"  [{i+1:3d}/{n_total}] "
            f"{task_type:22s} -> T{tribe_idx} "
            f"[{v}] r={reward:+.3f}  tok={tokens}"
            f"{exc_str}"
        )

        # Append to runs.jsonl in real time (same as live mode)
        if log_dir:
            from pathlib import Path as _P
            jsonl = _P(log_dir) / "runs.jsonl"
            with open(jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

    n_pass = sum(1 for r in all_results if r["verdict"] == "PASS")
    stats = {
        "pass_rate":        n_pass / max(len(all_results), 1),
        "total_deep_audits": 0,
        "patches_stored":    0,
        "state_backend":     "mock-python",
    }
    return all_results, stats, routing_totals


# ---------------------------------------------------------------------------
# Infrastructure validation
# ---------------------------------------------------------------------------

def check_vllm(url: str) -> bool:
    """Check if vLLM is healthy and responding."""
    try:
        import urllib.request
        # Strip /v1 suffix for health endpoint
        base = url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        req = urllib.request.Request(f"{base}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"    vLLM health check failed: {e}")
        return False


def check_redis(url: str) -> bool:
    """Check if Redis is responsive."""
    try:
        import redis as redis_lib
        r = redis_lib.from_url(url, decode_responses=True)
        r.ping()
        return True
    except ImportError:
        print("    redis package not installed")
        return False
    except Exception as e:
        print(f"    Redis check failed: {e}")
        return False


def check_sandbox_image(image: str) -> bool:
    """Check if the sandbox Docker image exists."""
    try:
        import docker
        client = docker.from_env()
        client.images.get(image)
        return True
    except ImportError:
        print("    docker package not installed")
        return False
    except Exception as e:
        print(f"    Sandbox image '{image}' not found: {e}")
        return False


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    results: list[dict],
    total_time: float,
    n_tasks: int,
) -> None:
    """Print a detailed experiment report."""
    n = len(results)
    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    n_fail = n - n_pass
    total_tokens = sum(r.get("tokens", 0) for r in results)

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT REPORT")
    print(f"{'=' * 70}")
    print(f"\n  Results:")
    print(f"    Total tasks:  {n}")
    print(f"    Passed:       {n_pass} ({100*n_pass/max(n,1):.1f}%)")
    print(f"    Failed:       {n_fail} ({100*n_fail/max(n,1):.1f}%)")
    print(f"\n  Performance:")
    print(f"    Total time:   {total_time:.1f}s")
    print(f"    Avg per task: {total_time/max(n,1):.2f}s")
    if total_tokens > 0:
        print(f"    Total tokens: {total_tokens:,}")
        print(f"    Avg tokens:   {total_tokens/max(n,1):.0f}")

    # Per-task-type breakdown
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0})
    for r in results:
        key = "pass" if r["verdict"] == "PASS" else "fail"
        by_type[r["task_type"]][key] += 1

    print(f"\n  Per-task breakdown:")
    for task_type in sorted(by_type):
        stats = by_type[task_type]
        total = stats["pass"] + stats["fail"]
        rate = 100 * stats["pass"] / max(total, 1)
        print(f"    {task_type:22s}: {stats['pass']}/{total} ({rate:.0f}%)")

    # Per-category breakdown (Phase 1B — only when category is recorded)
    if any(r.get("category") for r in results):
        by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0})
        for r in results:
            cat = r.get("category", "unknown")
            key = "pass" if r["verdict"] == "PASS" else "fail"
            by_cat[cat][key] += 1

        print(f"\n  Per-category breakdown:")
        for cat in sorted(by_cat):
            stats = by_cat[cat]
            total = stats["pass"] + stats["fail"]
            rate = 100 * stats["pass"] / max(total, 1)
            print(f"    {cat:22s}: {stats['pass']}/{total} ({rate:.0f}%)")

    # Per-tribe routing
    tribe_counts: dict[int, int] = defaultdict(int)
    tribe_pass: dict[int, int] = defaultdict(int)
    for r in results:
        tribe_counts[r["tribe_idx"]] += 1
        if r["verdict"] == "PASS":
            tribe_pass[r["tribe_idx"]] += 1

    print(f"\n  Per-tribe routing:")
    for tribe_idx in sorted(tribe_counts):
        total = tribe_counts[tribe_idx]
        passed = tribe_pass[tribe_idx]
        rate = 100 * passed / max(total, 1)
        print(f"    Tribe {tribe_idx}: {total} tasks, {passed} passed ({rate:.0f}%)")

    # Error analysis
    errors = defaultdict(int)
    for r in results:
        if r["verdict"] != "PASS" and r.get("exception"):
            errors[r["exception"]] += 1
    if errors:
        print(f"\n  Error distribution:")
        for exc, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"    {exc}: {count}")

    print(f"\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCN Phase 4 — Live LLM Experiment",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock mode (for testing this script without vLLM)",
    )
    parser.add_argument(
        "-n", "--num-tasks", type=int, default=30,
        help="Number of tasks to run (default 30)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="",
        help="Directory for state and logs",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Debug logging",
    )
    parser.add_argument(
        "--skip-checks", action="store_true",
        help="Skip infrastructure validation",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Clear all previous state (Redis + local files) before running",
    )
    parser.add_argument(
        "--stratified", action="store_true",
        help="Use stratified category sampling instead of round-robin (Phase 1B)",
    )
    parser.add_argument(
        "--tasks-per-category", type=int, default=50,
        help="Number of task attempts per category when using --stratified (default: 50)",
    )
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("mcn.live")

    # Resolve log directory
    import tempfile
    if args.log_dir:
        log_dir = args.log_dir
    elif MCNConfig.LOG_DIR and MCNConfig.LOG_DIR != "/results":
        log_dir = MCNConfig.LOG_DIR
    else:
        log_dir = tempfile.mkdtemp(prefix="mcn_live_")

    use_mock = args.mock
    mode_str = "MOCK" if use_mock else "LIVE (vLLM)"

    print(f"\n{'=' * 70}")
    print(f"  MCN PHASE 4 — LIVE LLM EXPERIMENT [{mode_str}]")
    print(f"{'=' * 70}")

    # --- Step 1: Validate infrastructure ---
    if not use_mock and not args.skip_checks:
        print(f"\n  INFRASTRUCTURE CHECK:")
        vllm_ok = check_vllm(MCNConfig.VLLM_BASE_URL)
        print(f"    vLLM ({MCNConfig.VLLM_BASE_URL}): {'OK' if vllm_ok else 'FAIL'}")

        redis_ok = check_redis(MCNConfig.REDIS_URL) if MCNConfig.USE_REDIS else True
        if MCNConfig.USE_REDIS:
            print(f"    Redis ({MCNConfig.REDIS_URL}): {'OK' if redis_ok else 'FAIL'}")

        if not vllm_ok:
            print(f"\n  ERROR: vLLM is not available. Start it first:")
            print(f"    docker compose up -d vllm")
            print(f"    # Wait for model to load (~2-3 minutes)")
            sys.exit(1)
    else:
        print(f"\n  Infrastructure checks: {'SKIPPED' if args.skip_checks else 'N/A (mock mode)'}")

    # --- Step 1.5: Clear previous state if --fresh ---
    if args.fresh:
        print(f"\n  Clearing previous state...")
        from mcn.state import MCNStateStore
        store = MCNStateStore(
            redis_url=MCNConfig.REDIS_URL if MCNConfig.USE_REDIS else "",
            log_dir=log_dir,
            use_redis=MCNConfig.USE_REDIS,
        )
        store.clear_all()
        # Also clear local files
        from pathlib import Path as P
        for f in [P(log_dir) / "bandit.pkl", P(log_dir) / "runs.jsonl"]:
            if f.exists():
                f.unlink()
                print(f"    Removed {f}")
        print(f"    State cleared (Redis + local)")

    # -----------------------------------------------------------------------
    # MOCK MODE — pure-Python execution, no Ray required
    # -----------------------------------------------------------------------
    if use_mock:
        if args.stratified:
            task_seq = make_stratified_task_sequence(args.tasks_per_category)
            n_tasks  = len(task_seq)
            n_cats   = len(set(cat for _, cat, _ in task_seq))
            print(
                f"\n  Running {n_tasks} stratified tasks "
                f"({args.tasks_per_category}/category x {n_cats} categories) [MOCK]"
            )
        else:
            task_seq = make_live_task_sequence(args.num_tasks)
            n_tasks  = args.num_tasks
            print(f"\n  Running {n_tasks} round-robin tasks [MOCK]")
        print(f"  {'=' * 66}")

        t_start = time.time()
        all_results, mock_stats, mock_routing = _run_mock_simulation(
            task_seq, log_dir=log_dir
        )
        total_time = time.time() - t_start

        print_report(all_results, total_time, n_tasks)

        print(f"\n  COUNCIL FINAL STATE (mock simulation):")
        print(f"    Pass rate:      {mock_stats['pass_rate']:.1%}")
        print(f"    Routing totals: {mock_routing}")
        print(f"    Deep audits:    {mock_stats['total_deep_audits']}")
        print(f"    Patches stored: {mock_stats['patches_stored']}")
        print(f"    State backend:  {mock_stats['state_backend']}")
        print(f"\n{'=' * 70}")
        print(f"  EXPERIMENT COMPLETE")
        print(f"{'=' * 70}\n")
        return

    # -----------------------------------------------------------------------
    # LIVE MODE — full Ray distributed execution
    # -----------------------------------------------------------------------
    try:
        import ray
    except ImportError:
        print("\n  ERROR: 'ray' package is not installed.")
        print(f"  ray requires Python <= 3.12; you are running Python {sys.version.split()[0]}.")
        print("  Use --mock to run without Ray/vLLM.")
        sys.exit(1)

    # --- Step 2: Init Ray ---
    ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)

    # --- Step 3: Create actors ---
    from mcn.tribe import TribeActor
    from mcn.overseer import OverseerActor

    # Per-tribe temperature: use TRIBE_TEMPERATURES[i] if configured and valid,
    # otherwise fall back to the global TRIBE_TEMPERATURE for all tribes.
    # Different temperatures create genuine behavioural heterogeneity that the
    # GNN router (or LinUCB) can learn to exploit: low-T tribes are deterministic
    # (good for well-specified tasks), high-T tribes are creative (good for tasks
    # where the low-T tribe is consistently wrong).
    def _tribe_temp(i: int) -> float:
        temps = MCNConfig.TRIBE_TEMPERATURES
        if len(temps) == MCNConfig.NUM_TRIBES:
            return temps[i]
        return MCNConfig.TRIBE_TEMPERATURE

    tribes = []
    for i in range(MCNConfig.NUM_TRIBES):
        prompt = MCNConfig.TRIBE_PROMPTS[i] if i < len(MCNConfig.TRIBE_PROMPTS) else f"You are tribe {i}."
        temp = _tribe_temp(i)
        tribes.append(
            TribeActor.remote(
                tribe_id=f"tribe_{i}",
                system_prompt=prompt,
                model=MCNConfig.VLLM_MODEL,
                temperature=temp,
                max_tokens=MCNConfig.TRIBE_MAX_TOKENS,
                use_mock=use_mock,
                failure_bias="",  # no bias in live mode
            )
        )
        if use_mock:
            print(f"  Tribe {i}: MOCK")
        else:
            print(f"  Tribe {i}: {MCNConfig.VLLM_MODEL}")

    overseer = OverseerActor.remote()

    # Sandboxes — use subprocess sandboxes for both modes.
    # The mcn-runner Docker container provides isolation; no need for Docker-in-Docker.
    from main import MockSandboxExecutor
    n_sandboxes = MCNConfig.NUM_SANDBOXES if not use_mock else 2
    sandboxes = [MockSandboxExecutor.remote() for _ in range(n_sandboxes)]
    print(f"  {n_sandboxes} subprocess sandboxes (pytest)")

    # Council
    from mcn.council import CouncilActor
    council = CouncilActor.remote(
        tribe_handles=tribes,
        overseer_handles=[overseer],
        sandbox_handles=sandboxes,
        alpha=MCNConfig.BANDIT_ALPHA,
        log_dir=log_dir,
    )
    print(f"  Council: alpha={MCNConfig.BANDIT_ALPHA}, dim=18")
    print(f"  Log dir: {log_dir}")

    # --- Step 3.5: Start MLflow tracking (Phase 5) ---
    from mcn.tracking import MCNTracker
    tracker = MCNTracker(
        tracking_uri=MCNConfig.MLFLOW_TRACKING_URI if MCNConfig.USE_MLFLOW else "",
        experiment_name=MCNConfig.MLFLOW_EXPERIMENT_NAME,
        enabled=MCNConfig.USE_MLFLOW,
    )
    if tracker.enabled:
        from datetime import datetime
        router_type = "gnn" if MCNConfig.USE_GNN_ROUTER else "linucb"
        patch_type = "chromadb" if MCNConfig.USE_CHROMADB else "memory"
        _run_label = (f"{args.tasks_per_category}x8cat"
                      if args.stratified else f"{args.num_tasks}tasks")
        tracker.start_run(
            run_name=f"live-{_run_label}-{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                "model": MCNConfig.VLLM_MODEL,
                "n_tribes": MCNConfig.NUM_TRIBES,
                "n_tasks": (args.tasks_per_category * 8
                            if args.stratified else args.num_tasks),
                "router": router_type,
                "patch_store": patch_type,
                "alpha": MCNConfig.BANDIT_ALPHA,
                "mode": "mock" if use_mock else "live",
            },
        )
        print(f"  MLflow tracking: ENABLED (experiment={MCNConfig.MLFLOW_EXPERIMENT_NAME})")
    else:
        print(f"  MLflow tracking: disabled")

    # --- Step 4: Run experiment ---
    if args.stratified:
        task_seq = make_stratified_task_sequence(args.tasks_per_category)
        n_tasks = len(task_seq)
        n_cats  = len(set(cat for _, cat, _ in task_seq))
        print(f"\n  Running {n_tasks} stratified tasks ({args.tasks_per_category}/category x {n_cats} categories)")
    else:
        task_seq = make_live_task_sequence(args.num_tasks)
        n_tasks  = args.num_tasks
        print(f"\n  Running {n_tasks} round-robin tasks across {len(LIVE_TASKS)} task types")
    print(f"  {'='*66}")

    all_results: list[dict] = []
    t_start = time.time()

    for i, (task_type, category, task) in enumerate(task_seq):
        result = ray.get(council.run_task.remote(task))

        record = {
            "run": i + 1,
            "task_type": task_type,
            "category": category,
            "tribe_idx": int(result.tribe_id.split("_")[1]),
            "tribe_id": result.tribe_id,
            "verdict": result.verdict.name,
            "reward": result.reward,
            "exception": result.failure_info.exception_type,
            "tokens": result.generation_tokens,
        }
        all_results.append(record)

        # Write categorized record to disk in real-time for analyze_routing.py
        cat_jsonl = Path(log_dir) / "categorized_runs.jsonl"
        with open(cat_jsonl, "a", encoding="utf-8") as _fh:
            _fh.write(json.dumps(record) + "\n")

        v = "+" if record["verdict"] == "PASS" else "-"
        token_str = f" tok={record['tokens']}" if record["tokens"] > 0 else ""
        print(
            f"  [{i+1:3d}/{n_tasks}] "
            f"{task_type:20s} -> T{record['tribe_idx']} "
            f"[{v}] r={record['reward']:+.3f}{token_str}"
            f"{'  ' + record['exception'] if record['exception'] else ''}"
        )

    total_time = time.time() - t_start

    # --- Step 5: Report ---
    print_report(all_results, total_time, n_tasks)

    # Final council stats
    stats = ray.get(council.get_stats.remote())
    routing = ray.get(council.get_routing_history.remote())
    print(f"\n  COUNCIL FINAL STATE:")
    print(f"    Pass rate:      {stats['pass_rate']:.1%}")
    print(f"    Routing totals: {routing}")
    print(f"    Deep audits:    {stats['total_deep_audits']}")
    print(f"    Patches stored: {stats['patches_stored']}")
    print(f"    State backend:  {stats.get('state_backend', 'local')}")

    # Save final state
    ray.get(council.save_state.remote())

    # --- Step 6: MLflow summary + artifacts (Phase 5) ---
    if tracker.enabled:
        n_pass = sum(1 for r in all_results if r["verdict"] == "PASS")
        total_tokens = sum(r.get("tokens", 0) for r in all_results)
        tracker.log_summary({
            "pass_rate": n_pass / max(len(all_results), 1),
            "total_tasks": len(all_results),
            "total_passes": n_pass,
            "total_tokens": total_tokens,
            "total_time": total_time,
        })
        # Log runs.jsonl and categorized_runs.jsonl as artifacts
        jsonl_path = Path(log_dir) / "runs.jsonl"
        if jsonl_path.exists():
            tracker.log_artifact(str(jsonl_path))
        cat_jsonl_path = Path(log_dir) / "categorized_runs.jsonl"
        if cat_jsonl_path.exists():
            tracker.log_artifact(str(cat_jsonl_path))
        tracker.end_run()
        print(f"\n  MLflow run complete — view at {MCNConfig.MLFLOW_TRACKING_URI}")

    ray.shutdown()
    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
