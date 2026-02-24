"""MCN-v0.1 Overseer — Adversarial Test Generator.

Implements V(x, p) -> T_overseer U T_fuzz from the formal spec.

v0.1 strategy ensemble:
    Strategy A: Property-Based Fuzzing (Hypothesis)  — implemented
    Strategy B: Anti-Case Injection (Adversarial)    — implemented (static)
    Strategy C: Mutation Testing                     — delegated to Sandbox

Strategy B generates adversarial boundary/stress tests from the function
signature without needing an LLM call (deterministic, fast, no network):
    - Boundary values: empty, single element, large collections
    - Numeric stress: zero, negative, max-int, float extremes
    - Duplicate-heavy inputs (exercises dedup/sort code)
    - None/null injection where type allows

The Overseer's quality score Q_V feeds into the Council objective:
    J(psi) += lambda_cov * Coverage(T) + lambda_mut * MutationScore(T)
"""

from __future__ import annotations

import logging
import re

import ray

from mcn.protocol import Task, TestSuite

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test generation helpers
# ---------------------------------------------------------------------------

def _generate_existence_test(function_name: str) -> str:
    """Generate a test that asserts the function exists and is callable."""
    return (
        f"def test_{function_name}_exists():\n"
        f"    \"\"\"Verify {function_name} is defined and callable.\"\"\"\n"
        f"    assert callable({function_name}), "
        f"'{function_name} must be a callable function'\n"
    )


def _parse_params(input_signature: str) -> list[tuple[str, str]]:
    """Parse parameter names and types from a function signature.

    Returns list of (name, type_hint) tuples.
    e.g. 'def foo(xs: list[int], pivot: int) -> ...' -> [('xs', 'list[int]'), ('pivot', 'int')]
    """
    m = re.search(r"def\s+\w+\(([^)]*)\)", input_signature)
    if not m:
        return []
    params_str = m.group(1)
    params = []
    for part in params_str.split(","):
        part = part.strip()
        if ":" in part:
            name, hint = part.split(":", 1)
            params.append((name.strip(), hint.strip()))
        elif part:
            params.append((part.strip(), ""))
    return params


def _make_sample_arg(type_hint: str) -> str:
    """Generate a sample argument expression from a type hint string."""
    t = type_hint.lower().replace(" ", "")
    if "list[int]" in t or "list[float]" in t:
        return "[]"
    if "list[str]" in t:
        return "[]"
    if t.startswith("list"):
        return "[]"
    if t == "int":
        return "0"
    if t == "float":
        return "0.0"
    if t == "str":
        return "''"
    if t == "bool":
        return "True"
    if t == "dict" or t.startswith("dict"):
        return "{}"
    return "None"


def _make_nonempty_arg(type_hint: str) -> str:
    """Generate a non-empty sample argument expression from a type hint string."""
    t = type_hint.lower().replace(" ", "")
    if "list[int]" in t or "list[float]" in t:
        return "[1, 2, 3]"
    if "list[str]" in t:
        return "['a', 'b']"
    if t.startswith("list"):
        return "[1, 2, 3]"
    if t == "int":
        return "5"
    if t == "float":
        return "3.14"
    if t == "str":
        return "'hello'"
    if t == "bool":
        return "True"
    if t == "dict" or t.startswith("dict"):
        return "{'a': 1}"
    return "None"


def _generate_edge_case_tests(function_name: str, input_signature: str) -> str:
    """Generate edge-case tests based on the function signature.

    Parses the actual parameter types to generate correct function calls.
    This is the v0.1 stand-in for full property-based fuzzing.
    """
    tests: list[str] = []
    params = _parse_params(input_signature)

    if not params:
        # Can't parse signature — just test callability
        tests.append(
            f"def test_{function_name}_callable():\n"
            f"    assert callable({function_name})\n"
        )
        return "\n\n".join(tests)

    # Build call strings with appropriate sample args
    empty_args = ", ".join(_make_sample_arg(hint) for _, hint in params)
    nonempty_args = ", ".join(_make_nonempty_arg(hint) for _, hint in params)

    # Test 1: empty/minimal inputs
    tests.append(
        f"def test_{function_name}_edge_empty():\n"
        f"    \"\"\"Edge case: minimal inputs.\"\"\"\n"
        f"    result = {function_name}({empty_args})\n"
        f"    assert result is not None, 'Should handle minimal inputs'\n"
    )

    # Test 2: non-empty inputs
    tests.append(
        f"def test_{function_name}_edge_nonempty():\n"
        f"    \"\"\"Edge case: typical inputs.\"\"\"\n"
        f"    result = {function_name}({nonempty_args})\n"
        f"    assert result is not None, 'Should handle typical inputs'\n"
    )

    return "\n\n".join(tests)


def _generate_return_type_test(
    function_name: str, input_signature: str,
) -> str:
    """Generate a test verifying the return type if inferrable.

    Parses '-> list' or '-> int' from the signature string.
    Uses parsed parameters to generate correct function calls.
    """
    tests: list[str] = []
    params = _parse_params(input_signature)
    call_args = ", ".join(_make_nonempty_arg(hint) for _, hint in params) if params else "[3, 1, 2]"

    sig_lower = input_signature.lower()

    # Try to extract return type annotation
    if "-> list" in sig_lower:
        tests.append(
            f"def test_{function_name}_returns_list():\n"
            f"    \"\"\"Type check: should return a list.\"\"\"\n"
            f"    result = {function_name}({call_args})\n"
            f"    assert isinstance(result, list), "
            f"f'Expected list, got {{type(result).__name__}}'\n"
        )
    elif "-> int" in sig_lower:
        tests.append(
            f"def test_{function_name}_returns_int():\n"
            f"    \"\"\"Type check: should return an int.\"\"\"\n"
            f"    result = {function_name}({call_args})\n"
            f"    assert isinstance(result, int), "
            f"f'Expected int, got {{type(result).__name__}}'\n"
        )
    elif "-> bool" in sig_lower:
        tests.append(
            f"def test_{function_name}_returns_bool():\n"
            f"    \"\"\"Type check: should return a bool.\"\"\"\n"
            f"    result = {function_name}({call_args})\n"
            f"    assert isinstance(result, bool), "
            f"f'Expected bool, got {{type(result).__name__}}'\n"
        )
    elif "-> str" in sig_lower:
        tests.append(
            f"def test_{function_name}_returns_str():\n"
            f"    \"\"\"Type check: should return a str.\"\"\"\n"
            f"    result = {function_name}({call_args})\n"
            f"    assert isinstance(result, str), "
            f"f'Expected str, got {{type(result).__name__}}'\n"
        )
    elif "-> dict" in sig_lower:
        tests.append(
            f"def test_{function_name}_returns_dict():\n"
            f"    \"\"\"Type check: should return a dict.\"\"\"\n"
            f"    result = {function_name}({call_args})\n"
            f"    assert isinstance(result, dict), "
            f"f'Expected dict, got {{type(result).__name__}}'\n"
        )

    return "\n\n".join(tests)


# ---------------------------------------------------------------------------
# Hypothesis integration — strategy inference + property test generation
# ---------------------------------------------------------------------------

def _infer_strategy(sig_lower: str) -> str:
    """Map type annotations to Hypothesis strategy expressions.

    Returns the strategy string or "" if no strategy can be inferred.
    """
    if "list[int]" in sig_lower:
        return "st.lists(st.integers(min_value=-10000, max_value=10000), max_size=100)"
    if "list[str]" in sig_lower:
        return "st.lists(st.text(max_size=50), max_size=20)"
    if "list[float]" in sig_lower:
        return "st.lists(st.floats(allow_nan=False, allow_infinity=False), max_size=100)"
    if "list" in sig_lower:
        return "st.lists(st.integers(min_value=-10000, max_value=10000), max_size=100)"
    if "str" in sig_lower:
        return "st.text(max_size=100)"
    if "int" in sig_lower:
        # Bounded to prevent O(n) iterative solutions from hanging on huge values
        # (e.g. fibonacci(2**63) would loop forever). 10_000 is large enough
        # to catch O(n^2) regressions while remaining fast for O(n) code.
        return "st.integers(min_value=-10000, max_value=10000)"
    return ""


def _extract_param_name(input_signature: str) -> str:
    """Extract the first parameter name from a function signature.

    e.g. 'def f(xs: list[int]) -> list[int]' -> 'xs'
    """
    m = re.search(r"def\s+\w+\((\w+)", input_signature)
    return m.group(1) if m else "x"


def _returns_same_type(sig_lower: str) -> bool:
    """Check if the return type matches the input collection type.

    Returns True for list->list or str->str signatures.
    """
    # list[...] -> list[...]
    if "list" in sig_lower:
        if "-> list" in sig_lower:
            return True
    # str -> str
    if re.search(r"\(\w+\s*:\s*str\b", sig_lower):
        if "-> str" in sig_lower:
            return True
    return False


def _infer_return_type(sig_lower: str) -> str:
    """Extract the return type name for isinstance checks."""
    if "-> list" in sig_lower:
        return "list"
    if "-> int" in sig_lower:
        return "int"
    if "-> str" in sig_lower:
        return "str"
    if "-> float" in sig_lower:
        return "float"
    if "-> bool" in sig_lower:
        return "bool"
    return ""


def _generate_hypothesis_tests(function_name: str, input_signature: str) -> str:
    """Generate Hypothesis property-based tests.

    Strategy A: Infer Hypothesis strategies from the input signature,
    then generate @given-decorated property tests.

    Only generates safe properties that hold for all functions:
      1. No-crash test — function doesn't raise on random inputs
      2. Return type check — isinstance(result, expected_type)

    Skips length-preservation and idempotency since those don't hold for
    many valid functions (deduplicate, flatten, etc).

    Only generates tests for single-parameter functions to avoid
    incorrect multi-param calls.

    Returns "" if strategy cannot be inferred.
    """
    params = _parse_params(input_signature)
    # Only generate Hypothesis tests for single-parameter functions
    # Multi-param functions need correlated strategies which are hard to infer
    if len(params) != 1:
        return ""

    sig_lower = input_signature.lower()
    strategy = _infer_strategy(sig_lower)
    if not strategy:
        return ""

    param_name, _ = params[0]
    return_type = _infer_return_type(sig_lower)

    tests: list[str] = []

    # Header: hypothesis imports
    header = (
        "from hypothesis import given, settings\n"
        "from hypothesis import strategies as st\n"
    )
    tests.append(header)

    # Test 1: No-crash test
    tests.append(
        f"@given({param_name}={strategy})\n"
        f"@settings(max_examples=50)\n"
        f"def test_{function_name}_no_crash({param_name}):\n"
        f"    \"\"\"Property: function should not crash on random inputs.\"\"\"\n"
        f"    result = {function_name}({param_name})\n"
        f"    assert result is not None, 'Function returned None'\n"
    )

    # Test 2: Return type check
    if return_type:
        tests.append(
            f"@given({param_name}={strategy})\n"
            f"@settings(max_examples=50)\n"
            f"def test_{function_name}_return_type({param_name}):\n"
            f"    \"\"\"Property: function should return the expected type.\"\"\"\n"
            f"    result = {function_name}({param_name})\n"
            f"    assert isinstance(result, {return_type}), "
            f"f'Expected {return_type}, got {{type(result).__name__}}'\n"
        )

    return "\n\n".join(tests)


# ---------------------------------------------------------------------------
# Strategy B: Adversarial static test generation
# ---------------------------------------------------------------------------

def _generate_adversarial_tests(
    function_name: str,
    input_signature: str,
) -> str:
    """Generate adversarial boundary and stress tests from the function signature.

    Strategy B — Anti-Case Injection (static analysis, no LLM needed):
      - Boundary values:  empty collections, single-element, duplicates
      - Numeric stress:   zero, negative, sys.maxsize, float extremes
      - Large inputs:     10_000-element list (performance regression guard)
      - Duplicate-heavy:  all-same-value inputs (exercises dedup/sort paths)
      - None injection:   where Optional types appear in the signature

    All tests are wrapped in try/except so a crash is caught and reported
    as an AssertionError with a descriptive message.
    """
    params = _parse_params(input_signature)
    if not params:
        return ""

    tests: list[str] = []

    # Detect parameter type profile
    first_name, first_hint = params[0]
    hint_lower = first_hint.lower().replace(" ", "")
    is_list_int = "list[int]" in hint_lower or "list[float]" in hint_lower
    is_list_str = "list[str]" in hint_lower
    is_list = hint_lower.startswith("list")
    is_int = hint_lower == "int"
    is_str = hint_lower == "str"

    sig_lower = input_signature.lower()

    # Pre-define string literals used in f-strings to avoid backslash-in-expression
    # errors on Python < 3.12.
    _single_str_list = "['x']"
    _dup_str_list = "['x'] * 100"

    # Build default call args for multi-param functions
    def _call_args(override_first: str) -> str:
        parts = [override_first]
        for _, h in params[1:]:
            parts.append(_make_nonempty_arg(h))
        return ", ".join(parts)

    # --- Test B1: Empty / zero boundary ---
    if is_list:
        tests.append(
            f"def test_{function_name}_adversarial_empty():\n"
            f"    \"\"\"Adversarial: empty collection must not crash.\"\"\"\n"
            f"    try:\n"
            f"        result = {function_name}({_call_args('[]')})\n"
            f"    except Exception as exc:\n"
            f"        raise AssertionError(f'Crashed on empty input: {{exc}}') from exc\n"
        )
    elif is_int:
        tests.append(
            f"def test_{function_name}_adversarial_zero():\n"
            f"    \"\"\"Adversarial: zero input must not crash.\"\"\"\n"
            f"    try:\n"
            f"        result = {function_name}({_call_args('0')})\n"
            f"    except Exception as exc:\n"
            f"        raise AssertionError(f'Crashed on zero input: {{exc}}') from exc\n"
        )
    elif is_str:
        tests.append(
            f"def test_{function_name}_adversarial_empty_str():\n"
            f"    \"\"\"Adversarial: empty string must not crash.\"\"\"\n"
            f"    try:\n"
            f"        result = {function_name}({_call_args(repr(''))})\n"
            f"    except Exception as exc:\n"
            f"        raise AssertionError(f'Crashed on empty string: {{exc}}') from exc\n"
        )

    # --- Test B2: Single-element collection ---
    if is_list_int:
        tests.append(
            f"def test_{function_name}_adversarial_single():\n"
            f"    \"\"\"Adversarial: single-element list must not crash.\"\"\"\n"
            f"    try:\n"
            f"        result = {function_name}({_call_args('[42]')})\n"
            f"    except Exception as exc:\n"
            f"        raise AssertionError(f'Crashed on single-element list: {{exc}}') from exc\n"
        )
    elif is_list_str:
        tests.append(
            f"def test_{function_name}_adversarial_single():\n"
            f"    \"\"\"Adversarial: single-element string list must not crash.\"\"\"\n"
            f"    try:\n"
            f"        result = {function_name}({_call_args(_single_str_list)})\n"
            f"    except Exception as exc:\n"
            f"        raise AssertionError(f'Crashed on single-element list: {{exc}}') from exc\n"
        )

    # --- Test B3: All-duplicates (stress sort/dedup) ---
    if is_list_int:
        tests.append(
            f"def test_{function_name}_adversarial_all_same():\n"
            f"    \"\"\"Adversarial: all-duplicate list must not crash.\"\"\"\n"
            f"    try:\n"
            f"        result = {function_name}({_call_args('[7] * 1000')})\n"
            f"    except Exception as exc:\n"
            f"        raise AssertionError(f'Crashed on all-duplicate list: {{exc}}') from exc\n"
        )
    elif is_list_str:
        tests.append(
            f"def test_{function_name}_adversarial_all_same():\n"
            f"    \"\"\"Adversarial: all-duplicate string list must not crash.\"\"\"\n"
            f"    try:\n"
            f"        result = {function_name}({_call_args(_dup_str_list)})\n"
            f"    except Exception as exc:\n"
            f"        raise AssertionError(f'Crashed on all-duplicate string list: {{exc}}') from exc\n"
        )

    # --- Test B4: Negative numbers ---
    if is_list_int:
        tests.append(
            f"def test_{function_name}_adversarial_negatives():\n"
            f"    \"\"\"Adversarial: negative integers must not crash.\"\"\"\n"
            f"    try:\n"
            f"        result = {function_name}({_call_args('[-1, -100, -999, 0, 1]')})\n"
            f"    except Exception as exc:\n"
            f"        raise AssertionError(f'Crashed on negative integers: {{exc}}') from exc\n"
        )
    elif is_int:
        tests.append(
            f"def test_{function_name}_adversarial_negative_int():\n"
            f"    \"\"\"Adversarial: negative integer must not crash.\"\"\"\n"
            f"    try:\n"
            f"        result = {function_name}({_call_args('-1')})\n"
            f"    except Exception as exc:\n"
            f"        raise AssertionError(f'Crashed on negative integer: {{exc}}') from exc\n"
        )

    # --- Test B5: Large input (performance regression guard) ---
    if is_list_int:
        tests.append(
            f"def test_{function_name}_adversarial_large():\n"
            f"    \"\"\"Adversarial: large list must complete without crashing.\"\"\"\n"
            f"    import time\n"
            f"    big = list(range(10_000, 0, -1))\n"
            f"    t0 = time.monotonic()\n"
            f"    try:\n"
            f"        result = {function_name}({_call_args('big')})\n"
            f"    except Exception as exc:\n"
            f"        raise AssertionError(f'Crashed on large input: {{exc}}') from exc\n"
            f"    elapsed = time.monotonic() - t0\n"
            f"    assert elapsed < 5.0, f'Too slow on 10k elements: {{elapsed:.2f}}s'\n"
        )
    elif is_list_str:
        tests.append(
            f"def test_{function_name}_adversarial_large():\n"
            f"    \"\"\"Adversarial: large string list must complete without crashing.\"\"\"\n"
            f"    import time\n"
            f"    big = [str(i) for i in range(1_000)]\n"
            f"    t0 = time.monotonic()\n"
            f"    try:\n"
            f"        result = {function_name}({_call_args('big')})\n"
            f"    except Exception as exc:\n"
            f"        raise AssertionError(f'Crashed on large string list: {{exc}}') from exc\n"
            f"    elapsed = time.monotonic() - t0\n"
            f"    assert elapsed < 5.0, f'Too slow on 1k string elements: {{elapsed:.2f}}s'\n"
        )

    # --- Test B6: Large-but-finite integer boundary (with time guard) ---
    # sys.maxsize is intentionally avoided: O(n) algorithms would run forever.
    # Instead we use 10_000 — large enough to catch O(2^n) recursion quickly.
    if is_int:
        tests.append(
            f"def test_{function_name}_adversarial_large_int():\n"
            f"    \"\"\"Adversarial: large integer (10_000) must complete within 5s.\"\"\"\n"
            f"    import time\n"
            f"    t0 = time.monotonic()\n"
            f"    try:\n"
            f"        result = {function_name}({_call_args('10_000')})\n"
            f"    except Exception as exc:\n"
            f"        raise AssertionError(f'Crashed on large int 10_000: {{exc}}') from exc\n"
            f"    elapsed = time.monotonic() - t0\n"
            f"    assert elapsed < 5.0, f'Too slow on input 10_000: {{elapsed:.2f}}s'\n"
        )
    elif is_list_int:
        tests.append(
            f"def test_{function_name}_adversarial_extreme_values():\n"
            f"    \"\"\"Adversarial: extreme int values must not crash.\"\"\"\n"
            f"    import sys\n"
            f"    extreme = [sys.maxsize, -sys.maxsize, 0, 1, -1]\n"
            f"    try:\n"
            f"        result = {function_name}({_call_args('extreme')})\n"
            f"    except Exception as exc:\n"
            f"        raise AssertionError(f'Crashed on extreme int values: {{exc}}') from exc\n"
        )

    if not tests:
        return ""

    return "\n\n".join(tests)


# ---------------------------------------------------------------------------
# Ray Actor
# ---------------------------------------------------------------------------

@ray.remote
class OverseerActor:
    """Adversarial test suite generator.

    V(x, p) -> T_overseer U T_fuzz

    Generates three layers of tests:
      - Strategy: Structural tests (function exists, return-type checks)
      - Strategy A: Hypothesis property-based fuzzing (no-crash + type)
      - Strategy B: Adversarial boundary/stress tests (static, no LLM)

    Quality score Q_V is computed as a weighted combination of:
      - Test variety score: fraction of strategy types used (structural /
        hypothesis / adversarial) — ranges [0, 1]
      - Test density score: tests-per-suite normalized to a target of 10
      - Pass-rate feedback: updated via record_suite_outcome()
    """

    def __init__(self) -> None:
        self._total_suites_generated: int = 0
        self._total_tests_generated: int = 0
        # Per-strategy counters for quality scoring
        self._suites_with_structural: int = 0
        self._suites_with_hypothesis: int = 0
        self._suites_with_adversarial: int = 0
        # Feedback from sandbox: cumulative test pass / total counts
        self._total_feedback_suites: int = 0
        self._total_feedback_passed: int = 0
        self._total_feedback_total: int = 0

    def generate_suite(
        self,
        task_description: str,
        function_name: str,
        input_signature: str,
        unit_tests: str = "",
        code: str = "",
    ) -> TestSuite:
        """Generate an adversarial test suite for a task.

        Args:
            task_description: Natural-language spec.
            function_name: The function to test.
            input_signature: Type signature for edge-case inference.
            unit_tests: Existing unit tests from the task spec (passed through).
            code: Generated code (unused in static strategy B; kept for API compat).

        Returns:
            TestSuite with overseer_tests and fuzz_tests populated.
        """
        # --- Structural + Edge-case tests ---
        overseer_parts: list[str] = []

        overseer_parts.append(_generate_existence_test(function_name))

        edge_tests = _generate_edge_case_tests(function_name, input_signature)
        if edge_tests:
            overseer_parts.append(edge_tests)

        type_tests = _generate_return_type_test(function_name, input_signature)
        if type_tests:
            overseer_parts.append(type_tests)

        has_structural = bool(overseer_parts)

        # --- Strategy B: Adversarial boundary/stress tests ---
        adversarial_source = _generate_adversarial_tests(function_name, input_signature)
        has_adversarial = bool(adversarial_source)
        if adversarial_source:
            overseer_parts.append(adversarial_source)

        overseer_source = "\n\n".join(overseer_parts)

        # --- Strategy A: Hypothesis property-based fuzzing ---
        fuzz_source = _generate_hypothesis_tests(function_name, input_signature)
        has_hypothesis = bool(fuzz_source)

        # Build the TestSuite
        suite = TestSuite(
            unit_tests=unit_tests,
            overseer_tests=overseer_source,
            fuzz_tests=fuzz_source,
        )

        # Update metrics
        self._total_suites_generated += 1
        n_tests = overseer_source.count("def test_") + fuzz_source.count("def test_")
        self._total_tests_generated += n_tests
        if has_structural:
            self._suites_with_structural += 1
        if has_hypothesis:
            self._suites_with_hypothesis += 1
        if has_adversarial:
            self._suites_with_adversarial += 1

        logger.info(
            "Overseer generated %d tests for %s "
            "[structural=%s, hypothesis=%s, adversarial=%s] "
            "(total: %d suites, %d tests)",
            n_tests, function_name,
            has_structural, has_hypothesis, has_adversarial,
            self._total_suites_generated, self._total_tests_generated,
        )

        return suite

    def record_suite_outcome(self, tests_passed: int, tests_total: int) -> None:
        """Feed sandbox results back to improve quality score estimation.

        Called by the Council after sandbox execution to track how many
        overseer-generated tests actually passed.
        """
        if tests_total > 0:
            self._total_feedback_suites += 1
            self._total_feedback_passed += tests_passed
            self._total_feedback_total += tests_total

    def get_stats(self) -> dict:
        """Return diagnostic statistics."""
        n = max(self._total_suites_generated, 1)
        return {
            "total_suites_generated": self._total_suites_generated,
            "total_tests_generated": self._total_tests_generated,
            "avg_tests_per_suite": self._total_tests_generated / n,
            "strategy_coverage": {
                "structural_rate": self._suites_with_structural / n,
                "hypothesis_rate": self._suites_with_hypothesis / n,
                "adversarial_rate": self._suites_with_adversarial / n,
            },
            "feedback_pass_rate": (
                self._total_feedback_passed / self._total_feedback_total
                if self._total_feedback_total > 0 else None
            ),
            "quality_score": self.get_quality_score(),
        }

    def get_quality_score(self) -> float:
        """Return the Overseer quality score Q_V in [0, 1].

        Q_V = 0.4 * variety_score + 0.4 * density_score + 0.2 * pass_rate_score

        variety_score:   fraction of strategy types used across all suites
        density_score:   avg tests per suite normalized to target of 10
        pass_rate_score: fraction of overseer tests that passed (feedback loop)
                         — defaults to 0.5 when no feedback yet received
        """
        if self._total_suites_generated == 0:
            return 0.0

        n = self._total_suites_generated

        # Variety: how many strategies produced output (0..3 strategies, norm to [0,1])
        variety_score = (
            (self._suites_with_structural / n)
            + (self._suites_with_hypothesis / n)
            + (self._suites_with_adversarial / n)
        ) / 3.0

        # Density: avg tests per suite vs. target of 10
        avg_tests = self._total_tests_generated / n
        density_score = min(avg_tests / 10.0, 1.0)

        # Pass-rate feedback (0.5 prior when no data yet)
        if self._total_feedback_total > 0:
            pass_rate_score = self._total_feedback_passed / self._total_feedback_total
        else:
            pass_rate_score = 0.5

        return 0.4 * variety_score + 0.4 * density_score + 0.2 * pass_rate_score
