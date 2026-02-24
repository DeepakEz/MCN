"""Unit tests for mcn.overseer — test generation helpers and OverseerActor."""

from __future__ import annotations

import pytest

from mcn.overseer import (
    _generate_adversarial_tests,
    _generate_edge_case_tests,
    _generate_existence_test,
    _generate_hypothesis_tests,
    _generate_return_type_test,
    _infer_return_type,
    _infer_strategy,
    _parse_params,
    OverseerActor,
)
from mcn.protocol import TestSuite


# ---------------------------------------------------------------------------
# _parse_params
# ---------------------------------------------------------------------------

class TestParseParams:
    def test_single_param(self):
        params = _parse_params("def f(xs: list[int]) -> list[int]")
        assert params == [("xs", "list[int]")]

    def test_two_params(self):
        params = _parse_params("def f(xs: list[int], n: int) -> int")
        assert len(params) == 2
        assert params[0] == ("xs", "list[int]")
        assert params[1] == ("n", "int")

    def test_no_type_hints(self):
        params = _parse_params("def f(xs)")
        assert len(params) == 1
        assert params[0][0] == "xs"

    def test_no_params(self):
        params = _parse_params("def f() -> int")
        assert params == []

    def test_invalid_signature(self):
        params = _parse_params("not a function signature")
        assert params == []


# ---------------------------------------------------------------------------
# _generate_existence_test
# ---------------------------------------------------------------------------

class TestExistenceTest:
    def test_contains_function_name(self):
        src = _generate_existence_test("sort_list")
        assert "sort_list" in src

    def test_contains_callable_check(self):
        src = _generate_existence_test("my_fn")
        assert "callable" in src

    def test_is_valid_def(self):
        src = _generate_existence_test("foo")
        assert src.startswith("def test_")


# ---------------------------------------------------------------------------
# _generate_edge_case_tests
# ---------------------------------------------------------------------------

class TestEdgeCaseTests:
    def test_list_int_generates_tests(self):
        src = _generate_edge_case_tests("sort_fn", "def sort_fn(xs: list[int]) -> list[int]")
        assert "def test_" in src
        assert "sort_fn" in src

    def test_no_params_fallback(self):
        src = _generate_edge_case_tests("my_fn", "not parseable")
        assert "callable" in src

    def test_empty_and_nonempty_tests(self):
        src = _generate_edge_case_tests("f", "def f(xs: list[int]) -> list[int]")
        assert "empty" in src.lower() or "[]" in src


# ---------------------------------------------------------------------------
# _generate_return_type_test
# ---------------------------------------------------------------------------

class TestReturnTypeTest:
    def test_list_return_generates_isinstance(self):
        src = _generate_return_type_test("f", "def f(xs: list[int]) -> list")
        assert "isinstance" in src
        assert "list" in src

    def test_int_return(self):
        src = _generate_return_type_test("f", "def f(n: int) -> int")
        assert "isinstance" in src
        assert "int" in src

    def test_no_return_annotation(self):
        src = _generate_return_type_test("f", "def f(xs: list[int])")
        assert src == ""


# ---------------------------------------------------------------------------
# _infer_strategy / _infer_return_type
# ---------------------------------------------------------------------------

class TestInferHelpers:
    def test_infer_strategy_list_int(self):
        s = _infer_strategy("def f(xs: list[int]) -> list[int]")
        assert "integers" in s

    def test_infer_strategy_list_str(self):
        s = _infer_strategy("def f(xs: list[str]) -> list[str]")
        assert "text" in s or "str" in s.lower()

    def test_infer_strategy_unknown(self):
        s = _infer_strategy("def f(x: dict) -> dict")
        assert s == ""

    def test_infer_return_type_list(self):
        assert _infer_return_type("def f(xs: list[int]) -> list[int]") == "list"

    def test_infer_return_type_int(self):
        assert _infer_return_type("def f(n: int) -> int") == "int"

    def test_infer_return_type_none(self):
        assert _infer_return_type("def f(xs: list[int])") == ""


# ---------------------------------------------------------------------------
# _generate_hypothesis_tests
# ---------------------------------------------------------------------------

class TestHypothesisTests:
    def test_single_list_int_param(self):
        src = _generate_hypothesis_tests(
            "sort_fn", "def sort_fn(xs: list[int]) -> list[int]"
        )
        assert "from hypothesis import" in src
        assert "@given" in src
        assert "test_sort_fn_no_crash" in src

    def test_multi_param_returns_empty(self):
        src = _generate_hypothesis_tests(
            "find_fn", "def find_fn(xs: list[int], target: int) -> int"
        )
        assert src == ""

    def test_unknown_type_returns_empty(self):
        src = _generate_hypothesis_tests("f", "def f(x: dict) -> dict")
        assert src == ""

    def test_return_type_test_generated_for_list(self):
        src = _generate_hypothesis_tests(
            "sort_fn", "def sort_fn(xs: list[int]) -> list[int]"
        )
        assert "test_sort_fn_return_type" in src


# ---------------------------------------------------------------------------
# _generate_adversarial_tests (Strategy B)
# ---------------------------------------------------------------------------

class TestAdversarialTests:
    def test_list_int_generates_tests(self):
        src = _generate_adversarial_tests(
            "sort_fn", "def sort_fn(xs: list[int]) -> list[int]"
        )
        assert "def test_" in src
        assert "sort_fn" in src

    def test_empty_boundary_present(self):
        src = _generate_adversarial_tests(
            "sort_fn", "def sort_fn(xs: list[int]) -> list[int]"
        )
        assert "empty" in src.lower() or "[]" in src

    def test_large_input_test_present(self):
        src = _generate_adversarial_tests(
            "sort_fn", "def sort_fn(xs: list[int]) -> list[int]"
        )
        assert "large" in src.lower() or "10_000" in src or "10000" in src

    def test_negatives_test_present(self):
        src = _generate_adversarial_tests(
            "sort_fn", "def sort_fn(xs: list[int]) -> list[int]"
        )
        assert "negativ" in src.lower() or "-1" in src

    def test_duplicates_test_present(self):
        src = _generate_adversarial_tests(
            "sort_fn", "def sort_fn(xs: list[int]) -> list[int]"
        )
        assert "same" in src.lower() or "dup" in src.lower() or "* 1000" in src

    def test_int_param_generates_tests(self):
        src = _generate_adversarial_tests("f", "def f(n: int) -> int")
        assert "def test_" in src

    def test_string_param_generates_tests(self):
        src = _generate_adversarial_tests("f", "def f(s: str) -> str")
        assert "def test_" in src

    def test_no_params_returns_empty(self):
        src = _generate_adversarial_tests("f", "not a signature")
        assert src == ""

    def test_crash_wrapped_in_try_except(self):
        src = _generate_adversarial_tests(
            "sort_fn", "def sort_fn(xs: list[int]) -> list[int]"
        )
        assert "try:" in src
        assert "except Exception" in src
        assert "AssertionError" in src

    def test_multiple_tests_generated(self):
        src = _generate_adversarial_tests(
            "sort_fn", "def sort_fn(xs: list[int]) -> list[int]"
        )
        count = src.count("def test_")
        assert count >= 4  # empty, single, duplicates, negatives, large, extreme


# ---------------------------------------------------------------------------
# OverseerActor.generate_suite
# ---------------------------------------------------------------------------

class TestOverseerActorGenerateSuite:
    def test_returns_test_suite(self):
        overseer = OverseerActor()
        suite = overseer.generate_suite(
            task_description="Sort a list",
            function_name="sort_fn",
            input_signature="def sort_fn(xs: list[int]) -> list[int]",
        )
        assert isinstance(suite, TestSuite)

    def test_overseer_tests_not_empty(self):
        overseer = OverseerActor()
        suite = overseer.generate_suite(
            task_description="Sort",
            function_name="sort_fn",
            input_signature="def sort_fn(xs: list[int]) -> list[int]",
        )
        assert suite.overseer_tests.strip() != ""

    def test_fuzz_tests_generated_for_list_int(self):
        overseer = OverseerActor()
        suite = overseer.generate_suite(
            task_description="Sort",
            function_name="sort_fn",
            input_signature="def sort_fn(xs: list[int]) -> list[int]",
        )
        assert "@given" in suite.fuzz_tests

    def test_unit_tests_passed_through(self):
        overseer = OverseerActor()
        suite = overseer.generate_suite(
            task_description="Sort",
            function_name="sort_fn",
            input_signature="def sort_fn(xs: list[int]) -> list[int]",
            unit_tests="def test_custom(): pass",
        )
        assert "test_custom" in suite.unit_tests

    def test_stats_updated(self):
        overseer = OverseerActor()
        overseer.generate_suite("d", "f", "def f(xs: list[int]) -> list[int]")
        stats = overseer.get_stats()
        assert stats["total_suites_generated"] == 1
        assert stats["total_tests_generated"] > 0

    def test_quality_score_in_range(self):
        overseer = OverseerActor()
        for _ in range(5):
            overseer.generate_suite("d", "f", "def f(xs: list[int]) -> list[int]")
        q = overseer.get_quality_score()
        assert 0.0 <= q <= 1.0

    def test_record_suite_outcome_affects_stats(self):
        overseer = OverseerActor()
        overseer.generate_suite("d", "f", "def f(xs: list[int]) -> list[int]")
        overseer.record_suite_outcome(tests_passed=8, tests_total=10)
        stats = overseer.get_stats()
        assert stats["feedback_pass_rate"] is not None
        assert abs(stats["feedback_pass_rate"] - 0.8) < 1e-6

    def test_strategy_coverage_tracked(self):
        overseer = OverseerActor()
        overseer.generate_suite("d", "sort_fn", "def sort_fn(xs: list[int]) -> list[int]")
        stats = overseer.get_stats()
        sc = stats["strategy_coverage"]
        assert sc["structural_rate"] == pytest.approx(1.0)
        assert sc["adversarial_rate"] == pytest.approx(1.0)
        assert sc["hypothesis_rate"] == pytest.approx(1.0)
