"""Unit tests for mcn.protocol — types, constants, classify_exception."""

from __future__ import annotations

import numpy as np
import pytest

from mcn.protocol import (
    CONTEXT_DIM,
    EXC_TO_CATEGORY,
    FailureCategory,
    FailureSignature,
    GateVerdict,
    N_EXC,
    N_METRICS,
    N_TASK_TYPES,
    OverseerDecision,
    PatchRecord,
    Task,
    TestSuite,
    classify_exception,
)


# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------

class TestDimensionConstants:
    def test_context_dim_equals_sum(self):
        assert CONTEXT_DIM == N_TASK_TYPES + N_EXC + N_METRICS

    def test_context_dim_value(self):
        assert CONTEXT_DIM == 18

    def test_n_task_types(self):
        assert N_TASK_TYPES == 3

    def test_n_exc(self):
        assert N_EXC == 10

    def test_n_metrics(self):
        assert N_METRICS == 5


# ---------------------------------------------------------------------------
# classify_exception
# ---------------------------------------------------------------------------

class TestClassifyException:
    def test_known_exceptions(self):
        mapping = {
            "TypeError": FailureCategory.TYPE_ERROR,
            "IndexError": FailureCategory.INDEX_ERROR,
            "ValueError": FailureCategory.VALUE_ERROR,
            "KeyError": FailureCategory.KEY_ERROR,
            "AttributeError": FailureCategory.ATTRIBUTE_ERROR,
            "RecursionError": FailureCategory.RECURSION_ERROR,
            "TimeoutError": FailureCategory.TIMEOUT,
            "MemoryError": FailureCategory.MEMORY_ERROR,
            "AssertionError": FailureCategory.ASSERTION_ERROR,
            "ImportError": FailureCategory.IMPORT_ERROR,
            "ModuleNotFoundError": FailureCategory.IMPORT_ERROR,
            "SyntaxError": FailureCategory.SYNTAX_ERROR,
        }
        for exc_str, expected in mapping.items():
            assert classify_exception(exc_str) == expected, f"Failed for {exc_str}"

    def test_unknown_exception_returns_runtime_other(self):
        assert classify_exception("ZeroDivisionError") == FailureCategory.RUNTIME_OTHER
        assert classify_exception("") == FailureCategory.RUNTIME_OTHER
        assert classify_exception("FooBarError") == FailureCategory.RUNTIME_OTHER

    def test_exc_to_category_is_complete(self):
        for exc_str in EXC_TO_CATEGORY:
            result = classify_exception(exc_str)
            assert result != FailureCategory.RUNTIME_OTHER or exc_str == "RuntimeError"


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

class TestTask:
    def test_default_task_id_generated(self):
        t1 = Task()
        t2 = Task()
        assert t1.task_id != t2.task_id

    def test_task_is_frozen(self):
        t = Task(description="test")
        with pytest.raises(Exception):
            t.description = "modified"  # type: ignore[misc]

    def test_task_fields(self):
        t = Task(
            description="Sort a list",
            function_name="sort_list",
            timeout_seconds=15.0,
            memory_limit_mb=512,
        )
        assert t.description == "Sort a list"
        assert t.function_name == "sort_list"
        assert t.timeout_seconds == 15.0
        assert t.memory_limit_mb == 512
        assert t.unit_tests == ""


# ---------------------------------------------------------------------------
# TestSuite.combined_source
# ---------------------------------------------------------------------------

class TestTestSuite:
    def test_combined_empty(self):
        suite = TestSuite()
        assert suite.combined_source() == ""

    def test_combined_joins_nonempty(self):
        suite = TestSuite(unit_tests="def test_a(): pass", self_tests="def test_b(): pass")
        combined = suite.combined_source()
        assert "test_a" in combined
        assert "test_b" in combined

    def test_combined_skips_whitespace_only(self):
        suite = TestSuite(unit_tests="  \n  ", self_tests="def test_x(): pass")
        combined = suite.combined_source()
        assert "test_x" in combined
        assert combined.strip() == "def test_x(): pass"

    def test_combined_order(self):
        suite = TestSuite(
            unit_tests="UNIT",
            self_tests="SELF",
            overseer_tests="OVERSEER",
            fuzz_tests="FUZZ",
        )
        combined = suite.combined_source()
        assert combined.index("UNIT") < combined.index("SELF")
        assert combined.index("SELF") < combined.index("OVERSEER")
        assert combined.index("OVERSEER") < combined.index("FUZZ")


# ---------------------------------------------------------------------------
# FailureSignature.to_feature_vector
# ---------------------------------------------------------------------------

class TestFailureSignature:
    def test_success_is_success(self):
        fs = FailureSignature(category=FailureCategory.NONE)
        assert fs.is_success

    def test_failure_is_not_success(self):
        fs = FailureSignature(category=FailureCategory.TYPE_ERROR)
        assert not fs.is_success

    def test_feature_vector_shape(self):
        fs = FailureSignature(
            category=FailureCategory.INDEX_ERROR,
            tests_passed=3,
            tests_failed=1,
            elapsed_seconds=2.5,
        )
        vec = fs.to_feature_vector(dim=32)
        assert vec.shape == (32,)
        assert vec.dtype == np.float32

    def test_feature_vector_one_hot(self):
        fs = FailureSignature(category=FailureCategory.NONE)
        vec = fs.to_feature_vector(dim=32)
        n_cats = len(FailureCategory)
        cat_slice = vec[:n_cats]
        assert cat_slice[FailureCategory.NONE.value - 1] == pytest.approx(1.0)
        assert cat_slice.sum() == pytest.approx(1.0)

    def test_feature_vector_padding(self):
        fs = FailureSignature()
        vec = fs.to_feature_vector(dim=64)
        assert vec.shape == (64,)

    def test_test_ratios(self):
        fs = FailureSignature(tests_passed=3, tests_failed=1, tests_errored=0)
        vec = fs.to_feature_vector(dim=32)
        n_cats = len(FailureCategory)
        # pass ratio should be 0.75, fail ratio 0.25
        assert vec[n_cats + 3] == pytest.approx(0.75)
        assert vec[n_cats + 4] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# PatchRecord
# ---------------------------------------------------------------------------

class TestPatchRecord:
    def test_utility_zero_when_never_applied(self):
        p = PatchRecord()
        assert p.utility == 0.0

    def test_utility_ratio(self):
        p = PatchRecord(times_applied=10, verified_successes=7)
        assert p.utility == pytest.approx(0.7)

    def test_unique_patch_ids(self):
        ids = {PatchRecord().patch_id for _ in range(50)}
        assert len(ids) == 50
