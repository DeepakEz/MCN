"""Unit tests for mcn.util.failure_encoder — FailureEncoder."""

from __future__ import annotations

import numpy as np
import pytest

from mcn.util.failure_encoder import (
    EXCEPTION_TYPES,
    TASK_KEYWORDS,
    N_EXC,
    N_METRICS,
    N_TASK_TYPES,
    FailureEncoder,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestEncoderConstants:
    def test_n_exc_matches_list(self):
        assert N_EXC == len(EXCEPTION_TYPES)

    def test_n_task_types_matches_list(self):
        assert N_TASK_TYPES == len(TASK_KEYWORDS)

    def test_n_metrics(self):
        assert N_METRICS == 5

    def test_exception_types_have_no_duplicates(self):
        assert len(EXCEPTION_TYPES) == len(set(EXCEPTION_TYPES))

    def test_task_keywords_have_no_duplicates(self):
        assert len(TASK_KEYWORDS) == len(set(TASK_KEYWORDS))


# ---------------------------------------------------------------------------
# FailureEncoder construction
# ---------------------------------------------------------------------------

class TestFailureEncoderInit:
    def test_default_dim(self):
        enc = FailureEncoder()
        assert enc.dim == 18

    def test_custom_dim(self):
        enc = FailureEncoder(dim=24)
        assert enc.dim == 24

    def test_dim_too_small_raises(self):
        with pytest.raises(ValueError, match="dim must be"):
            FailureEncoder(dim=5)

    def test_metric_scaler_dim(self):
        enc = FailureEncoder()
        assert enc.metric_scaler.dim == N_METRICS


# ---------------------------------------------------------------------------
# task_type_onehot
# ---------------------------------------------------------------------------

class TestTaskTypeOnehot:
    def test_memory_keyword(self):
        enc = FailureEncoder()
        vec = enc.task_type_onehot("sort_memory_safe")
        assert vec[TASK_KEYWORDS.index("memory")] == 1.0
        assert vec.sum() == pytest.approx(1.0)

    def test_timeout_keyword(self):
        enc = FailureEncoder()
        vec = enc.task_type_onehot("compute_timeout_limit")
        assert vec[TASK_KEYWORDS.index("timeout")] == 1.0

    def test_index_keyword(self):
        enc = FailureEncoder()
        vec = enc.task_type_onehot("find_index_in_list")
        assert vec[TASK_KEYWORDS.index("index")] == 1.0

    def test_no_keyword_is_zero(self):
        enc = FailureEncoder()
        vec = enc.task_type_onehot("sort_numbers")
        assert vec.sum() == pytest.approx(0.0)

    def test_case_insensitive(self):
        enc = FailureEncoder()
        vec = enc.task_type_onehot("MEMORY_SAFE_SORT")
        assert vec[TASK_KEYWORDS.index("memory")] == 1.0

    def test_output_shape(self):
        enc = FailureEncoder()
        vec = enc.task_type_onehot("any_name")
        assert vec.shape == (N_TASK_TYPES,)


# ---------------------------------------------------------------------------
# detect_exception / exception_to_onehot
# ---------------------------------------------------------------------------

class TestExceptionDetection:
    def test_detects_index_error(self):
        enc = FailureEncoder()
        assert enc.detect_exception("list index out of range\nIndexError: ...") == "IndexError"

    def test_detects_type_error(self):
        enc = FailureEncoder()
        assert enc.detect_exception("TypeError: unsupported operand") == "TypeError"

    def test_no_exception_returns_none(self):
        enc = FailureEncoder()
        assert enc.detect_exception("All tests passed!") is None

    def test_does_not_match_substring(self):
        enc = FailureEncoder()
        # "MyCustomIndexError" should NOT match
        result = enc.detect_exception("raise MyCustomIndexError('boom')")
        assert result is None

    def test_onehot_known_exception(self):
        enc = FailureEncoder()
        vec = enc.exception_to_onehot("IndexError")
        assert vec[EXCEPTION_TYPES.index("IndexError")] == 1.0
        assert vec.sum() == pytest.approx(1.0)

    def test_onehot_unknown_exception(self):
        enc = FailureEncoder()
        vec = enc.exception_to_onehot("ZeroDivisionError")
        assert vec.sum() == pytest.approx(0.0)

    def test_onehot_none(self):
        enc = FailureEncoder()
        vec = enc.exception_to_onehot(None)
        assert vec.sum() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

class TestEncode:
    def test_output_shape(self):
        enc = FailureEncoder()
        ctx = enc.encode()
        assert ctx.shape == (18,)

    def test_output_dtype(self):
        enc = FailureEncoder()
        ctx = enc.encode()
        assert ctx.dtype == np.float32

    def test_encode_no_crash_empty(self):
        enc = FailureEncoder()
        ctx = enc.encode(trace="", metrics={}, function_name="")
        assert ctx.shape == (18,)

    def test_task_type_visible_in_output(self):
        enc = FailureEncoder()
        ctx = enc.encode(function_name="sort_memory_safe")
        # First N_TASK_TYPES dims should have the memory one-hot
        assert ctx[TASK_KEYWORDS.index("memory")] == pytest.approx(1.0)

    def test_exception_visible_in_output(self):
        enc = FailureEncoder()
        ctx = enc.encode(trace="IndexError: list index out of range")
        idx_slot = N_TASK_TYPES + EXCEPTION_TYPES.index("IndexError")
        assert ctx[idx_slot] == pytest.approx(1.0)

    def test_custom_dim_pads(self):
        enc = FailureEncoder(dim=24)
        ctx = enc.encode()
        assert ctx.shape == (24,)

    def test_metric_scaler_updates(self):
        enc = FailureEncoder()
        before = enc.metric_scaler.count
        enc.encode(metrics={"runtime_ms": 500.0, "tests_total": 5})
        assert enc.metric_scaler.count == before + 1


# ---------------------------------------------------------------------------
# encode_task_context
# ---------------------------------------------------------------------------

class TestEncodeTaskContext:
    def test_shape(self):
        enc = FailureEncoder()
        ctx = enc.encode_task_context("sort_memory_safe")
        assert ctx.shape == (18,)

    def test_task_type_set(self):
        enc = FailureEncoder()
        ctx = enc.encode_task_context("find_index")
        assert ctx[TASK_KEYWORDS.index("index")] == pytest.approx(1.0)

    def test_rest_zeros(self):
        enc = FailureEncoder()
        ctx = enc.encode_task_context("sort_memory_safe")
        # Dims beyond task-type block should be zero (no failure info)
        assert np.all(ctx[N_TASK_TYPES:] == 0.0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestFailureEncoderPersistence:
    def test_roundtrip(self):
        enc = FailureEncoder()
        enc.encode(metrics={"runtime_ms": 200.0, "tests_total": 3})
        sd = enc.state_dict()
        restored = FailureEncoder.from_state_dict(sd)
        assert restored.dim == enc.dim
        assert restored.metric_scaler.count == enc.metric_scaler.count

    def test_state_dict_keys(self):
        enc = FailureEncoder()
        sd = enc.state_dict()
        assert "dim" in sd
        assert "metric_scaler" in sd
