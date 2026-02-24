"""Unit tests for mcn.memory — PatchRegistry knowledge diffusion store."""

from __future__ import annotations

import numpy as np
import pytest

from mcn.memory import PatchEntry, PatchRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return PatchRegistry(dim=18, min_attempts=2, max_patches=10, similarity_threshold=0.3)


def _make_context(seed: int = 0, dim: int = 18) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


# ---------------------------------------------------------------------------
# PatchEntry.lesson
# ---------------------------------------------------------------------------

class TestPatchEntry:
    def test_lesson_contains_description(self):
        p = PatchEntry(task_description="Sort a list of integers", code="def f(x): return sorted(x)")
        assert "Sort a list" in p.lesson

    def test_lesson_contains_code(self):
        p = PatchEntry(code="def f(x): return sorted(x)")
        assert "def f(x)" in p.lesson

    def test_lesson_contains_failure_category(self):
        p = PatchEntry(failure_category="IndexError")
        assert "IndexError" in p.lesson


# ---------------------------------------------------------------------------
# record_attempt
# ---------------------------------------------------------------------------

class TestRecordAttempt:
    def test_first_attempt_returns_one(self, registry):
        count = registry.record_attempt("task_1")
        assert count == 1

    def test_second_attempt_returns_two(self, registry):
        registry.record_attempt("task_1")
        count = registry.record_attempt("task_1")
        assert count == 2

    def test_different_tasks_independent(self, registry):
        registry.record_attempt("task_a")
        registry.record_attempt("task_a")
        count_b = registry.record_attempt("task_b")
        assert count_b == 1


# ---------------------------------------------------------------------------
# register_candidate
# ---------------------------------------------------------------------------

class TestRegisterCandidate:
    def test_failed_task_not_registered(self, registry):
        registry.record_attempt("t1")
        registry.record_attempt("t1")
        patch = registry.register_candidate(
            task_id="t1", description="d", function_name="f",
            code="code", context_vector=_make_context(0),
            passed=False,
        )
        assert patch is None
        assert registry.n_patches == 0

    def test_single_attempt_not_qualified(self, registry):
        registry.record_attempt("t1")  # only 1 attempt
        patch = registry.register_candidate(
            task_id="t1", description="d", function_name="f",
            code="code", context_vector=_make_context(0),
            passed=True,
        )
        assert patch is None

    def test_hard_won_solution_registers(self, registry):
        registry.record_attempt("t1")
        registry.record_attempt("t1")
        patch = registry.register_candidate(
            task_id="t1", description="Sort", function_name="sort_fn",
            code="def sort_fn(x): return sorted(x)",
            context_vector=_make_context(0),
            passed=True,
            failure_category="IndexError",
        )
        assert patch is not None
        assert patch.function_name == "sort_fn"
        assert patch.attempt_count == 2
        assert registry.n_patches == 1

    def test_fifo_eviction_at_capacity(self):
        reg = PatchRegistry(dim=18, min_attempts=1, max_patches=3)
        for i in range(5):
            reg.record_attempt(f"t{i}")
            reg.register_candidate(
                task_id=f"t{i}", description=f"task {i}", function_name=f"f{i}",
                code=f"def f{i}(): pass", context_vector=_make_context(i),
                passed=True,
            )
        assert reg.n_patches == 3

    def test_patch_id_is_unique(self, registry):
        for i in range(5):
            registry.record_attempt(f"t{i}")
            registry.record_attempt(f"t{i}")
        patches = [
            registry.register_candidate(
                task_id=f"t{i}", description="d", function_name="f",
                code="code", context_vector=_make_context(i),
                passed=True,
            )
            for i in range(5)
        ]
        ids = [p.patch_id for p in patches if p is not None]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_empty_registry_returns_empty(self, registry):
        result = registry.search(_make_context(0))
        assert result == []

    def test_zero_query_returns_empty(self, registry):
        registry.record_attempt("t1")
        registry.record_attempt("t1")
        registry.register_candidate(
            task_id="t1", description="d", function_name="f",
            code="code", context_vector=_make_context(0), passed=True,
        )
        result = registry.search(np.zeros(18, dtype=np.float32))
        assert result == []

    def test_similar_context_found(self):
        reg = PatchRegistry(dim=18, min_attempts=1, similarity_threshold=0.5)
        ctx = _make_context(seed=7)
        reg.record_attempt("t1")
        reg.register_candidate(
            task_id="t1", description="d", function_name="f",
            code="code", context_vector=ctx.copy(), passed=True,
        )
        # Search with the same context — should be found
        results = reg.search(ctx, top_k=1)
        assert len(results) == 1
        assert results[0].function_name == "f"

    def test_dissimilar_context_not_found(self):
        reg = PatchRegistry(dim=18, min_attempts=1, similarity_threshold=0.9)
        ctx_a = _make_context(seed=0)
        ctx_b = _make_context(seed=99)
        reg.record_attempt("t1")
        reg.register_candidate(
            task_id="t1", description="d", function_name="f",
            code="code", context_vector=ctx_a.copy(), passed=True,
        )
        results = reg.search(ctx_b, top_k=1)
        assert results == []

    def test_top_k_limits_results(self):
        reg = PatchRegistry(dim=18, min_attempts=1, similarity_threshold=0.0)
        ctx = _make_context(seed=3)
        for i in range(5):
            reg.record_attempt(f"t{i}")
            reg.register_candidate(
                task_id=f"t{i}", description="d", function_name=f"f{i}",
                code="code", context_vector=ctx.copy(), passed=True,
            )
        results = reg.search(ctx, top_k=2)
        assert len(results) <= 2

    def test_search_increments_times_served(self):
        reg = PatchRegistry(dim=18, min_attempts=1, similarity_threshold=0.0)
        ctx = _make_context(seed=5)
        reg.record_attempt("t1")
        patch = reg.register_candidate(
            task_id="t1", description="d", function_name="f",
            code="code", context_vector=ctx.copy(), passed=True,
        )
        assert patch is not None
        reg.search(ctx, top_k=1)
        assert patch.times_served == 1


# ---------------------------------------------------------------------------
# Persistence (state_dict / from_state_dict)
# ---------------------------------------------------------------------------

class TestPatchRegistryPersistence:
    def test_roundtrip_empty(self, registry):
        sd = registry.state_dict()
        restored = PatchRegistry.from_state_dict(sd)
        assert restored.dim == registry.dim
        assert restored.n_patches == 0

    def test_roundtrip_with_patches(self):
        reg = PatchRegistry(dim=18, min_attempts=1)
        ctx = _make_context(0)
        reg.record_attempt("t1")
        reg.register_candidate(
            task_id="t1", description="Sort", function_name="sort_fn",
            code="def sort_fn(x): return sorted(x)",
            context_vector=ctx, passed=True, failure_category="IndexError",
        )
        sd = reg.state_dict()
        restored = PatchRegistry.from_state_dict(sd)
        assert restored.n_patches == 1
        assert restored._patches[0].function_name == "sort_fn"
        assert np.allclose(restored._patches[0].context_vector, ctx)
