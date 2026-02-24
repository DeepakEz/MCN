"""Shared pytest fixtures for MCN unit tests.

The Ray mock is installed at module-import time (before pytest collection)
so that MCN modules that do `import ray` at the top level can be imported
without a running Ray cluster.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Install Ray mock immediately at import time — must happen before any MCN
# module with `import ray` is collected.
# ---------------------------------------------------------------------------

if "ray" not in sys.modules:
    _ray_mock = types.ModuleType("ray")
    _ray_mock.init = lambda **kw: None          # type: ignore[attr-defined]
    _ray_mock.shutdown = lambda: None           # type: ignore[attr-defined]
    _ray_mock.get = lambda refs: (              # type: ignore[attr-defined]
        [r() if callable(r) else r for r in refs]
        if isinstance(refs, list)
        else (refs() if callable(refs) else refs)
    )
    _ray_mock.remote = lambda cls_or_fn: cls_or_fn  # type: ignore[attr-defined]
    sys.modules["ray"] = _ray_mock


# ---------------------------------------------------------------------------
# Common data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def zero_context():
    """18-dim zero context vector."""
    return np.zeros(18, dtype=np.float32)


@pytest.fixture
def unit_context():
    """18-dim all-ones context vector (normalized)."""
    v = np.ones(18, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def task_memory_sig():
    return "def sort_memory_safe(xs: list[int]) -> list[int]"


@pytest.fixture
def task_index_sig():
    return "def find_index(xs: list[int], target: int) -> int"
