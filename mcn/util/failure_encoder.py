"""MCN-v0.1 Failure Encoder — Trace-to-Vector Embedding.

Parses raw sandbox traces (stdout + stderr) and structured metrics
into a fixed-size feature vector for LinUCB context.

This is the Phi operator from the formal spec:
    s = Phi(f) in R^d

The encoder composes:
    1. Task-type keyword detection -> one-hot vector (size N_TASK_TYPES)
    2. Regex-based exception detection -> one-hot vector (size 10)
    3. Normalized resource metrics -> Z-scored floats (size 5)
    4. Total output dimension is fixed at CONTEXT_DIM (default 18)

The context vector layout (dim=18 default):
    [0..3)    one-hot task-type keywords (memory, timeout, index)
    [3..13)   one-hot exception type (10 slots)
    [13]      Z-scored log(runtime_ms + 1)
    [14]      Z-scored peak_mem_mb
    [15]      test pass ratio (0..1)
    [16]      test fail ratio (0..1)
    [17]      Z-scored log(n_tests + 1)

The Z-score normalization uses RunningScaler (Welford's algorithm)
so statistics are updated incrementally — no batch computation needed.

Usage:
    encoder = FailureEncoder(dim=18)
    ctx = encoder.encode_context(
        function_name="sort_memory_safe",
        trace="MemoryError: ...",
        metrics={"runtime_ms": 450},
    )
"""

from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from mcn.util.normalization import RunningScaler


# ---------------------------------------------------------------------------
# Exception types we detect via regex (ordered, stable indices)
# ---------------------------------------------------------------------------

# The 10 exception families we track.  Index in this list = one-hot slot.
EXCEPTION_TYPES: list[str] = [
    "IndexError",
    "KeyError",
    "TypeError",
    "ValueError",
    "RecursionError",
    "TimeoutError",
    "MemoryError",
    "AttributeError",
    "AssertionError",     # pytest assertion failures
    "SyntaxError",
]

N_EXC = len(EXCEPTION_TYPES)  # 10
_N_EXC = N_EXC  # backward-compat alias

# Pre-compiled pattern:  matches "IndexError", "KeyError:", etc.
# Uses word boundaries to avoid matching substrings like "MyCustomIndexError"
_EXC_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(e) for e in EXCEPTION_TYPES) + r")\b"
)

# Resource metric slots (after the one-hot block)
# [runtime_ms, peak_mem_mb, test_pass_ratio, test_fail_ratio, n_tests_log]
N_METRICS = 5
_N_METRICS = N_METRICS  # backward-compat alias

# Task-type keywords detected in function names (for task-type one-hot)
# This gives LinUCB the ability to learn per-task-type routing.
TASK_KEYWORDS: list[str] = ["memory", "timeout", "index"]
N_TASK_TYPES = len(TASK_KEYWORDS)  # 3
_N_TASK_TYPES = N_TASK_TYPES  # backward-compat alias


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class FailureEncoder:
    """Encode task type + sandbox traces + metrics into fixed-size vectors.

    The output vector layout (dim=18 default):
        [0..3)    one-hot task-type keywords (memory, timeout, index)
        [3..13)   one-hot exception type (10 slots)
        [13]      Z-scored log(runtime_ms + 1)
        [14]      Z-scored peak_mem_mb
        [15]      test pass ratio (0..1)
        [16]      test fail ratio (0..1)
        [17]      Z-scored log(n_tests + 1)

    The Z-score normalizer is updated incrementally via RunningScaler
    so it adapts to the observed distribution of metrics.

    Attributes:
        dim: Total output vector dimensionality.
        metric_scaler: RunningScaler for the continuous metric block.
    """

    def __init__(self, dim: int = _N_TASK_TYPES + _N_EXC + _N_METRICS) -> None:
        if dim < _N_TASK_TYPES + _N_EXC + _N_METRICS:
            raise ValueError(
                f"dim must be >= {_N_TASK_TYPES + _N_EXC + _N_METRICS}, got {dim}"
            )
        self.dim = dim
        # Scaler covers only the continuous metrics (not the one-hots)
        self.metric_scaler = RunningScaler(dim=_N_METRICS)

    def task_type_onehot(self, function_name: str) -> NDArray[np.float32]:
        """Convert a function name to a task-type one-hot vector.

        Scans the function name for known task-type keywords
        (memory, timeout, index). Returns a one-hot vector where
        the first matching keyword gets activated.

        This gives LinUCB the ability to distinguish task types
        BEFORE any failure has been observed.
        """
        vec = np.zeros(_N_TASK_TYPES, dtype=np.float32)
        fn_lower = function_name.lower()
        for i, kw in enumerate(TASK_KEYWORDS):
            if kw in fn_lower:
                vec[i] = 1.0
                break  # one-hot: only first match
        return vec

    def detect_exception(self, trace: str) -> Optional[str]:
        """Find the first known exception type in a trace string.

        Scans stdout + stderr for recognized exception names.
        Returns the exception type string, or None if not found.
        """
        match = _EXC_PATTERN.search(trace)
        if match:
            return match.group(1)
        return None

    def exception_to_onehot(self, exc_type: Optional[str]) -> NDArray[np.float32]:
        """Convert an exception type to a one-hot vector.

        Returns a zero vector if exc_type is None or unrecognized.
        """
        vec = np.zeros(_N_EXC, dtype=np.float32)
        if exc_type and exc_type in EXCEPTION_TYPES:
            idx = EXCEPTION_TYPES.index(exc_type)
            vec[idx] = 1.0
        return vec

    def encode(
        self,
        trace: str = "",
        metrics: Optional[dict] = None,
        function_name: str = "",
    ) -> NDArray[np.float32]:
        """Encode task type + failure trace + metrics into a fixed-size vector.

        Args:
            trace: Raw sandbox output (stdout + stderr combined).
                   The encoder scans this for exception type names.
            metrics: Optional dict with keys:
                - runtime_ms:   float, execution time in milliseconds
                - peak_mem_mb:  float, peak memory in MB
                - tests_passed: int
                - tests_failed: int
                - tests_total:  int
                Missing keys default to 0.
            function_name: Task function name (for task-type one-hot).

        Returns:
            Feature vector of shape (self.dim,).
        """
        if metrics is None:
            metrics = {}

        # --- 1. Task-type one-hot (3 dims) ---
        task_vec = self.task_type_onehot(function_name)

        # --- 2. Exception one-hot (10 dims) ---
        exc_type = self.detect_exception(trace)
        exc_vec = self.exception_to_onehot(exc_type)

        # --- 3. Continuous metrics (5 dims) ---
        runtime_ms = metrics.get("runtime_ms", 0.0)
        peak_mem_mb = metrics.get("peak_mem_mb", 0.0)
        tests_passed = metrics.get("tests_passed", 0)
        tests_failed = metrics.get("tests_failed", 0)
        tests_total = metrics.get("tests_total", 0)

        # Log-scale runtime (smooths outliers, makes distribution more Gaussian)
        log_runtime = math.log1p(max(runtime_ms, 0.0))
        log_tests = math.log1p(max(tests_total, 0))

        # Test pass/fail ratios
        if tests_total > 0:
            pass_ratio = tests_passed / tests_total
            fail_ratio = tests_failed / tests_total
        else:
            pass_ratio = 0.0
            fail_ratio = 0.0

        raw_metrics = np.array(
            [log_runtime, peak_mem_mb, pass_ratio, fail_ratio, log_tests],
            dtype=np.float32,
        )

        # Z-score normalize (update scaler, then transform)
        normed_metrics = self.metric_scaler.update_and_transform(raw_metrics)

        # --- 4. Concatenate + pad to dim ---
        combined = np.concatenate([task_vec, exc_vec, normed_metrics])

        # Pad if dim > 18 (future extensibility)
        if len(combined) < self.dim:
            combined = np.concatenate([
                combined,
                np.zeros(self.dim - len(combined), dtype=np.float32),
            ])

        return combined.astype(np.float32)

    def encode_from_sandbox(
        self,
        stdout: str,
        stderr: str,
        elapsed_seconds: float,
        tests_passed: int = 0,
        tests_failed: int = 0,
        tests_total: int = 0,
        peak_memory_mb: float = 0.0,
        function_name: str = "",
    ) -> NDArray[np.float32]:
        """Convenience: encode directly from SandboxResult fields.

        This is the primary interface used by the Council.
        """
        trace = stdout + "\n" + stderr
        metrics = {
            "runtime_ms": elapsed_seconds * 1000.0,
            "peak_mem_mb": peak_memory_mb,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "tests_total": tests_total,
        }
        return self.encode(trace=trace, metrics=metrics, function_name=function_name)

    def encode_task_context(self, function_name: str) -> NDArray[np.float32]:
        """Encode ONLY the task-type features (for first attempt with no failure).

        Returns a vector with task-type one-hot and zeros for failure/metrics.
        This gives LinUCB per-task-type signal even before any failures.
        """
        task_vec = self.task_type_onehot(function_name)
        rest = np.zeros(self.dim - _N_TASK_TYPES, dtype=np.float32)
        return np.concatenate([task_vec, rest]).astype(np.float32)

    def state_dict(self) -> dict:
        """Serialize encoder state (scaler statistics)."""
        return {
            "dim": self.dim,
            "metric_scaler": self.metric_scaler.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> FailureEncoder:
        """Restore from serialized state."""
        enc = cls(dim=d["dim"])
        enc.metric_scaler = RunningScaler.from_state_dict(d["metric_scaler"])
        return enc

    def __repr__(self) -> str:
        return (
            f"FailureEncoder(dim={self.dim}, "
            f"samples_seen={self.metric_scaler.count})"
        )
