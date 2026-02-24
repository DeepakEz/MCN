"""Unit tests for mcn.util.normalization — RunningScaler (Welford)."""

from __future__ import annotations

import numpy as np
import pytest

from mcn.util.normalization import RunningScaler


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestRunningScalerInit:
    def test_dim(self):
        s = RunningScaler(dim=5)
        assert s.dim == 5

    def test_count_starts_zero(self):
        s = RunningScaler(dim=3)
        assert s.count == 0

    def test_transform_before_update_is_identity(self):
        s = RunningScaler(dim=3)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # With count=0, should return zeros or the raw value (implementation defined)
        out = s.transform(x)
        assert out.shape == (3,)


# ---------------------------------------------------------------------------
# update_and_transform
# ---------------------------------------------------------------------------

class TestUpdateAndTransform:
    def test_count_increments(self):
        s = RunningScaler(dim=2)
        x = np.array([1.0, 2.0], dtype=np.float32)
        s.update_and_transform(x)
        assert s.count == 1

    def test_output_shape(self):
        s = RunningScaler(dim=4)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = s.update_and_transform(x)
        assert out.shape == (4,)

    def test_output_dtype_float32(self):
        s = RunningScaler(dim=2)
        x = np.array([5.0, 10.0], dtype=np.float32)
        out = s.update_and_transform(x)
        assert out.dtype == np.float32

    def test_constant_stream_normalizes_to_zero(self):
        s = RunningScaler(dim=1)
        x = np.array([5.0], dtype=np.float32)
        for _ in range(100):
            out = s.update_and_transform(x)
        # Constant stream: mean=5, std≈0 -> should output ~0
        assert abs(float(out[0])) < 1e-3

    def test_gaussian_stream_approximately_standardized(self):
        rng = np.random.default_rng(42)
        s = RunningScaler(dim=1)
        outputs = []
        for _ in range(1000):
            x = rng.normal(loc=3.0, scale=2.0, size=1).astype(np.float32)
            out = s.update_and_transform(x)
            outputs.append(float(out[0]))
        # After warm-up, last 500 outputs should be roughly standard normal
        tail = outputs[500:]
        assert abs(np.mean(tail)) < 0.2
        assert abs(np.std(tail) - 1.0) < 0.3

    def test_multiple_dims_independent(self):
        s = RunningScaler(dim=2)
        rng = np.random.default_rng(0)
        for _ in range(200):
            x = np.array([
                rng.normal(0.0, 1.0),
                rng.normal(100.0, 10.0),
            ], dtype=np.float32)
            s.update_and_transform(x)
        # Mean should be tracked separately per dim
        assert abs(s.mean[0]) < 0.5
        assert abs(s.mean[1] - 100.0) < 3.0


# ---------------------------------------------------------------------------
# Persistence (state_dict / from_state_dict)
# ---------------------------------------------------------------------------

class TestRunningScalerPersistence:
    def test_roundtrip_empty(self):
        s = RunningScaler(dim=3)
        sd = s.state_dict()
        restored = RunningScaler.from_state_dict(sd)
        assert restored.dim == 3
        assert restored.count == 0

    def test_roundtrip_with_data(self):
        s = RunningScaler(dim=2)
        rng = np.random.default_rng(7)
        for _ in range(50):
            x = rng.standard_normal(2).astype(np.float32)
            s.update_and_transform(x)
        sd = s.state_dict()
        restored = RunningScaler.from_state_dict(sd)
        assert restored.count == s.count
        assert np.allclose(restored.mean, s.mean)

    def test_roundtrip_continues_correctly(self):
        s = RunningScaler(dim=1)
        x = np.array([2.0], dtype=np.float32)
        for _ in range(10):
            s.update_and_transform(x)
        sd = s.state_dict()
        restored = RunningScaler.from_state_dict(sd)
        out_orig = s.update_and_transform(x)
        out_rest = restored.update_and_transform(x)
        assert np.allclose(out_orig, out_rest, atol=1e-5)
