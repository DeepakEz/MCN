"""MCN-v0.1 Running Scaler — Welford's Online Algorithm.

Provides incremental Z-score normalization for failure signature
embeddings and bandit context vectors. No batch computation needed;
updates are O(1) per sample.

Usage:
    scaler = RunningScaler(dim=32)
    scaler.update(vec)           # update statistics
    z = scaler.transform(vec)    # Z-score normalize
    z = scaler.update_and_transform(vec)  # both in one call
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class RunningScaler:
    """Welford's online algorithm for per-dimension mean and variance.

    Maintains running statistics and provides Z-score normalization.
    Thread-safety note: not thread-safe. In MCN, each Ray actor owns
    its own scaler instance, so this is fine.

    Attributes:
        dim: Dimensionality of input vectors.
        count: Number of samples seen so far.
        mean: Running mean vector.
        m2: Running sum-of-squared-deviations (for variance).
        eps: Floor for standard deviation to prevent division by zero.
    """

    __slots__ = ("dim", "count", "mean", "m2", "eps")

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.count: int = 0
        self.mean: NDArray[np.float64] = np.zeros(dim, dtype=np.float64)
        self.m2: NDArray[np.float64] = np.zeros(dim, dtype=np.float64)
        self.eps = eps

    @property
    def variance(self) -> NDArray[np.float64]:
        """Population variance (not sample variance).

        Returns zeros if fewer than 2 samples have been seen.
        """
        if self.count < 2:
            return np.zeros(self.dim, dtype=np.float64)
        return self.m2 / self.count

    @property
    def std(self) -> NDArray[np.float64]:
        """Standard deviation, floored at eps."""
        return np.maximum(np.sqrt(self.variance), self.eps)

    def update(self, x: NDArray[np.floating]) -> None:
        """Incorporate a new sample into running statistics.

        Welford's algorithm:
            delta  = x - mean
            mean  += delta / count
            delta2 = x - mean   (note: using *updated* mean)
            m2    += delta * delta2

        Args:
            x: Input vector of shape (dim,).
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.dim,):
            raise ValueError(
                f"Expected shape ({self.dim},), got {x.shape}"
            )
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def transform(self, x: NDArray[np.floating]) -> NDArray[np.float32]:
        """Z-score normalize a vector using current statistics.

        Returns the raw vector (cast to float32) if fewer than 2
        samples have been seen — no reliable variance estimate yet.

        Args:
            x: Input vector of shape (dim,).

        Returns:
            Normalized vector of shape (dim,) as float32.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.dim,):
            raise ValueError(
                f"Expected shape ({self.dim},), got {x.shape}"
            )
        if self.count < 2:
            return x.astype(np.float32)
        normalized = (x - self.mean) / self.std
        return normalized.astype(np.float32)

    def update_and_transform(
        self, x: NDArray[np.floating]
    ) -> NDArray[np.float32]:
        """Update statistics with x, then return its Z-score.

        This is the typical call in the MCN pipeline: each new failure
        signature embedding is incorporated and normalized in one step.
        """
        self.update(x)
        return self.transform(x)

    def update_batch(self, xs: NDArray[np.floating]) -> None:
        """Incorporate a batch of samples.

        Args:
            xs: Array of shape (n, dim).
        """
        xs = np.asarray(xs, dtype=np.float64)
        if xs.ndim != 2 or xs.shape[1] != self.dim:
            raise ValueError(
                f"Expected shape (n, {self.dim}), got {xs.shape}"
            )
        for x in xs:
            self.update(x)

    # ------------------------------------------------------------------
    # Persistence (JSONL-compatible)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        return {
            "dim": self.dim,
            "count": self.count,
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> RunningScaler:
        """Restore from a serialized dictionary."""
        scaler = cls(dim=d["dim"], eps=d["eps"])
        scaler.count = d["count"]
        scaler.mean = np.array(d["mean"], dtype=np.float64)
        scaler.m2 = np.array(d["m2"], dtype=np.float64)
        return scaler

    def save(self, path: str | Path) -> None:
        """Write state to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.state_dict(), f)

    @classmethod
    def load(cls, path: str | Path) -> RunningScaler:
        """Load state from a JSON file."""
        with open(path) as f:
            return cls.from_state_dict(json.load(f))

    def __repr__(self) -> str:
        return (
            f"RunningScaler(dim={self.dim}, count={self.count}, "
            f"mean_norm={np.linalg.norm(self.mean):.4f})"
        )
