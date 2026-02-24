"""MCN-v0.1 Patch Registry — Knowledge Diffusion via Cosine Similarity.

Implements the patch store P from the formal spec:
    P_ij = verified trajectories from Tribe i applicable to Tribe j.

A PatchEntry stores a verified solution along with its failure context
vector. When a new task arrives, the PatchRegistry performs cosine
similarity search over stored patches to find relevant few-shot hints.

Qualification rule: a solution only becomes a patch if:
    1. It passed all tests (verified correct)
    2. The task required >= min_attempts attempts (it was "hard-won")

This ensures patches capture genuinely useful knowledge — easy tasks
don't pollute the store with trivial solutions.

Usage:
    registry = PatchRegistry(dim=18)
    registry.record_attempt("task_123")
    registry.record_attempt("task_123")  # 2nd attempt
    patch = registry.register_candidate(
        task_id="task_123", description="Sort a list...",
        function_name="sort_safe", code="def sort_safe(xs): ...",
        context_vector=ctx, passed=True, failure_category="IndexError",
    )
    # Later, for a new task:
    matches = registry.search(new_context, top_k=1)
    hint = matches[0].lesson if matches else ""
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Patch Entry
# ---------------------------------------------------------------------------

@dataclass
class PatchEntry:
    """A single verified solution stored for knowledge diffusion.

    Attributes:
        patch_id: Unique identifier.
        task_description: The task's natural-language spec (for hint text).
        function_name: The function that was solved.
        code: The verified solution code.
        context_vector: Failure context at solve time (for similarity search).
        failure_category: What type of failure preceded this success.
        attempt_count: How many attempts before this solution succeeded.
        times_served: How many times this patch was served as a hint.
        created_at: Timestamp of registration.
    """
    patch_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_description: str = ""
    function_name: str = ""
    code: str = ""
    context_vector: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(18, dtype=np.float32),
    )
    failure_category: str = ""
    attempt_count: int = 0
    times_served: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def lesson(self) -> str:
        """Format this patch as a few-shot hint string.

        This is injected into the tribe's prompt via the `hint` parameter.
        """
        lines = [
            f"# Hint from a previously solved similar problem:",
            f"# Task: {self.task_description[:100]}",
            f"# Failure type overcome: {self.failure_category}",
            f"# Verified solution ({self.attempt_count} attempts to solve):",
            self.code,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Patch Registry
# ---------------------------------------------------------------------------

class PatchRegistry:
    """Cosine-similarity patch store for knowledge diffusion.

    Stores verified solutions and serves them as few-shot hints
    to tribes facing similar failures.

    Attributes:
        dim: Dimensionality of context vectors (must match FailureEncoder).
        min_attempts: Minimum attempt count for a solution to qualify as a patch.
        max_patches: Maximum number of patches stored (FIFO eviction).
        similarity_threshold: Minimum cosine similarity for a match.
    """

    def __init__(
        self,
        dim: int = 18,
        min_attempts: int = 2,
        max_patches: int = 100,
        similarity_threshold: float = 0.3,
    ) -> None:
        self.dim = dim
        self.min_attempts = min_attempts
        self.max_patches = max_patches
        self.similarity_threshold = similarity_threshold

        self._attempt_counts: dict[str, int] = {}
        self._patches: list[PatchEntry] = []

    def record_attempt(self, task_id: str) -> int:
        """Increment and return the attempt count for a task.

        Called on every task attempt (pass or fail) to track difficulty.
        """
        self._attempt_counts[task_id] = self._attempt_counts.get(task_id, 0) + 1
        return self._attempt_counts[task_id]

    def register_candidate(
        self,
        task_id: str,
        description: str,
        function_name: str,
        code: str,
        context_vector: NDArray[np.float32],
        passed: bool,
        failure_category: str = "",
    ) -> Optional[PatchEntry]:
        """Register a solution as a patch candidate.

        Qualification: passed=True AND attempt_count >= min_attempts.
        Returns the PatchEntry if qualified, None otherwise.

        FIFO eviction: when at capacity, oldest patch is removed.
        """
        if not passed:
            return None

        attempt_count = self._attempt_counts.get(task_id, 1)
        if attempt_count < self.min_attempts:
            return None

        # Create the patch entry
        patch = PatchEntry(
            task_description=description,
            function_name=function_name,
            code=code,
            context_vector=np.asarray(context_vector, dtype=np.float32).copy(),
            failure_category=failure_category,
            attempt_count=attempt_count,
        )

        # FIFO eviction at capacity
        if len(self._patches) >= self.max_patches:
            self._patches.pop(0)

        self._patches.append(patch)
        return patch

    def search(
        self,
        context_vector: NDArray[np.floating],
        top_k: int = 1,
    ) -> list[PatchEntry]:
        """Find patches most similar to the given context vector.

        Uses cosine similarity, filtered by similarity_threshold.
        Returns up to top_k patches sorted by similarity (descending).

        Also increments times_served for returned patches.
        """
        if not self._patches:
            return []

        query = np.asarray(context_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return []

        # Compute cosine similarities
        similarities: list[tuple[float, int]] = []
        for i, patch in enumerate(self._patches):
            p_norm = np.linalg.norm(patch.context_vector)
            if p_norm < 1e-10:
                continue
            cos_sim = float(np.dot(query, patch.context_vector) / (query_norm * p_norm))
            if cos_sim >= self.similarity_threshold:
                similarities.append((cos_sim, i))

        # Sort by similarity (descending), take top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, idx in similarities[:top_k]:
            patch = self._patches[idx]
            patch.times_served += 1
            results.append(patch)

        return results

    @property
    def n_patches(self) -> int:
        """Number of patches currently stored."""
        return len(self._patches)

    # ------------------------------------------------------------------
    # Persistence (JSON-safe)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        return {
            "dim": self.dim,
            "min_attempts": self.min_attempts,
            "max_patches": self.max_patches,
            "similarity_threshold": self.similarity_threshold,
            "attempt_counts": dict(self._attempt_counts),
            "patches": [
                {
                    "patch_id": p.patch_id,
                    "task_description": p.task_description,
                    "function_name": p.function_name,
                    "code": p.code,
                    "context_vector": p.context_vector.tolist(),
                    "failure_category": p.failure_category,
                    "attempt_count": p.attempt_count,
                    "times_served": p.times_served,
                    "created_at": p.created_at,
                }
                for p in self._patches
            ],
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> PatchRegistry:
        """Restore from a serialized dictionary."""
        registry = cls(
            dim=d["dim"],
            min_attempts=d["min_attempts"],
            max_patches=d["max_patches"],
            similarity_threshold=d["similarity_threshold"],
        )
        registry._attempt_counts = dict(d["attempt_counts"])
        for pd in d["patches"]:
            patch = PatchEntry(
                patch_id=pd["patch_id"],
                task_description=pd["task_description"],
                function_name=pd["function_name"],
                code=pd["code"],
                context_vector=np.array(pd["context_vector"], dtype=np.float32),
                failure_category=pd["failure_category"],
                attempt_count=pd["attempt_count"],
                times_served=pd["times_served"],
                created_at=pd["created_at"],
            )
            registry._patches.append(patch)
        return registry

    def __repr__(self) -> str:
        return (
            f"PatchRegistry(n_patches={self.n_patches}, "
            f"tasks_tracked={len(self._attempt_counts)}, "
            f"dim={self.dim})"
        )
