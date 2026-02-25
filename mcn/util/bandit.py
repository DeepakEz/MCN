"""MCN-v0.1 LinUCB Contextual Bandit — Council Router.

Implements the routing policy rho_psi(. | x, s) from the formal spec.

Policy (UCB phase):
    a_t = argmax_i [ theta_i^T x_t + alpha * sqrt(x_t^T A_i^{-1} x_t) ]

With epsilon-greedy warm-up to prevent cold-start routing collapse:
    With probability epsilon: pick a uniformly random available arm.
    With probability 1-epsilon: pick the UCB-argmax arm.
    Epsilon decays multiplicatively each step: epsilon <- max(epsilon_min, epsilon * epsilon_decay).

where:
    theta_i : learned weight vector for arm (Tribe) i
    A_i     : covariance matrix (uncertainty) for arm i
    alpha   : exploration parameter (UCB width)

Key design choices for MCN-v0.1:
    - Closed-form recursive least squares updates (no SGD instability).
    - Mask support: tribes in a cooling period are excluded from selection.
    - Bounded regret O(sqrt(T)) — real theoretical guarantee.
    - Epsilon-greedy warm-up prevents cold-start collapse (all UCB scores equal
      at t=0, so hard argmax always picks arm 0).
    - gamma << eta separation is automatic via confidence bound shrinkage.

Usage:
    bandit = LinUCB(n_arms=4, dim=64, alpha=1.0, epsilon=0.3, epsilon_decay=0.99)
    arm = bandit.select(context, mask=[True, True, False, True])
    bandit.update(context, arm=0, reward=1.0)
"""

from __future__ import annotations

import json
import random as _random
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class LinUCB:
    """Linear Upper Confidence Bound contextual bandit with epsilon-greedy warm-up.

    Each arm i maintains:
        A_i : (dim x dim) positive-definite matrix = I + sum(x x^T)
        b_i : (dim,) vector = sum(r * x)
        theta_i : A_i^{-1} b_i (recomputed on demand)

    Epsilon-greedy warm-up prevents cold-start routing collapse: at t=0 all
    UCB scores are equal (all theta=0, all A=I) so hard argmax always picks
    arm 0, resulting in degenerate 100/0/0 routing.  With epsilon > 0, the
    first ~1/(1-epsilon_decay) steps are random, giving all arms a chance to
    accumulate signal before the UCB phase takes over.

    Attributes:
        n_arms: Number of arms (tribes).
        dim: Dimensionality of context vector.
        alpha: Exploration parameter controlling UCB width.
        epsilon: Current exploration probability (decays over time).
        epsilon_min: Minimum exploration probability (floor).
        epsilon_decay: Multiplicative decay factor per select() call.
        A: Array of covariance matrices, shape (n_arms, dim, dim).
        b: Array of reward-weighted context sums, shape (n_arms, dim).
        counts: Per-arm pull counts.
        explore_counts: Number of steps chosen via random exploration.
    """

    def __init__(
        self,
        n_arms: int,
        dim: int,
        alpha: float = 1.0,
        epsilon: float = 0.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.99,
    ) -> None:
        if n_arms <= 0:
            raise ValueError(f"n_arms must be positive, got {n_arms}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

        self.n_arms = n_arms
        self.dim = dim
        self.alpha = alpha

        # Epsilon-greedy warm-up parameters
        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = epsilon_decay

        # A_i = I_d for each arm (ridge initialization)
        self.A: NDArray[np.float64] = np.stack(
            [np.eye(dim, dtype=np.float64) for _ in range(n_arms)]
        )  # shape: (n_arms, dim, dim)

        # b_i = 0 for each arm
        self.b: NDArray[np.float64] = np.zeros(
            (n_arms, dim), dtype=np.float64
        )  # shape: (n_arms, dim)

        # pull counts (for diagnostics / logging)
        self.counts: NDArray[np.int64] = np.zeros(n_arms, dtype=np.int64)

        # exploration counts (steps where random was chosen over UCB)
        self.explore_counts: int = 0

        # total updates (for diagnostics)
        self.total_updates: int = 0

    def _compute_theta(self, arm: int) -> NDArray[np.float64]:
        """Compute theta_i = A_i^{-1} b_i via solve (numerically stable)."""
        return np.linalg.solve(self.A[arm], self.b[arm])

    def _compute_ucb(
        self,
        arm: int,
        x: NDArray[np.float64],
        theta: NDArray[np.float64],
    ) -> float:
        """Compute UCB score: theta^T x + alpha * sqrt(x^T A^{-1} x).

        Uses solve instead of explicit inverse for numerical stability.
        """
        exploitation = float(theta @ x)

        # x^T A^{-1} x = x^T (A^{-1} x) — compute via solve
        A_inv_x = np.linalg.solve(self.A[arm], x)
        exploration = self.alpha * float(np.sqrt(x @ A_inv_x))

        return exploitation + exploration

    def select(
        self,
        context: NDArray[np.floating],
        mask: Optional[list[bool]] = None,
    ) -> int:
        """Select an arm (tribe) given context, respecting cooling mask.

        Uses epsilon-greedy exploration to avoid cold-start collapse:
            - With probability epsilon: choose a uniformly random available arm.
            - With probability 1-epsilon: choose the UCB-argmax arm.
        Epsilon decays multiplicatively each call: epsilon <- max(epsilon_min, epsilon * epsilon_decay).

        Args:
            context: Context vector of shape (dim,).
                     Concatenation of [TaskEmbedding, FailureSignature].
            mask: Boolean list of length n_arms. True = available,
                  False = in cooling period (excluded from selection).
                  If None, all arms are available.

        Returns:
            Index of the selected arm.

        Raises:
            ValueError: If no arms are available (all masked out).
        """
        x = np.asarray(context, dtype=np.float64)
        if x.shape != (self.dim,):
            raise ValueError(
                f"Context shape mismatch: expected ({self.dim},), got {x.shape}"
            )

        if mask is None:
            mask = [True] * self.n_arms

        if not any(mask):
            raise ValueError("All arms are masked — no tribe available")

        available = [i for i in range(self.n_arms) if mask[i]]

        # Epsilon-greedy: random arm with probability epsilon
        if self.epsilon > 0.0 and _random.random() < self.epsilon:
            chosen = _random.choice(available)
            self.explore_counts += 1
            # Decay epsilon toward epsilon_min
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return chosen

        # UCB argmax phase
        best_arm = -1
        best_score = -np.inf

        for i in available:
            theta_i = self._compute_theta(i)
            score = self._compute_ucb(i, x, theta_i)
            if score > best_score:
                best_score = score
                best_arm = i

        # Decay epsilon even on UCB steps (consistent schedule)
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return best_arm

    def select_with_scores(
        self,
        context: NDArray[np.floating],
        mask: Optional[list[bool]] = None,
    ) -> tuple[int, NDArray[np.float64]]:
        """Select an arm and return all UCB scores (for diagnostics).

        Applies the same epsilon-greedy logic as select().

        Returns:
            (selected_arm, scores) where scores[i] = -inf if masked.
            On a random explore step, the chosen arm's score is set to +inf
            to signal that selection was random, not UCB-driven.
        """
        x = np.asarray(context, dtype=np.float64)
        if x.shape != (self.dim,):
            raise ValueError(
                f"Context shape mismatch: expected ({self.dim},), got {x.shape}"
            )

        if mask is None:
            mask = [True] * self.n_arms

        # If all arms are masked (e.g., all cooling), fall back to all-available.
        # Council.py guards against this, but we degrade gracefully here too.
        if not any(mask):
            mask = [True] * self.n_arms

        available = [i for i in range(self.n_arms) if mask[i]]
        scores = np.full(self.n_arms, -np.inf, dtype=np.float64)

        # Compute UCB scores for all available arms regardless (useful for logging)
        for i in available:
            theta_i = self._compute_theta(i)
            scores[i] = self._compute_ucb(i, x, theta_i)

        # Epsilon-greedy: random arm with probability epsilon
        if self.epsilon > 0.0 and _random.random() < self.epsilon:
            chosen = _random.choice(available)
            self.explore_counts += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            # Mark as explore step in scores (chosen arm gets +inf sentinel)
            explore_scores = scores.copy()
            explore_scores[chosen] = np.inf
            return chosen, explore_scores

        best_arm = int(np.argmax(scores))
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return best_arm, scores

    def update(
        self,
        context: NDArray[np.floating],
        arm: int,
        reward: float,
    ) -> None:
        """Update arm statistics after observing a reward.

        Recursive least squares:
            A_i <- A_i + x x^T
            b_i <- b_i + r * x

        Args:
            context: Context vector of shape (dim,).
            arm: Index of the arm that was pulled.
            reward: Observed reward (typically E - costs).
        """
        x = np.asarray(context, dtype=np.float64)
        if x.shape != (self.dim,):
            raise ValueError(
                f"Context shape mismatch: expected ({self.dim},), got {x.shape}"
            )
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(
                f"arm index {arm} out of range [0, {self.n_arms})"
            )

        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.counts[arm] += 1
        self.total_updates += 1

    def get_uncertainty(self, arm: int) -> float:
        """Return trace of A_i^{-1} — overall uncertainty for an arm.

        Lower = more confident. Useful for monitoring convergence.
        """
        A_inv = np.linalg.inv(self.A[arm])
        return float(np.trace(A_inv))

    def get_theta(self, arm: int) -> NDArray[np.float64]:
        """Return current weight vector for an arm (for inspection)."""
        return self._compute_theta(arm)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary.

        Stored as bandit.json per the MCN-v0.1 file convention.
        Includes epsilon state so warm-up continues correctly across restarts.
        """
        return {
            "type": "linucb",
            "n_arms": self.n_arms,
            "dim": self.dim,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "explore_counts": self.explore_counts,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "counts": self.counts.tolist(),
            "total_updates": self.total_updates,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> LinUCB:
        """Restore from a serialized dictionary."""
        bandit = cls(
            n_arms=d["n_arms"],
            dim=d["dim"],
            alpha=d["alpha"],
            epsilon=d.get("epsilon", 0.0),
            epsilon_min=d.get("epsilon_min", 0.05),
            epsilon_decay=d.get("epsilon_decay", 0.99),
        )
        bandit.A = np.array(d["A"], dtype=np.float64)
        bandit.b = np.array(d["b"], dtype=np.float64)
        bandit.counts = np.array(d["counts"], dtype=np.int64)
        bandit.explore_counts = d.get("explore_counts", 0)
        bandit.total_updates = d["total_updates"]
        return bandit

    def save(self, path: str | Path) -> None:
        """Write state to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.state_dict(), f)

    @classmethod
    def load(cls, path: str | Path) -> LinUCB:
        """Load state from a JSON file."""
        with open(path) as f:
            return cls.from_state_dict(json.load(f))

    def __repr__(self) -> str:
        return (
            f"LinUCB(n_arms={self.n_arms}, dim={self.dim}, "
            f"alpha={self.alpha}, epsilon={self.epsilon:.4f}, "
            f"total_updates={self.total_updates}, explore_steps={self.explore_counts}, "
            f"counts={self.counts.tolist()})"
        )


# ---------------------------------------------------------------------------
# Task-type -> category mapping (single source of truth for bandit module)
# ---------------------------------------------------------------------------

TASK_CATEGORY_MAP: dict[str, str] = {
    "sort_list": "data_structures", "deduplicate": "data_structures",
    "flatten": "recursive", "partition": "data_structures",
    "reverse_string": "string", "is_palindrome": "string",
    "is_anagram": "string", "longest_unique": "string",
    "word_count": "string", "invert_dict": "data_structures",
    "running_sum": "iterative", "search_insert": "iterative",
    "merge_intervals": "data_structures", "valid_brackets": "data_structures",
    "has_cycle": "graph", "fibonacci": "iterative",
    "climb_stairs": "dynamic_programming", "unique_paths": "dynamic_programming",
    "lis": "dynamic_programming", "is_prime": "math",
    "gcd": "math", "permutations": "recursive",
    "factorial": "iterative", "digit_sum": "iterative",
    "power_set": "recursive", "generate_parens": "recursive",
    "nested_sum": "recursive", "max_subarray": "dynamic_programming",
    "coin_change": "dynamic_programming", "topological_sort": "graph",
    "count_components": "graph", "num_islands": "graph",
    "is_bipartite": "graph", "compress_string": "string",
    "roman_to_int": "parsing", "title_case": "parsing",
    "camel_to_snake": "parsing", "count_vowels": "parsing",
    "decode_run_length": "parsing", "lcm": "math",
    "single_number": "math", "is_perfect_square": "math",
}

# Sorted canonical category list (index = category slot in posteriors)
CATEGORIES: list[str] = sorted(set(TASK_CATEGORY_MAP.values()))
# ['data_structures', 'dynamic_programming', 'graph', 'iterative',
#  'math', 'parsing', 'recursive', 'string']


# ---------------------------------------------------------------------------
# Category-aware Thompson Sampling
# ---------------------------------------------------------------------------

class CategoryThompsonSampling:
    """Category-aware Thompson Sampling bandit for binary rewards.

    Maintains Beta(alpha[k,i], beta[k,i]) posteriors per (task category k, arm i).
    At selection time, samples from the current category's Beta posteriors and
    picks the arm with the highest sample.

    Addresses the oracle gap of LinUCB by:
      1. Maintaining separate uncertainty per category — no cross-category
         interference that caused LinUCB to lock onto a single tribe.
      2. Natural exploration via posterior sampling — no need for an epsilon
         schedule or explicit UCB alpha that depends on the context dimension.
      3. Binary reward mapping (reward > 0 -> success) — well-suited to
         the pass/fail nature of code synthesis tasks.

    Call set_category(task_type_or_category) before each select()/update() pair.

    Args:
        n_arms: Number of arms (tribes).
        dim: Context dimensionality (kept for API compatibility with LinUCB).
        alpha_prior: Beta prior pseudo-count (alpha = beta = alpha_prior).
            Higher = more initial exploration (default 1.0 = uniform prior).
    """

    def __init__(
        self,
        n_arms: int,
        dim: int,
        alpha_prior: float = 1.0,
        **kwargs,
    ) -> None:
        if n_arms <= 0:
            raise ValueError(f"n_arms must be positive, got {n_arms}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self.n_arms = n_arms
        self.dim = dim
        self.alpha = kwargs.get("alpha", 1.0)  # stored for repr / logging only

        n_cats = len(CATEGORIES)
        # Beta posterior parameters: shape (n_categories, n_arms)
        self._alpha_prior = alpha_prior
        self.alpha_k: NDArray[np.float64] = np.full(
            (n_cats, n_arms), alpha_prior, dtype=np.float64
        )
        self.beta_k: NDArray[np.float64] = np.full(
            (n_cats, n_arms), alpha_prior, dtype=np.float64
        )

        self._cat_idx: dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}
        self._current_cat_idx: int = 0  # set by set_category()

        self.counts: NDArray[np.int64] = np.zeros(n_arms, dtype=np.int64)
        self.explore_counts: int = 0   # always 0 — TS has no explicit epsilon
        self.total_updates: int = 0

    # ------------------------------------------------------------------
    # Category context (must be called before select / update)
    # ------------------------------------------------------------------

    def set_category(self, task_type_or_category: str) -> None:
        """Set current task context before calling select() or update().

        Accepts either a task_type key (e.g. 'fibonacci') or a category name
        (e.g. 'iterative'). Maps via TASK_CATEGORY_MAP if needed.
        Falls back to index 0 for unknown inputs.
        """
        cat = TASK_CATEGORY_MAP.get(task_type_or_category, task_type_or_category)
        self._current_cat_idx = self._cat_idx.get(cat, 0)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        context: NDArray[np.floating],
        mask: Optional[list[bool]] = None,
    ) -> int:
        """Sample from Beta posteriors for the current category and pick best arm.

        Args:
            context: Ignored (kept for API compatibility with LinUCB).
            mask: Boolean list of length n_arms. True = available.
        """
        k = self._current_cat_idx
        if mask is None:
            mask = [True] * self.n_arms
        if not any(mask):
            raise ValueError("All arms are masked — no tribe available")

        available = [i for i in range(self.n_arms) if mask[i]]
        samples = np.array([
            np.random.beta(self.alpha_k[k, i], self.beta_k[k, i])
            for i in available
        ])
        return available[int(np.argmax(samples))]

    def select_with_scores(
        self,
        context: NDArray[np.floating],
        mask: Optional[list[bool]] = None,
    ) -> tuple[int, NDArray[np.float64]]:
        """Select arm via Thompson sampling; return posterior means as scores.

        Returns:
            (selected_arm, scores) where scores[i] = posterior mean for arm i,
            or -inf if masked.
        """
        k = self._current_cat_idx
        if mask is None:
            mask = [True] * self.n_arms
        if not any(mask):
            mask = [True] * self.n_arms

        available = [i for i in range(self.n_arms) if mask[i]]
        # Posterior means as interpretable scores
        scores = np.full(self.n_arms, -np.inf, dtype=np.float64)
        for i in available:
            scores[i] = self.alpha_k[k, i] / (self.alpha_k[k, i] + self.beta_k[k, i])

        # Thompson sample
        samples = np.array([
            np.random.beta(self.alpha_k[k, i], self.beta_k[k, i])
            for i in available
        ])
        chosen = available[int(np.argmax(samples))]
        return chosen, scores

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        context: NDArray[np.floating],
        arm: int,
        reward: float,
    ) -> None:
        """Update Beta posterior based on binary outcome.

        Maps reward to binary: reward > 0 -> success (alpha++), else failure (beta++).

        Args:
            context: Ignored (kept for API compatibility with LinUCB).
            arm: Index of the arm that was pulled.
            reward: Observed reward (positive = PASS, negative = FAIL).
        """
        k = self._current_cat_idx
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"arm index {arm} out of range [0, {self.n_arms})")

        if reward > 0.0:
            self.alpha_k[k, arm] += 1.0
        else:
            self.beta_k[k, arm] += 1.0

        self.counts[arm] += 1
        self.total_updates += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_uncertainty(self, arm: int) -> float:
        """Return Beta posterior variance for current category and given arm."""
        k = self._current_cat_idx
        a, b = self.alpha_k[k, arm], self.beta_k[k, arm]
        return float((a * b) / ((a + b) ** 2 * (a + b + 1)))

    def get_theta(self, arm: int) -> NDArray[np.float64]:
        """Return per-category posterior means for an arm (shape: n_categories)."""
        return self.alpha_k[:, arm] / (self.alpha_k[:, arm] + self.beta_k[:, arm])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "type": "category_thompson",
            "n_arms": self.n_arms,
            "dim": self.dim,
            "alpha_prior": self._alpha_prior,
            "alpha_k": self.alpha_k.tolist(),
            "beta_k": self.beta_k.tolist(),
            "counts": self.counts.tolist(),
            "total_updates": self.total_updates,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "CategoryThompsonSampling":
        ts = cls(
            n_arms=d["n_arms"],
            dim=d["dim"],
            alpha_prior=d.get("alpha_prior", 1.0),
        )
        ts.alpha_k = np.array(d["alpha_k"], dtype=np.float64)
        ts.beta_k = np.array(d["beta_k"], dtype=np.float64)
        ts.counts = np.array(d["counts"], dtype=np.int64)
        ts.total_updates = d["total_updates"]
        return ts

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.state_dict(), f)

    @classmethod
    def load(cls, path: str | Path) -> "CategoryThompsonSampling":
        with open(path) as f:
            return cls.from_state_dict(json.load(f))

    def __repr__(self) -> str:
        k = self._current_cat_idx
        means = [
            f"{self.alpha_k[k,i]/(self.alpha_k[k,i]+self.beta_k[k,i]):.3f}"
            for i in range(self.n_arms)
        ]
        return (
            f"CategoryThompsonSampling(n_arms={self.n_arms}, "
            f"cat={CATEGORIES[k]}, posterior_means={means}, "
            f"total_updates={self.total_updates}, counts={self.counts.tolist()})"
        )
