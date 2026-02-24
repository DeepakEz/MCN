"""MCN -- Phase 2E: Oracle Overseer + Noise Sweep Simulation.

Pure-Python simulation of MCN bandit routing under known-ground-truth conditions.

Replaces real LLM/test execution with a synthetic reward function:
    P_success(tribe, category) = CAPABILITY_MATRIX[tribe][category]

Noise injection: with probability epsilon, flip the observed reward (0 -> 1 or 1 -> 0).
This models noisy / stochastic test feedback.

Routers simulated:
    LinUCB      -- linear contextual bandit with UCB exploration
    GNN-Proxy   -- epsilon-greedy with per-(tribe,category) Q-values (annealing epsilon)

Noise sweep: epsilon in {0.00, 0.05, 0.10, 0.20, 0.30, 0.50}

Hypotheses
----------
H1: At epsilon=0.00%, both routers converge to near-oracle performance (<5pp gap).
H2: LinUCB pass rate drops faster under noise than GNN-Proxy.
H3: GNN-Proxy maintains higher pass rate at epsilon=20% and epsilon=30%.

Usage:
    python run_oracle_experiment.py
    python run_oracle_experiment.py --episodes 1000 --seeds 10 --seed 42
    python run_oracle_experiment.py --save-csv results/oracle_sweep.csv --save-json results/oracle_sweep.json

Output:
    - Console table of pass rate x noise level x router
    - Optional CSV:  one row per (router, noise_level) with aggregated metrics
    - Optional JSON: full per-seed results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Ground-truth tribal capability matrix
# ---------------------------------------------------------------------------

CATEGORIES: list[str] = [
    "iterative",
    "recursive",
    "dynamic_programming",
    "graph",
    "string",
    "data_structures",
    "parsing",
    "math",
]

N_CATS    = len(CATEGORIES)
CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}

# P[tribe_idx][category] = true success probability (ground truth, noiseless)
# Designed to give each tribe a distinct strength profile:
#   T0: strong at iterative, string, data_structures, math
#   T1: strong at parsing, string, data_structures  (slight edge on parsing)
#   T2: strong at dynamic_programming, recursive, graph
CAPABILITY_MATRIX: list[dict[str, float]] = [
    # T0
    {
        "iterative":           0.90,
        "recursive":           0.60,
        "dynamic_programming": 0.65,
        "graph":               0.35,
        "string":              0.95,
        "data_structures":     0.95,
        "parsing":             0.85,
        "math":                0.85,
    },
    # T1
    {
        "iterative":           0.85,
        "recursive":           0.70,
        "dynamic_programming": 0.75,
        "graph":               0.45,
        "string":              0.95,
        "data_structures":     0.93,
        "parsing":             0.88,
        "math":                0.82,
    },
    # T2
    {
        "iterative":           0.75,
        "recursive":           0.80,
        "dynamic_programming": 0.85,
        "graph":               0.50,
        "string":              0.90,
        "data_structures":     0.92,
        "parsing":             0.85,
        "math":                0.75,
    },
]

N_TRIBES     = len(CAPABILITY_MATRIX)
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50]


# ---------------------------------------------------------------------------
# Oracle metrics (noiseless ground truth)
# ---------------------------------------------------------------------------

def compute_oracle(category_dist: list[float] | None = None) -> dict[str, Any]:
    """Compute oracle pass rate and best tribe per category.

    Args:
        category_dist: Probability weights for each category (uniform if None).

    Returns:
        {
            "oracle_rate":  float,
            "best_tribe":   {cat: tribe_idx},
            "best_rate":    {cat: float},
        }
    """
    if category_dist is None:
        category_dist = [1.0 / N_CATS] * N_CATS

    best_tribe: dict[str, int]   = {}
    best_rate:  dict[str, float] = {}
    for cat in CATEGORIES:
        rates = [(CAPABILITY_MATRIX[ti][cat], ti) for ti in range(N_TRIBES)]
        br, bt = max(rates)
        best_tribe[cat] = bt
        best_rate[cat]  = br

    oracle_rate = sum(
        category_dist[CAT_INDEX[cat]] * best_rate[cat]
        for cat in CATEGORIES
    )
    return {
        "oracle_rate": oracle_rate,
        "best_tribe":  best_tribe,
        "best_rate":   best_rate,
    }


# ---------------------------------------------------------------------------
# Context vector generator (18-dim: 8 one-hot + 10 noise dims)
# ---------------------------------------------------------------------------

_N_DIMS = 18

def make_context(cat: str, rng: random.Random) -> list[float]:
    """18-dim context vector: first 8 dims = category one-hot, rest = Gaussian noise."""
    vec = [0.0] * _N_DIMS
    vec[CAT_INDEX[cat]] = 1.0
    for i in range(N_CATS, _N_DIMS):
        vec[i] = rng.gauss(0.0, 0.3)
    return vec


# ---------------------------------------------------------------------------
# Noisy reward sampler
# ---------------------------------------------------------------------------

def sample_reward(tribe: int, cat: str, epsilon: float, rng: random.Random) -> int:
    """Sample a (possibly noisy) binary reward for tribe on cat.

    With probability epsilon the true outcome is flipped (noise injection).
    Returns 1 (pass) or 0 (fail).
    """
    true_success = rng.random() < CAPABILITY_MATRIX[tribe][cat]
    if rng.random() < epsilon:
        return 1 - int(true_success)
    return int(true_success)


# ---------------------------------------------------------------------------
# Router: LinUCB (disjoint, one arm per tribe)
# ---------------------------------------------------------------------------

class LinUCBRouter:
    """Disjoint Linear Upper Confidence Bound bandit.

    Each arm maintains its own (A, b) model.  Context: 18-dim vector.
    Reference: Li et al. (2010), "A Contextual-Bandit Approach to
    Personalized News Article Recommendation".
    """

    def __init__(self, n_arms: int = N_TRIBES, n_dims: int = _N_DIMS, alpha: float = 1.0):
        self.alpha  = alpha
        self.n_arms = n_arms
        self.n_dims = n_dims
        # A[arm] = identity(n_dims); b[arm] = zeros(n_dims)
        self.A: list[list[list[float]]] = [
            [[float(i == j) for j in range(n_dims)] for i in range(n_dims)]
            for _ in range(n_arms)
        ]
        self.b: list[list[float]] = [[0.0] * n_dims for _ in range(n_arms)]

    # --- small linear algebra helpers (no numpy required) ---

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _matvec(M: list[list[float]], v: list[float]) -> list[float]:
        return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]

    @staticmethod
    def _solve(A: list[list[float]], b: list[float]) -> list[float]:
        """Solve A x = b via Gauss-Jordan elimination (in-place on copy)."""
        n = len(b)
        aug = [A[i][:] + [b[i]] for i in range(n)]
        for col in range(n):
            # Partial pivot
            max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[max_row] = aug[max_row], aug[col]
            pivot = aug[col][col]
            if abs(pivot) < 1e-12:
                continue
            aug[col] = [v / pivot for v in aug[col]]
            for row in range(n):
                if row != col:
                    f = aug[row][col]
                    aug[row] = [aug[row][j] - f * aug[col][j] for j in range(n + 1)]
        return [aug[i][n] for i in range(n)]

    @staticmethod
    def _rank1(A: list[list[float]], x: list[float]) -> list[list[float]]:
        """A += x @ x.T  (rank-1 update)."""
        n = len(x)
        return [[A[i][j] + x[i] * x[j] for j in range(n)] for i in range(n)]

    def select(self, ctx: list[float]) -> int:
        """Select arm with highest UCB score."""
        scores = []
        for arm in range(self.n_arms):
            theta  = self._solve(self.A[arm], self.b[arm])
            Ainv_x = self._solve(self.A[arm], ctx)
            exploit = self._dot(theta, ctx)
            explore = self.alpha * math.sqrt(max(0.0, self._dot(ctx, Ainv_x)))
            scores.append(exploit + explore)
        return max(range(self.n_arms), key=lambda i: scores[i])

    def update(self, arm: int, ctx: list[float], reward: float) -> None:
        """Update arm model with observed (context, reward) pair."""
        self.A[arm] = self._rank1(self.A[arm], ctx)
        n = len(ctx)
        self.b[arm] = [self.b[arm][i] + reward * ctx[i] for i in range(n)]


# ---------------------------------------------------------------------------
# Router: GNN-Proxy (epsilon-greedy with tabular Q-values per category)
# ---------------------------------------------------------------------------

class GNNProxyRouter:
    """GNN-Proxy: epsilon-greedy with per-(arm, category) Q-value estimates.

    Simulates a category-aware GNN router using a tabular Q-table.
    Epsilon decays linearly from eps_start -> eps_end over the run.
    Q-values are updated via exponential moving average (or 1/n for n <= 10).
    """

    def __init__(
        self,
        n_arms:    int        = N_TRIBES,
        categories: list[str] = CATEGORIES,
        eps_start: float      = 0.30,
        eps_end:   float      = 0.05,
        lr:        float      = 0.20,
    ):
        self.n_arms     = n_arms
        self.categories = categories
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.lr         = lr
        self._step      = 0
        self._max_steps = 1

        # Q[arm][cat] initialised to 0.5 (prior: coin flip)
        self.Q: list[dict[str, float]] = [
            {c: 0.5 for c in categories} for _ in range(n_arms)
        ]
        self._cnt: list[dict[str, int]] = [
            {c: 0 for c in categories} for _ in range(n_arms)
        ]

    def set_max_steps(self, n: int) -> None:
        self._max_steps = max(n, 1)

    def _epsilon(self) -> float:
        frac = min(self._step / self._max_steps, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select(self, cat: str, rng: random.Random) -> int:
        if rng.random() < self._epsilon():
            return rng.randint(0, self.n_arms - 1)
        return max(range(self.n_arms), key=lambda a: self.Q[a][cat])

    def update(self, arm: int, cat: str, reward: float) -> None:
        self._step += 1
        self._cnt[arm][cat] += 1
        cnt = self._cnt[arm][cat]
        lr  = 1.0 / cnt if cnt <= 10 else self.lr
        self.Q[arm][cat] += lr * (reward - self.Q[arm][cat])


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------

def run_simulation(
    router_cls: str,
    n_episodes: int,
    noise:      float,
    rng:        random.Random,
    category_dist: list[float] | None = None,
    **router_kwargs: Any,
) -> dict[str, Any]:
    """Run one simulation episode sweep.

    Args:
        router_cls:    "linucb" or "gnn"
        n_episodes:    Number of task episodes
        noise:         Reward flip probability epsilon
        rng:           Seeded random generator
        category_dist: Category sampling weights (uniform if None)
        **router_kwargs: Forwarded to router constructor

    Returns:
        {
            "router":              str,
            "noise":               float,
            "pass_rate":           float,   # overall observed pass rate
            "oracle_rate":         float,
            "gap":                 float,   # oracle_rate - pass_rate
            "gini":                float,   # Gini coefficient of routing distribution
            "entropy":             float,   # routing entropy (bits)
            "convergence_episode": int,     # episode index at 90% of final pass rate
            "tribe_counts":        list[int],
            "episodes":            list[dict],
        }
    """
    if category_dist is None:
        category_dist = [1.0 / N_CATS] * N_CATS

    oracle      = compute_oracle(category_dist)
    oracle_rate = oracle["oracle_rate"]

    # Instantiate router
    if router_cls == "linucb":
        router: LinUCBRouter | GNNProxyRouter = LinUCBRouter(
            n_arms=N_TRIBES, **router_kwargs
        )
    elif router_cls == "gnn":
        router = GNNProxyRouter(n_arms=N_TRIBES, **router_kwargs)
        router.set_max_steps(n_episodes)
    else:
        raise ValueError(f"Unknown router: {router_cls!r}")

    episodes: list[dict]       = []
    tribe_counts: list[int]    = [0] * N_TRIBES
    running_pass  = 0
    running_total = 0

    for ep in range(n_episodes):
        cat = rng.choices(CATEGORIES, weights=category_dist, k=1)[0]
        ctx = make_context(cat, rng)

        # Select tribe
        if router_cls == "linucb":
            tribe = router.select(ctx)
        else:
            tribe = router.select(cat, rng)

        # Observe (noisy) reward
        reward = sample_reward(tribe, cat, noise, rng)

        # Update router
        if router_cls == "linucb":
            router.update(tribe, ctx, float(reward))
        else:
            router.update(tribe, cat, float(reward))

        tribe_counts[tribe] += 1
        running_pass  += reward
        running_total += 1

        episodes.append({
            "ep":         ep,
            "cat":        cat,
            "tribe":      tribe,
            "reward":     reward,
            "cumul_rate": running_pass / running_total,
        })

    pass_rate = running_pass / max(running_total, 1)

    # Gini coefficient of routing distribution
    fracs = sorted(c / max(n_episodes, 1) for c in tribe_counts)
    n_t   = len(fracs)
    s     = sum(fracs) or 1
    gini  = (
        sum((2 * (i + 1) - n_t - 1) * fracs[i] for i in range(n_t)) / (n_t * s)
        if n_t > 1 else 0.0
    )

    # Routing entropy (bits)
    entropy = 0.0
    for c in tribe_counts:
        p = c / max(n_episodes, 1)
        if p > 0:
            entropy -= p * math.log2(p)

    # Convergence: first episode >= 90% of final pass rate (pessimistic: n_episodes if never)
    target        = 0.90 * pass_rate
    conv_episode  = n_episodes
    for ep_data in episodes:
        if ep_data["cumul_rate"] >= target:
            conv_episode = ep_data["ep"]
            break

    return {
        "router":              router_cls,
        "noise":               noise,
        "pass_rate":           pass_rate,
        "oracle_rate":         oracle_rate,
        "gap":                 oracle_rate - pass_rate,
        "gini":                gini,
        "entropy":             entropy,
        "convergence_episode": conv_episode,
        "tribe_counts":        tribe_counts,
        "episodes":            episodes,
    }


# ---------------------------------------------------------------------------
# Noise sweep
# ---------------------------------------------------------------------------

def run_noise_sweep(
    n_episodes:   int              = 500,
    n_seeds:      int              = 5,
    noise_levels: list[float]      | None = None,
    routers:      list[str]        | None = None,
    verbose:      bool             = True,
) -> list[dict]:
    """Run full noise sweep across routers and epsilon levels.

    Returns:
        List of aggregated result dicts, one per (router, noise_level).
        Each dict includes a "seed_results" key with per-seed raw results.
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS
    if routers is None:
        routers = ["linucb", "gnn"]

    oracle      = compute_oracle()
    oracle_rate = oracle["oracle_rate"]

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  PHASE 2E -- ORACLE OVERSEER NOISE SWEEP")
        print(f"{'=' * 70}")
        print(f"  Oracle rate (perfect routing): {100*oracle_rate:.1f}%")
        print(f"  Best tribe per category:")
        for cat in CATEGORIES:
            bt   = oracle["best_tribe"][cat]
            br   = oracle["best_rate"][cat]
            all_r = [CAPABILITY_MATRIX[t][cat] for t in range(N_TRIBES)]
            print(
                f"    {cat:22s}: T{bt} ({100*br:.0f}%)  "
                f"[T0={100*all_r[0]:.0f}%  T1={100*all_r[1]:.0f}%  T2={100*all_r[2]:.0f}%]"
            )
        print()
        print(f"  Episodes per run:  {n_episodes}")
        print(f"  Seeds:             {n_seeds}")
        print(f"  Noise levels (epsilon):  {[f'{e:.0%}' for e in noise_levels]}")
        print(f"  Routers:           {routers}")
        print()

    all_results: list[dict] = []

    for router in routers:
        for noise in noise_levels:
            seed_results = []
            for seed in range(n_seeds):
                rng = random.Random(seed * 1000 + int(noise * 1000))
                res = run_simulation(router, n_episodes, noise, rng)
                seed_results.append(res)

            # Aggregate across seeds
            n = n_seeds
            avg_pass = sum(r["pass_rate"]           for r in seed_results) / n
            avg_gap  = sum(r["gap"]                 for r in seed_results) / n
            avg_gini = sum(r["gini"]                for r in seed_results) / n
            avg_ent  = sum(r["entropy"]             for r in seed_results) / n
            avg_conv = sum(r["convergence_episode"] for r in seed_results) / n
            std_pass = math.sqrt(
                sum((r["pass_rate"] - avg_pass) ** 2 for r in seed_results)
                / max(n - 1, 1)
            )

            agg: dict[str, Any] = {
                "router":              router,
                "noise":               noise,
                "pass_rate_mean":      avg_pass,
                "pass_rate_std":       std_pass,
                "oracle_rate":         oracle_rate,
                "gap_mean":            avg_gap,
                "gini_mean":           avg_gini,
                "entropy_mean":        avg_ent,
                "convergence_ep_mean": avg_conv,
                "n_seeds":             n_seeds,
                "n_episodes":          n_episodes,
                "seed_results":        seed_results,
            }
            all_results.append(agg)

            if verbose:
                sign = "+" if avg_gap >= 0 else ""
                print(
                    f"  [{router.upper():7s}] eps={noise:.0%}  "
                    f"pass={100*avg_pass:5.1f}% +-{100*std_pass:.1f}  "
                    f"gap={sign}{100*avg_gap:5.1f}pp  "
                    f"gini={avg_gini:.3f}  "
                    f"H={avg_ent:.2f}bits  "
                    f"conv@ep{avg_conv:.0f}"
                )

    return all_results


# ---------------------------------------------------------------------------
# Hypothesis evaluation
# ---------------------------------------------------------------------------

def evaluate_hypotheses(results: list[dict]) -> None:
    """Print hypothesis test results.

    H1: At epsilon=0%, both routers converge to near-oracle performance (<5pp gap).
    H2: LinUCB pass rate drops more steeply under noise (epsilon=0->30%) than GNN-Proxy.
    H3: GNN-Proxy maintains higher pass rate at epsilon=20% and epsilon=30%.
    """
    print(f"\n{'=' * 70}")
    print(f"  HYPOTHESIS EVALUATION")
    print(f"{'=' * 70}")

    def get(router: str, noise: float, key: str) -> float:
        for r in results:
            if r["router"] == router and abs(r["noise"] - noise) < 1e-9:
                return float(r.get(key, float("nan")))
        return float("nan")

    oracle_rate = results[0]["oracle_rate"] if results else 0.0

    # H1: near-oracle at epsilon=0%
    gap_l0 = get("linucb", 0.00, "gap_mean")
    gap_g0 = get("gnn",    0.00, "gap_mean")
    h1_l   = gap_l0 < 0.05
    h1_g   = gap_g0 < 0.05
    print(f"\n  H1: At eps=0%, both routers reach near-oracle performance (<5pp gap)")
    print(f"    LinUCB gap = {100*gap_l0:+.1f}pp  [[v]]" if h1_l else f"    LinUCB gap = {100*gap_l0:+.1f}pp  [[x]]")
    print(f"    GNN gap    = {100*gap_g0:+.1f}pp  [[v]]" if h1_g else f"    GNN gap    = {100*gap_g0:+.1f}pp  [[x]]")
    print(f"    H1 verdict: {'SUPPORTED' if h1_l and h1_g else 'PARTIALLY SUPPORTED' if h1_l or h1_g else 'NOT SUPPORTED'}")

    # H2: LinUCB degrades faster (larger drop from epsilon=0 to epsilon=0.30)
    linucb_drop = get("linucb", 0.00, "pass_rate_mean") - get("linucb", 0.30, "pass_rate_mean")
    gnn_drop    = get("gnn",    0.00, "pass_rate_mean") - get("gnn",    0.30, "pass_rate_mean")
    h2 = linucb_drop > gnn_drop
    print(f"\n  H2: LinUCB degrades faster under noise (epsilon=0 -> 30%)")
    print(f"    LinUCB drop: {100*linucb_drop:+.1f}pp")
    print(f"    GNN drop:    {100*gnn_drop:+.1f}pp")
    print(f"    H2 verdict: {'SUPPORTED' if h2 else 'NOT SUPPORTED'}  "
          f"(LinUCB {'>' if h2 else '<='} GNN drop)")

    # H3: GNN maintains higher pass rate at epsilon=20% and epsilon=30%
    l20 = get("linucb", 0.20, "pass_rate_mean")
    g20 = get("gnn",    0.20, "pass_rate_mean")
    l30 = get("linucb", 0.30, "pass_rate_mean")
    g30 = get("gnn",    0.30, "pass_rate_mean")
    h3_20 = g20 > l20
    h3_30 = g30 > l30
    print(f"\n  H3: GNN maintains higher pass rate at moderate noise (epsilon=20%, epsilon=30%)")
    print(f"    At epsilon=20%: GNN={100*g20:.1f}%  LinUCB={100*l20:.1f}%  -> {'GNN [v]' if h3_20 else 'LinUCB [v]'}")
    print(f"    At epsilon=30%: GNN={100*g30:.1f}%  LinUCB={100*l30:.1f}%  -> {'GNN [v]' if h3_30 else 'LinUCB [v]'}")
    verdict3 = (
        "SUPPORTED"         if h3_20 and h3_30 else
        "PARTIALLY SUPPORTED" if h3_20 or h3_30 else
        "NOT SUPPORTED"
    )
    print(f"    H3 verdict: {verdict3}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results: list[dict]) -> None:
    """Print a compact epsilon x router pass-rate summary table."""
    print(f"\n{'=' * 70}")
    print(f"  NOISE SWEEP SUMMARY TABLE  (pass rate mean +- std)")
    print(f"{'=' * 70}")

    routers           = list(dict.fromkeys(r["router"] for r in results))
    noise_levels_seen = sorted(set(r["noise"] for r in results))
    oracle_rate       = results[0]["oracle_rate"] if results else 0.0

    print(f"\n  Oracle rate (perfect routing): {100*oracle_rate:.1f}%\n")

    col_w  = 16
    header = f"  {'epsilon':>6} | " + " | ".join(f"{rt.upper():^{col_w}}" for rt in routers)
    print(header)
    print(f"  {'-' * len(header)}")

    def get_cell(router: str, noise: float) -> str:
        for r in results:
            if r["router"] == router and abs(r["noise"] - noise) < 1e-9:
                mean = 100 * r["pass_rate_mean"]
                std  = 100 * r["pass_rate_std"]
                return f"{mean:5.1f}% +-{std:4.1f}"
        return "N/A"

    for noise in noise_levels_seen:
        row = f"  {noise:>5.0%} | " + " | ".join(
            f"{get_cell(rt, noise):^{col_w}}" for rt in routers
        )
        print(row)

    print(f"  {'-' * len(header)}")
    oracle_str = f"{100*oracle_rate:.1f}%"
    print(f"  {'Oracle':>6} | " + " | ".join(f"{oracle_str:^{col_w}}" for _ in routers))
    print()


# ---------------------------------------------------------------------------
# Category-level oracle breakdown table
# ---------------------------------------------------------------------------

def print_category_table() -> None:
    """Print per-category oracle breakdown."""
    oracle = compute_oracle()
    print(f"\n{'=' * 70}")
    print(f"  CATEGORY-LEVEL CAPABILITY MATRIX")
    print(f"{'=' * 70}")
    print(f"\n  {'Category':22s} | {'T0':>6} | {'T1':>6} | {'T2':>6} | Oracle T | Rate")
    print(f"  {'-'*65}")
    for cat in CATEGORIES:
        bt = oracle["best_tribe"][cat]
        br = oracle["best_rate"][cat]
        r  = [CAPABILITY_MATRIX[t][cat] for t in range(N_TRIBES)]
        print(
            f"  {cat:22s} | {100*r[0]:>5.0f}% | {100*r[1]:>5.0f}% | {100*r[2]:>5.0f}% | "
            f"T{bt:>6}  | {100*br:.0f}%"
        )
    print(f"\n  Overall oracle rate: {100*oracle['oracle_rate']:.1f}%\n")


# ---------------------------------------------------------------------------
# Output: CSV + JSON
# ---------------------------------------------------------------------------

def save_csv(results: list[dict], path: str) -> None:
    """Save aggregated sweep results as CSV."""
    fieldnames = [
        "router", "noise",
        "pass_rate_mean", "pass_rate_std",
        "oracle_rate",    "gap_mean",
        "gini_mean",      "entropy_mean",
        "convergence_ep_mean",
        "n_seeds",        "n_episodes",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  CSV saved to: {path}")


def save_json(results: list[dict], path: str) -> None:
    """Save sweep results as JSON (episode-level data excluded for size)."""
    slim = []
    for r in results:
        s = {k: v for k, v in r.items() if k not in ("seed_results",)}
        # Include per-seed summary statistics
        sr = r.get("seed_results", [])
        s["seed_pass_rates"]    = [round(x["pass_rate"], 4) for x in sr]
        s["seed_gaps"]          = [round(x["gap"],       4) for x in sr]
        s["seed_ginis"]         = [round(x["gini"],      4) for x in sr]
        s["seed_conv_episodes"] = [x["convergence_episode"] for x in sr]
        slim.append(s)
    with open(path, "w") as f:
        json.dump(slim, f, indent=2)
    print(f"  JSON saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCN Phase 2E: Oracle overseer noise sweep simulation",
    )
    parser.add_argument(
        "--episodes", type=int, default=500,
        help="Task episodes per simulation run (default: 500)",
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Random seeds to average over (default: 5)",
    )
    parser.add_argument(
        "--noise-levels", type=str, default="",
        help="Comma-separated epsilon values, e.g. '0,0.05,0.1,0.2,0.3,0.5'",
    )
    parser.add_argument(
        "--routers", type=str, default="linucb,gnn",
        help="Comma-separated router names: linucb, gnn (default: both)",
    )
    parser.add_argument(
        "--save-csv", type=str, default="",
        help="Path to save CSV summary (optional)",
    )
    parser.add_argument(
        "--save-json", type=str, default="",
        help="Path to save JSON results (optional)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-run console output",
    )
    args = parser.parse_args()

    noise_levels = NOISE_LEVELS
    if args.noise_levels:
        noise_levels = [float(x.strip()) for x in args.noise_levels.split(",")]

    routers = [r.strip() for r in args.routers.split(",")]
    for rt in routers:
        if rt not in ("linucb", "gnn"):
            print(f"ERROR: Unknown router {rt!r}. Choose from: linucb, gnn")
            sys.exit(1)

    print_category_table()

    results = run_noise_sweep(
        n_episodes=args.episodes,
        n_seeds=args.seeds,
        noise_levels=noise_levels,
        routers=routers,
        verbose=not args.quiet,
    )

    print_summary_table(results)
    evaluate_hypotheses(results)

    if args.save_csv:
        save_csv(results, args.save_csv)
    if args.save_json:
        save_json(results, args.save_json)

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
