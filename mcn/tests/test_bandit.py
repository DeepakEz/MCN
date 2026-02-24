"""Unit tests for mcn.util.bandit — LinUCB contextual bandit."""

from __future__ import annotations

import numpy as np
import pytest

from mcn.util.bandit import LinUCB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bandit_3arm():
    return LinUCB(n_arms=3, dim=18, alpha=1.5)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _random_context(dim: int = 18, rng=None) -> np.ndarray:
    gen = rng or np.random.default_rng(0)
    v = gen.standard_normal(dim).astype(np.float64)
    return v / (np.linalg.norm(v) + 1e-10)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestLinUCBInit:
    def test_n_arms(self):
        b = LinUCB(n_arms=5, dim=10, alpha=1.0)
        assert b.n_arms == 5

    def test_dim(self):
        b = LinUCB(n_arms=2, dim=8, alpha=0.5)
        assert b.dim == 8

    def test_alpha(self):
        b = LinUCB(n_arms=3, dim=18, alpha=2.0)
        assert b.alpha == pytest.approx(2.0)

    def test_counts_start_zero(self, bandit_3arm):
        assert list(bandit_3arm.counts) == [0, 0, 0]

    def test_total_updates_start_zero(self, bandit_3arm):
        assert bandit_3arm.total_updates == 0


# ---------------------------------------------------------------------------
# select / select_with_scores
# ---------------------------------------------------------------------------

class TestLinUCBSelect:
    def test_select_returns_valid_arm(self, bandit_3arm, rng):
        ctx = _random_context(rng=rng)
        arm = bandit_3arm.select(ctx)
        assert 0 <= arm < 3

    def test_select_with_scores_length(self, bandit_3arm, rng):
        ctx = _random_context(rng=rng)
        arm, scores = bandit_3arm.select_with_scores(ctx)
        assert len(scores) == 3
        assert 0 <= arm < 3

    def test_select_with_mask_respects_mask(self, bandit_3arm, rng):
        ctx = _random_context(rng=rng)
        mask = [True, False, False]
        arm, _ = bandit_3arm.select_with_scores(ctx, mask=mask)
        assert arm == 0

    def test_select_all_masked_falls_back(self, bandit_3arm, rng):
        ctx = _random_context(rng=rng)
        # All False: should fall back to allowing all
        arm, _ = bandit_3arm.select_with_scores(ctx, mask=[False, False, False])
        assert 0 <= arm < 3

    def test_select_increments_count(self, bandit_3arm, rng):
        ctx = _random_context(rng=rng)
        arm = bandit_3arm.select(ctx)
        # After at least one update, the arm's count should reflect usage
        bandit_3arm.update(ctx, arm=arm, reward=1.0)
        assert bandit_3arm.counts[arm] >= 1

    def test_select_consistent_dtype(self, bandit_3arm, rng):
        ctx = _random_context(rng=rng).astype(np.float32)
        arm = bandit_3arm.select(ctx)
        assert isinstance(arm, int)


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

class TestLinUCBUpdate:
    def test_update_increments_total(self, bandit_3arm, rng):
        ctx = _random_context(rng=rng)
        bandit_3arm.update(ctx, arm=0, reward=1.0)
        assert bandit_3arm.total_updates == 1

    def test_update_multiple_arms(self, bandit_3arm, rng):
        ctx = _random_context(rng=rng)
        bandit_3arm.update(ctx, arm=0, reward=1.0)
        bandit_3arm.update(ctx, arm=1, reward=-0.5)
        bandit_3arm.update(ctx, arm=2, reward=0.5)
        assert bandit_3arm.total_updates == 3

    def test_get_theta_shape(self, bandit_3arm, rng):
        ctx = _random_context(rng=rng)
        bandit_3arm.update(ctx, arm=0, reward=1.0)
        theta = bandit_3arm.get_theta(0)
        assert theta.shape == (18,)

    def test_positive_reward_increases_arm_preference(self):
        """After many positive-reward updates, arm 0 should be preferred on that context."""
        b = LinUCB(n_arms=3, dim=2, alpha=0.1)  # low alpha = exploit
        ctx = np.array([1.0, 0.0], dtype=np.float64)
        for _ in range(50):
            b.update(ctx, arm=0, reward=1.0)
            b.update(ctx, arm=1, reward=-1.0)
            b.update(ctx, arm=2, reward=-1.0)
        arm = b.select(ctx)
        assert arm == 0


# ---------------------------------------------------------------------------
# Persistence (state_dict / from_state_dict)
# ---------------------------------------------------------------------------

class TestLinUCBPersistence:
    def test_roundtrip_empty(self, bandit_3arm):
        sd = bandit_3arm.state_dict()
        restored = LinUCB.from_state_dict(sd)
        assert restored.n_arms == bandit_3arm.n_arms
        assert restored.dim == bandit_3arm.dim
        assert restored.alpha == pytest.approx(bandit_3arm.alpha)

    def test_roundtrip_with_updates(self, rng):
        b = LinUCB(n_arms=3, dim=18, alpha=1.5)
        for _ in range(20):
            ctx = _random_context(rng=rng)
            arm = b.select(ctx)
            b.update(ctx, arm=arm, reward=float(rng.standard_normal()))
        sd = b.state_dict()
        restored = LinUCB.from_state_dict(sd)
        assert restored.total_updates == b.total_updates
        assert np.allclose(restored.counts, b.counts)

    def test_state_dict_type_field(self, bandit_3arm):
        sd = bandit_3arm.state_dict()
        assert sd.get("type") == "linucb"
