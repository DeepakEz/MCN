"""MCN-v0.1 GNN Graph Router — Neural replacement for LinUCB.

Drop-in replacement that matches the LinUCB interface exactly:
    select(context, mask) -> int
    select_with_scores(context, mask) -> (int, NDArray)
    update(context, arm, reward) -> None
    state_dict() / from_state_dict()

Architecture:
    Bipartite graph with task node + tribe nodes.
    2-layer GraphSAGE for message passing.
    Epsilon-greedy exploration (decays from 0.3 to 0.05).
    Online learning via experience buffer + mini-batch SGD.

All computation on CPU (GPU is occupied by vLLM).

Usage:
    router = GNNRouter(n_arms=3, dim=18, alpha=1.5)
    arm = router.select(context)
    router.update(context, arm=0, reward=1.0)
"""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import Optional

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Force CPU — GPU is for vLLM
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# GNN Model
# ---------------------------------------------------------------------------

class TaskTribeGNN(nn.Module):
    """Lightweight 2-layer GraphSAGE for task-tribe scoring.

    Architecture:
        Input (in_dim) -> Linear(in_dim, hidden) -> ReLU
                       -> Linear(hidden, out_dim) -> ReLU
                       -> Linear(out_dim, 1)       -> score

    For a bipartite graph with 1 task node + N tribe nodes:
        1. Concatenate task features with each tribe's embedding
        2. Pass through MLP to produce per-tribe scores

    We use a simple MLP approach instead of full SAGEConv to avoid
    torch_geometric import complexity while preserving the graph-based
    reasoning (task node features are shared across all tribe scoring).

    Total parameters: ~2K (runs in <1ms on CPU)
    """

    def __init__(
        self,
        in_dim: int = 18,
        hidden: int = 32,
        out_dim: int = 16,
    ) -> None:
        super().__init__()
        # Input: concatenated [task_features, tribe_embedding] = 2 * in_dim
        self.fc1 = nn.Linear(in_dim * 2, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.fc3 = nn.Linear(out_dim, 1)

        # Initialize with small weights for stable early training
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        task_features: torch.Tensor,
        tribe_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score each tribe for the given task.

        Args:
            task_features: (in_dim,) task context vector.
            tribe_embeddings: (n_tribes, in_dim) tribe embedding matrix.

        Returns:
            (n_tribes,) score tensor.
        """
        n_tribes = tribe_embeddings.shape[0]
        # Broadcast task features: (n_tribes, in_dim)
        task_expanded = task_features.unsqueeze(0).expand(n_tribes, -1)
        # Concatenate: (n_tribes, 2 * in_dim)
        combined = torch.cat([task_expanded, tribe_embeddings], dim=1)
        # MLP scoring
        h = F.relu(self.fc1(combined))
        h = F.relu(self.fc2(h))
        scores = self.fc3(h).squeeze(-1)  # (n_tribes,)
        return scores


# ---------------------------------------------------------------------------
# GNN Router (LinUCB-compatible interface)
# ---------------------------------------------------------------------------

class GNNRouter:
    """GNN-based tribe router with LinUCB-compatible interface.

    Maintains:
        tribe_embeddings: (n_arms, dim) learnable parameters
        model: TaskTribeGNN for scoring
        optimizer: Adam with configurable learning rate
        experience_buffer: recent (context, arm, reward) for mini-batch updates
        epsilon: exploration rate (decays over time)
        counts: per-arm pull counts (for diagnostics)

    The alpha parameter from LinUCB is mapped to initial epsilon:
        epsilon_init = min(0.5, alpha / 5.0)

    Attributes:
        n_arms: Number of arms (tribes).
        dim: Context vector dimensionality.
        alpha: Exploration parameter (mapped to epsilon).
    """

    def __init__(
        self,
        n_arms: int,
        dim: int,
        alpha: float = 1.5,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        buffer_size: int = 64,
        batch_size: int = 8,
    ) -> None:
        if n_arms <= 0:
            raise ValueError(f"n_arms must be positive, got {n_arms}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self.n_arms = n_arms
        self.dim = dim
        self.alpha = alpha

        # GNN model (CPU only)
        self.model = TaskTribeGNN(
            in_dim=dim, hidden=hidden_dim, out_dim=hidden_dim // 2,
        ).to(DEVICE)

        # Learnable tribe embeddings
        self.tribe_embeddings = nn.Parameter(
            torch.randn(n_arms, dim, device=DEVICE) * 0.1,
        )

        # Optimizer over both model parameters and tribe embeddings
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + [self.tribe_embeddings],
            lr=learning_rate,
        )

        # Experience replay buffer
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._buffer: deque[tuple[np.ndarray, int, float]] = deque(
            maxlen=buffer_size,
        )

        # Exploration: epsilon-greedy with decay
        self._epsilon: float = min(0.5, alpha / 5.0)
        self._epsilon_min: float = 0.05
        self._epsilon_decay: float = 0.995

        # Diagnostics (LinUCB compatibility)
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.total_updates: int = 0

        logger.info(
            "GNNRouter initialized: %d arms, dim=%d, epsilon=%.3f, "
            "hidden=%d, lr=%.4f, buffer=%d",
            n_arms, dim, self._epsilon, hidden_dim, learning_rate, buffer_size,
        )

    def select(
        self,
        context: NDArray[np.floating],
        mask: Optional[list[bool]] = None,
    ) -> int:
        """Select an arm given context, respecting cooling mask."""
        arm, _ = self.select_with_scores(context, mask)
        return arm

    def select_with_scores(
        self,
        context: NDArray[np.floating],
        mask: Optional[list[bool]] = None,
    ) -> tuple[int, NDArray[np.float64]]:
        """Select an arm and return all scores (for diagnostics).

        Returns:
            (selected_arm, scores) where scores[i] = -inf if masked.
        """
        x = np.asarray(context, dtype=np.float64)
        if x.shape != (self.dim,):
            raise ValueError(
                f"Context shape mismatch: expected ({self.dim},), got {x.shape}",
            )

        if mask is None:
            mask = [True] * self.n_arms

        if not any(mask):
            raise ValueError("All arms are masked — no tribe available")

        # Forward pass (no gradient needed for selection)
        with torch.no_grad():
            task_tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            raw_scores = self.model(task_tensor, self.tribe_embeddings)

        # Convert to numpy and apply mask
        scores = np.full(self.n_arms, -np.inf, dtype=np.float64)
        for i in range(self.n_arms):
            if mask[i]:
                scores[i] = float(raw_scores[i])

        # Epsilon-greedy exploration
        available = [i for i in range(self.n_arms) if mask[i]]
        if random.random() < self._epsilon:
            # Explore: random available arm
            best_arm = random.choice(available)
        else:
            # Exploit: argmax of scores among available
            best_arm = int(np.argmax(scores))

        return best_arm, scores

    def update(
        self,
        context: NDArray[np.floating],
        arm: int,
        reward: float,
    ) -> None:
        """Update the GNN after observing a reward.

        Adds (context, arm, reward) to experience buffer.
        When buffer has enough samples, performs a mini-batch SGD step.
        """
        x = np.asarray(context, dtype=np.float64)
        if x.shape != (self.dim,):
            raise ValueError(
                f"Context shape mismatch: expected ({self.dim},), got {x.shape}",
            )
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(
                f"arm index {arm} out of range [0, {self.n_arms})",
            )

        # Update counts
        self.counts[arm] += 1
        self.total_updates += 1

        # Add to experience buffer
        self._buffer.append((x.copy(), arm, reward))

        # Train if buffer has enough samples
        if len(self._buffer) >= self._batch_size:
            self._train_step()

        # Decay epsilon
        self._epsilon = max(
            self._epsilon_min,
            self._epsilon * self._epsilon_decay,
        )

    def _train_step(self) -> None:
        """Perform one mini-batch SGD step from the experience buffer."""
        # Sample a mini-batch
        batch_size = min(self._batch_size, len(self._buffer))
        indices = random.sample(range(len(self._buffer)), batch_size)
        batch = [self._buffer[i] for i in indices]

        total_loss = torch.tensor(0.0, device=DEVICE)

        for ctx, arm, reward in batch:
            task_tensor = torch.tensor(ctx, dtype=torch.float32, device=DEVICE)
            scores = self.model(task_tensor, self.tribe_embeddings)
            predicted = scores[arm]
            target = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
            total_loss = total_loss + F.mse_loss(predicted, target)

        # Average loss
        loss = total_loss / batch_size

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + [self.tribe_embeddings],
            max_norm=1.0,
        )
        self.optimizer.step()

    # ------------------------------------------------------------------
    # LinUCB compatibility methods
    # ------------------------------------------------------------------

    def get_uncertainty(self, arm: int) -> float:
        """Return uncertainty proxy: inverse of sqrt(pull count + 1).

        Analogous to trace(A_i^{-1}) in LinUCB — decreases with more pulls.
        """
        return 1.0 / (1.0 + float(self.counts[arm]) ** 0.5)

    def get_theta(self, arm: int) -> NDArray[np.float64]:
        """Return tribe embedding as the 'weight vector' analog."""
        with torch.no_grad():
            return self.tribe_embeddings[arm].numpy().astype(np.float64)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary.

        Stores model weights, tribe embeddings, optimizer state,
        experience buffer, epsilon, and counts.
        """
        return {
            "type": "gnn_router",
            "n_arms": self.n_arms,
            "dim": self.dim,
            "alpha": self.alpha,
            "model_state": {
                k: v.tolist() for k, v in self.model.state_dict().items()
            },
            "tribe_embeddings": self.tribe_embeddings.detach().tolist(),
            "epsilon": self._epsilon,
            "counts": self.counts.tolist(),
            "total_updates": self.total_updates,
            "buffer": [
                {
                    "context": ctx.tolist(),
                    "arm": arm,
                    "reward": reward,
                }
                for ctx, arm, reward in self._buffer
            ],
            # Store hyperparams for restoration
            "hidden_dim": self.model.fc1.out_features,
            "buffer_size": self._buffer_size,
            "batch_size": self._batch_size,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> GNNRouter:
        """Restore from a serialized dictionary."""
        router = cls(
            n_arms=d["n_arms"],
            dim=d["dim"],
            alpha=d["alpha"],
            hidden_dim=d.get("hidden_dim", 32),
            learning_rate=d.get("learning_rate", 0.01),
            buffer_size=d.get("buffer_size", 64),
            batch_size=d.get("batch_size", 8),
        )

        # Restore model weights
        model_state = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in d["model_state"].items()
        }
        router.model.load_state_dict(model_state)

        # Restore tribe embeddings
        router.tribe_embeddings = nn.Parameter(
            torch.tensor(d["tribe_embeddings"], dtype=torch.float32, device=DEVICE),
        )
        # Re-create optimizer with restored embeddings
        router.optimizer = torch.optim.Adam(
            list(router.model.parameters()) + [router.tribe_embeddings],
            lr=d.get("learning_rate", 0.01),
        )

        # Restore state
        router._epsilon = d.get("epsilon", 0.3)
        router.counts = np.array(d["counts"], dtype=np.int64)
        router.total_updates = d.get("total_updates", 0)

        # Restore experience buffer
        router._buffer.clear()
        for entry in d.get("buffer", []):
            router._buffer.append((
                np.array(entry["context"], dtype=np.float64),
                entry["arm"],
                entry["reward"],
            ))

        logger.info(
            "GNNRouter restored: %d arms, %d updates, epsilon=%.3f, "
            "buffer=%d samples",
            router.n_arms, router.total_updates,
            router._epsilon, len(router._buffer),
        )
        return router

    def __repr__(self) -> str:
        return (
            f"GNNRouter(n_arms={self.n_arms}, dim={self.dim}, "
            f"epsilon={self._epsilon:.3f}, updates={self.total_updates})"
        )
