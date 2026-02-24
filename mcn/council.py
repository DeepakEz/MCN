"""MCN-v0.1 Council — Meta-Optimizer and Orchestrator.

Ray actor that implements the Council G from the formal spec:
    - Routes tasks to tribes via LinUCB contextual bandit
    - Encodes failure traces via FailureEncoder for context vectors
    - Runs code in sandboxed Docker containers
    - Computes rewards and updates the bandit
    - Runs deep audit (mutation testing) on 10% of passing tasks
    - Serves patches from PatchRegistry as few-shot hints
    - Logs all attempts to runs.jsonl
    - Persists bandit state + patch registry to bandit.pkl every 10 tasks
    - Manages sandbox pool with backpressure

Context vector layout (CONTEXT_DIM = 18):
    [0..3)    one-hot task-type keywords (memory, timeout, index)
    [3..13)   one-hot exception type from last failure (or zeros on first try)
    [13]      Z-scored log(runtime_ms)
    [14]      Z-scored peak_mem_mb
    [15]      test pass ratio
    [16]      test fail ratio
    [17]      Z-scored log(n_tests)

Reward function:
    R = 1.0 if PASS, -0.5 if FAIL
    R -= 0.05 * log(1 + runtime_seconds)    (log-scale runtime penalty)
    R += 2.0 * mutation_score               (when deep-audited)
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import ray

from mcn.config import MCNConfig
from mcn.memory import PatchRegistry
from mcn.protocol import (
    AttemptResult,
    CONTEXT_DIM,
    FailureCategory,
    FailureSignature,
    GateVerdict,
    OverseerDecision,
    Task,
    TestSuite,
    classify_exception,
)
from mcn.sandbox import SandboxResult
from mcn.state import MCNStateStore
from mcn.util.bandit import LinUCB
from mcn.util.failure_encoder import FailureEncoder
from mcn.util.normalization import RunningScaler

logger = logging.getLogger(__name__)


# CONTEXT_DIM, EXC_TO_CATEGORY, classify_exception imported from mcn.protocol


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def _compute_reward(passed: bool, elapsed_seconds: float) -> float:
    """Compute reward with log-scale runtime penalty.

    R = base - 0.05 * log(1 + runtime_seconds)

    Pass:  base = +1.0  -> reward ~ +0.95 to +0.85 depending on speed
    Fail:  base = -0.5  -> reward ~ -0.55 to -0.65
    """
    base = 1.0 if passed else -0.5
    runtime_penalty = 0.05 * math.log1p(max(elapsed_seconds, 0.0))
    return base - runtime_penalty


# ---------------------------------------------------------------------------
# JSONL logger (kept for backward compat; state store handles persistence)
# ---------------------------------------------------------------------------

def _clean_record(record: dict) -> dict:
    """Make a record JSON-serializable (NumPy arrays, enums)."""
    clean = {}
    for k, v in record.items():
        if isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, (GateVerdict, OverseerDecision, FailureCategory)):
            clean[k] = v.name
        else:
            clean[k] = v
    return clean


# ---------------------------------------------------------------------------
# Ray Actor
# ---------------------------------------------------------------------------

@ray.remote
class CouncilActor:
    """Central orchestrator: routes tasks, runs sandbox, learns from results.

    Implements the Council G as a meta-optimizer with:
      - LinUCB for tribe selection (slow learning, O(sqrt(T)) regret)
      - FailureEncoder for context vectors (exception one-hot + Z-scored metrics)
      - Sandbox pool with backpressure
      - Deterministic evidence gate
      - Pickle persistence for bandit state

    Args:
        tribe_handles: List of TribeActor Ray handles.
        overseer_handles: List of OverseerActor Ray handles.
        sandbox_handles: List of SandboxExecutor Ray handles.
        alpha: LinUCB exploration parameter.
        log_dir: Directory for runs.jsonl and bandit.pkl.
    """

    def __init__(
        self,
        tribe_handles: list,
        overseer_handles: list,
        sandbox_handles: list,
        alpha: float = 1.5,
        log_dir: str = ".",
    ) -> None:
        self.tribes = tribe_handles
        self.overseers = overseer_handles
        self.n_tribes = len(tribe_handles)

        # Sandbox pool (backpressure mechanism)
        self.sandbox_pool: list = list(sandbox_handles)
        self._sandbox_pool_size: int = len(sandbox_handles)

        # FailureEncoder: trace -> R^18 context vector
        self.failure_encoder = FailureEncoder(dim=CONTEXT_DIM)

        # Router: GNN Graph Router (Phase 5) or LinUCB (default)
        if MCNConfig.USE_GNN_ROUTER:
            from mcn.util.gnn_router import GNNRouter
            self.bandit = GNNRouter(
                n_arms=self.n_tribes,
                dim=CONTEXT_DIM,
                alpha=alpha,
                hidden_dim=MCNConfig.GNN_HIDDEN_DIM,
                learning_rate=MCNConfig.GNN_LR,
                buffer_size=MCNConfig.GNN_BUFFER_SIZE,
                batch_size=MCNConfig.GNN_BATCH_SIZE,
            )
            logger.info("Using GNN Graph Router (Phase 5)")
        else:
            self.bandit = LinUCB(
                n_arms=self.n_tribes,
                dim=CONTEXT_DIM,
                alpha=alpha,
                epsilon=MCNConfig.BANDIT_EPSILON,
                epsilon_min=MCNConfig.BANDIT_EPSILON_MIN,
                epsilon_decay=MCNConfig.BANDIT_EPSILON_DECAY,
            )
            logger.info(
                "Using LinUCB router (epsilon=%.3f -> %.3f, decay=%.3f)",
                MCNConfig.BANDIT_EPSILON,
                MCNConfig.BANDIT_EPSILON_MIN,
                MCNConfig.BANDIT_EPSILON_DECAY,
            )

        # Patch store: ChromaDB (Phase 5) or in-memory PatchRegistry (default)
        if MCNConfig.USE_CHROMADB:
            from mcn.chroma_registry import ChromaPatchRegistry
            self.patch_registry = ChromaPatchRegistry(
                dim=CONTEXT_DIM,
                chroma_url=MCNConfig.CHROMADB_URL,
                persist_dir=MCNConfig.CHROMADB_PERSIST_DIR,
                collection_name=MCNConfig.CHROMADB_COLLECTION,
                min_attempts=MCNConfig.PATCH_MIN_ATTEMPTS,
            )
            logger.info("Using ChromaDB patch registry (Phase 5)")
        else:
            self.patch_registry = PatchRegistry(
                dim=CONTEXT_DIM,
                min_attempts=MCNConfig.PATCH_MIN_ATTEMPTS,
            )
            logger.info(
                "Using in-memory PatchRegistry (min_attempts=%d)",
                MCNConfig.PATCH_MIN_ATTEMPTS,
            )

        # MLflow experiment tracking (Phase 5)
        from mcn.tracking import MCNTracker
        self.tracker = MCNTracker(
            tracking_uri=MCNConfig.MLFLOW_TRACKING_URI if MCNConfig.USE_MLFLOW else "",
            experiment_name=MCNConfig.MLFLOW_EXPERIMENT_NAME,
            enabled=MCNConfig.USE_MLFLOW,
        )

        # Cooling period tracking
        self._cooling_remaining: list[int] = [0] * self.n_tribes
        self._cooling_k: int = 3

        # Track last failure per task_id (for retry routing)
        # Maps task_id -> last failure context vector
        self._last_failure: dict[str, np.ndarray] = {}

        # Logging / persistence
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # State store: Redis (if configured) with local file fallback
        self.state_store = MCNStateStore(
            redis_url=MCNConfig.REDIS_URL if MCNConfig.USE_REDIS else "",
            log_dir=str(self.log_dir),
            use_redis=MCNConfig.USE_REDIS,
        )

        # Legacy paths (used by state store's fallback)
        self.runs_path = self.log_dir / "runs.jsonl"
        self.bandit_pkl_path = self.log_dir / "bandit.pkl"

        # Stats
        self._total_attempts: int = 0
        self._total_passes: int = 0
        self._total_failures: int = 0
        self._total_deep_audits: int = 0

        # Try restoring previous state
        self._try_restore_state()

        router_type = "GNN" if MCNConfig.USE_GNN_ROUTER else "LinUCB"
        patch_type = "ChromaDB" if MCNConfig.USE_CHROMADB else "in-memory"
        mlflow_status = "on" if MCNConfig.USE_MLFLOW else "off"
        logger.info(
            "Council initialized: %d tribes, %d overseers, %d sandboxes, "
            "context_dim=%d, state=%s, router=%s, patches=%s, mlflow=%s",
            self.n_tribes, len(self.overseers), self._sandbox_pool_size,
            CONTEXT_DIM, "redis" if self.state_store.has_redis else "local",
            router_type, patch_type, mlflow_status,
        )

    # ------------------------------------------------------------------
    # State restoration
    # ------------------------------------------------------------------

    def _try_restore_state(self) -> None:
        """Attempt to restore bandit/encoder/patches from previous run."""
        state = self.state_store.load_state()
        if state is None:
            logger.info("No previous state found, starting fresh")
            return

        try:
            if "bandit" in state and state["bandit"]:
                # Restore router: GNN or LinUCB based on config
                if MCNConfig.USE_GNN_ROUTER and state["bandit"].get("type") == "gnn_router":
                    from mcn.util.gnn_router import GNNRouter
                    self.bandit = GNNRouter.from_state_dict(state["bandit"])
                    logger.info("GNN Router restored (updates=%d)", self.bandit.total_updates)
                elif not MCNConfig.USE_GNN_ROUTER and state["bandit"].get("type") != "gnn_router":
                    self.bandit = LinUCB.from_state_dict(state["bandit"])
                    # Always apply the *current* config's epsilon schedule after
                    # restoration.  The saved state may have epsilon already
                    # decayed to an old floor value (e.g. 0.05 from Run 7) so
                    # simply restoring it would silently ignore any .env tuning
                    # done between runs — causing the "97/1/2 collapse" we saw
                    # when the user ran Run 8 without --fresh.
                    self.bandit.epsilon = MCNConfig.BANDIT_EPSILON
                    self.bandit.epsilon_min = MCNConfig.BANDIT_EPSILON_MIN
                    self.bandit.epsilon_decay = MCNConfig.BANDIT_EPSILON_DECAY
                    logger.info(
                        "LinUCB state restored (updates=%d), "
                        "epsilon reset to %.3f -> %.3f (decay=%.4f) from config",
                        self.bandit.total_updates,
                        MCNConfig.BANDIT_EPSILON,
                        MCNConfig.BANDIT_EPSILON_MIN,
                        MCNConfig.BANDIT_EPSILON_DECAY,
                    )
                else:
                    logger.warning(
                        "Router type mismatch (saved=%s, config=%s), starting fresh router",
                        state["bandit"].get("type", "linucb"),
                        "gnn" if MCNConfig.USE_GNN_ROUTER else "linucb",
                    )

            if "failure_encoder" in state and state["failure_encoder"]:
                self.failure_encoder = FailureEncoder.from_state_dict(
                    state["failure_encoder"],
                )
                logger.info("FailureEncoder restored")

            if "patch_registry" in state and state["patch_registry"]:
                pr_state = state["patch_registry"]
                if MCNConfig.USE_CHROMADB and pr_state.get("backend") == "chromadb":
                    from mcn.chroma_registry import ChromaPatchRegistry
                    self.patch_registry = ChromaPatchRegistry.from_state_dict(
                        pr_state,
                        chroma_url=MCNConfig.CHROMADB_URL,
                        persist_dir=MCNConfig.CHROMADB_PERSIST_DIR,
                        collection_name=MCNConfig.CHROMADB_COLLECTION,
                    )
                    logger.info(
                        "ChromaPatchRegistry restored (%d patches)",
                        self.patch_registry.n_patches,
                    )
                elif not MCNConfig.USE_CHROMADB and pr_state.get("backend") != "chromadb":
                    self.patch_registry = PatchRegistry.from_state_dict(pr_state)
                    logger.info(
                        "PatchRegistry restored (%d patches)",
                        self.patch_registry.n_patches,
                    )
                else:
                    logger.warning(
                        "Patch registry type mismatch, keeping fresh registry",
                    )

            if "stats" in state and state["stats"]:
                stats = state["stats"]
                self._total_attempts = int(stats.get("total_attempts", 0))
                self._total_passes = int(stats.get("total_passes", 0))
                self._total_failures = int(stats.get("total_failures", 0))
                self._total_deep_audits = int(stats.get("total_deep_audits", 0))
                logger.info(
                    "Stats restored: %d attempts, %.1f%% pass rate",
                    self._total_attempts,
                    100.0 * self._total_passes / max(self._total_attempts, 1),
                )

        except Exception as e:
            logger.warning("State restoration failed (%s), starting fresh", e)

    # ------------------------------------------------------------------
    # Sandbox pool management (backpressure)
    # ------------------------------------------------------------------

    def _acquire_sandbox(self) -> object:
        """Pop a sandbox handle from the pool, blocking if empty."""
        while not self.sandbox_pool:
            time.sleep(0.1)
        return self.sandbox_pool.pop()

    def _release_sandbox(self, sandbox) -> None:
        """Return a sandbox handle to the pool."""
        self.sandbox_pool.append(sandbox)

    # ------------------------------------------------------------------
    # Context building (FailureEncoder integration)
    # ------------------------------------------------------------------

    def _build_context(
        self,
        task: Task,
        prev_failure_ctx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build the LinUCB context vector.

        On first attempt for a task: use task-type features (from function name)
        so LinUCB can distinguish between task types even before any failure.
        On retry: use the FailureEncoder output from the previous failure
        (which already includes task-type features).

        If we have a cached failure for this task_id, use it automatically.
        """
        if prev_failure_ctx is not None:
            return prev_failure_ctx.astype(np.float64)

        # Check if there's a cached failure from a previous attempt
        cached = self._last_failure.get(task.task_id)
        if cached is not None:
            return cached.astype(np.float64)

        # First attempt: task-type features only (no failure info yet)
        return self.failure_encoder.encode_task_context(
            task.function_name,
        ).astype(np.float64)

    # ------------------------------------------------------------------
    # Core execution loop
    # ------------------------------------------------------------------

    def run_task(self, task: Task) -> AttemptResult:
        """Execute the full MCN pipeline for a single task.

        Steps:
          1. Build context vector (from last failure or task-type features)
          2. Select tribe via LinUCB (respecting cooling mask)
          2b. Search PatchRegistry for relevant hints
          3. PARALLEL: tribe.generate(hint=patch) + overseer.generate_suite()
          4. Acquire sandbox, run code + tests
          5. Encode failure trace via FailureEncoder
          5b. Deep audit: 10% of passing tasks get mutation tested
          6. Compute reward (with mutation bonus if audited)
          7. Log to runs.jsonl, persist bandit every 10 tasks
          8. Record attempt + register patch candidate
        """
        attempt_start = time.time()

        # --- Step 1: Build context ---
        context = self._build_context(task)

        # --- Step 2: Select tribe via LinUCB ---
        cooling_mask = [c == 0 for c in self._cooling_remaining]
        if not any(cooling_mask):
            cooling_mask = [True] * self.n_tribes

        tribe_idx, ucb_scores = self.bandit.select_with_scores(
            context, mask=cooling_mask,
        )
        tribe = self.tribes[tribe_idx]
        tribe_id = f"tribe_{tribe_idx}"

        logger.info(
            "Task %s -> Tribe %d (UCB: %s)",
            task.task_id, tribe_idx,
            [f"{s:.3f}" for s in ucb_scores],
        )

        # --- Step 2b: Search PatchRegistry for hints ---
        patches = self.patch_registry.search(context, top_k=1)
        patch_hint = patches[0].lesson if patches else ""
        patches_used = [p.patch_id for p in patches]

        # Suppress patch hint when a reference solution is already provided.
        # Sending both a PatchRegistry lesson *and* a reference_solution gives
        # the model two competing code hints and empirically degrades
        # performance (fibonacci: ablation=60% vs MCN=20% in Run 7).
        if task.reference_solution and patch_hint:
            logger.info(
                "Suppressing patch %s for task %s (reference_solution present)",
                patches_used[0], task.task_id,
            )
            patch_hint = ""
            patches_used = []

        if patch_hint:
            logger.info(
                "Serving patch %s as hint for task %s",
                patches_used[0], task.task_id,
            )

        # --- Step 3: PARALLEL generation + test suite ---
        gen_ref = tribe.generate.remote(
            task_description=task.description,
            function_name=task.function_name,
            input_signature=task.input_signature,
            unit_tests=task.unit_tests,
            hint=patch_hint,
            reference_solution=task.reference_solution,
        )

        overseer = self.overseers[0]
        suite_ref = overseer.generate_suite.remote(
            task_description=task.description,
            function_name=task.function_name,
            input_signature=task.input_signature,
            unit_tests=task.unit_tests,
        )

        gen_result, test_suite = ray.get([gen_ref, suite_ref])

        # Handle generation failure
        if gen_result.error:
            logger.error(
                "Tribe %d generation failed: %s", tribe_idx, gen_result.error,
            )
            result = AttemptResult(
                task_id=task.task_id,
                tribe_id=tribe_id,
                plan=gen_result.plan,
                code="",
                verdict=GateVerdict.ERROR,
                overseer_decision=OverseerDecision.ESCALATE,
                failure_info=FailureSignature(
                    category=FailureCategory.UNKNOWN,
                    exception_type="GenerationError",
                    exception_message=gen_result.error,
                ),
                reward=-0.5,
                sandbox_log=gen_result.error,
            )
            self._log_and_update(
                task, result, context, tribe_idx,
                patches_used=patches_used,
            )
            return result

        # --- Step 4: Run in sandbox ---
        combined_tests = test_suite.combined_source()

        sandbox = self._acquire_sandbox()
        try:
            sandbox_result: SandboxResult = ray.get(
                sandbox.execute.remote(
                    code=gen_result.code,
                    test_source=combined_tests,
                    timeout_seconds=task.timeout_seconds,
                    mem_limit=f"{task.memory_limit_mb}m",
                )
            )
        finally:
            self._release_sandbox(sandbox)

        # --- Step 5: Encode failure via FailureEncoder ---
        failure_ctx = self.failure_encoder.encode_from_sandbox(
            stdout=sandbox_result.stdout,
            stderr=sandbox_result.stderr,
            elapsed_seconds=sandbox_result.elapsed_seconds,
            tests_passed=sandbox_result.tests_passed,
            tests_failed=sandbox_result.tests_failed,
            tests_total=sandbox_result.tests_total,
            function_name=task.function_name,
        )

        # Cache failure context for retry routing
        if not sandbox_result.passed:
            self._last_failure[task.task_id] = failure_ctx
        else:
            # Clear cached failure on success
            self._last_failure.pop(task.task_id, None)

        # --- Step 5b: Deep audit (10% of passing tasks) ---
        mutation_score = -1.0  # sentinel: not audited
        if sandbox_result.passed and random.random() < 0.1:
            logger.info(
                "Deep audit triggered for task %s", task.task_id,
            )
            audit_sandbox = self._acquire_sandbox()
            try:
                audit_result: SandboxResult = ray.get(
                    audit_sandbox.run_deep_audit.remote(
                        code=gen_result.code,
                        test_source=combined_tests,
                        timeout_seconds=task.timeout_seconds,
                        mem_limit=f"{task.memory_limit_mb}m",
                    )
                )
                mutation_score = audit_result.mutation_score
                self._total_deep_audits += 1
                logger.info(
                    "Deep audit complete: mutation_score=%.3f", mutation_score,
                )
            except Exception as e:
                logger.warning("Deep audit failed: %s", e)
            finally:
                self._release_sandbox(audit_sandbox)

        # Build FailureSignature (structured, for logging / AttemptResult)
        if sandbox_result.passed:
            failure_info = FailureSignature(
                category=FailureCategory.NONE,
                tests_passed=sandbox_result.tests_passed,
                tests_failed=0,
                tests_errored=0,
                elapsed_seconds=sandbox_result.elapsed_seconds,
            )
            verdict = GateVerdict.PASS
            overseer_decision = OverseerDecision.ACCEPT
        else:
            exc_category = classify_exception(sandbox_result.exception_type)
            if sandbox_result.timed_out:
                exc_category = FailureCategory.TIMEOUT

            failure_info = FailureSignature(
                category=exc_category,
                exception_type=sandbox_result.exception_type,
                exception_message=sandbox_result.exception_message,
                tests_passed=sandbox_result.tests_passed,
                tests_failed=sandbox_result.tests_failed,
                tests_errored=sandbox_result.tests_errored,
                failed_test_names=tuple(sandbox_result.failed_test_names),
                elapsed_seconds=sandbox_result.elapsed_seconds,
            )
            verdict = (
                GateVerdict.TIMEOUT if sandbox_result.timed_out
                else GateVerdict.FAIL
            )
            if exc_category in (
                FailureCategory.UNKNOWN, FailureCategory.RUNTIME_OTHER,
            ):
                overseer_decision = OverseerDecision.ESCALATE
            else:
                overseer_decision = OverseerDecision.REVISE

        # --- Step 6: Compute reward ---
        reward = _compute_reward(
            passed=sandbox_result.passed,
            elapsed_seconds=sandbox_result.elapsed_seconds,
        )

        # Mutation bonus: reward robust code
        if mutation_score >= 0:
            reward += 2.0 * mutation_score

        # --- Build AttemptResult ---
        result = AttemptResult(
            task_id=task.task_id,
            tribe_id=tribe_id,
            plan=gen_result.plan,
            code=gen_result.code,
            self_tests="",
            verdict=verdict,
            overseer_decision=overseer_decision,
            failure_signature=failure_ctx,
            cluster_id=-1,
            failure_info=failure_info,
            elapsed_seconds=time.time() - attempt_start,
            generation_tokens=gen_result.tokens_used,
            reward=reward,
            sandbox_log=(
                sandbox_result.stdout + "\n" + sandbox_result.stderr
            ).strip(),
        )

        # --- Step 7: Update bandit + log ---
        self._log_and_update(
            task, result, context, tribe_idx,
            mutation_score=mutation_score,
            patches_used=patches_used,
        )

        # --- Step 8: Record attempt + register patch ---
        self.patch_registry.record_attempt(task.task_id)
        if sandbox_result.passed:
            patch = self.patch_registry.register_candidate(
                task_id=task.task_id,
                description=task.description,
                function_name=task.function_name,
                code=gen_result.code,
                context_vector=failure_ctx,
                passed=True,
                failure_category=sandbox_result.exception_type,
            )
            if patch:
                logger.info(
                    "Patch registered: %s for task %s (attempts=%d)",
                    patch.patch_id, task.task_id, patch.attempt_count,
                )

        return result

    # ------------------------------------------------------------------
    # Internal: bandit update + logging + persistence
    # ------------------------------------------------------------------

    def _log_and_update(
        self,
        task: Task,
        result: AttemptResult,
        context: np.ndarray,
        tribe_idx: int,
        mutation_score: float = -1.0,
        patches_used: Optional[list[str]] = None,
    ) -> None:
        """Update the bandit, tick cooling, log, and persist."""

        # Update LinUCB
        self.bandit.update(context, arm=tribe_idx, reward=result.reward)

        # Tick cooling timers
        for i in range(self.n_tribes):
            if self._cooling_remaining[i] > 0:
                self._cooling_remaining[i] -= 1

        # Update stats
        self._total_attempts += 1
        if result.verdict == GateVerdict.PASS:
            self._total_passes += 1
        else:
            self._total_failures += 1

        # Log to JSONL
        log_record = {
            "run_number": self._total_attempts,
            "task_id": task.task_id,
            "task_type": getattr(task, "function_name", "unknown"),
            "tribe_id": result.tribe_id,
            "tribe_idx": tribe_idx,
            "attempt_id": result.attempt_id,
            "timestamp": result.timestamp,
            "verdict": result.verdict,
            "overseer_decision": result.overseer_decision,
            "reward": round(result.reward, 4),
            "elapsed_seconds": round(result.elapsed_seconds, 4),
            "generation_tokens": result.generation_tokens,
            "failure_category": result.failure_info.category,
            "exception_type": result.failure_info.exception_type,
            "tests_passed": result.failure_info.tests_passed,
            "tests_failed": result.failure_info.tests_failed,
            "failure_signature": result.failure_signature,
            "mutation_score": (
                round(mutation_score, 4) if mutation_score >= 0 else None
            ),
            "patches_used": patches_used or [],
        }
        try:
            self.state_store.append_run_log(_clean_record(log_record))
        except Exception as e:
            logger.warning("Failed to write run log: %s", e)

        # Persist full state every 10 tasks
        if self._total_attempts % 10 == 0:
            self._save_state()

        # MLflow tracking (Phase 5)
        try:
            self.tracker.log_task_attempt(
                run_number=self._total_attempts,
                task_id=task.task_id,
                tribe_idx=tribe_idx,
                verdict=result.verdict.name,
                reward=result.reward,
                elapsed_seconds=result.elapsed_seconds,
                tokens_used=result.generation_tokens,
                mutation_score=mutation_score,
                patches_used=patches_used,
            )
        except Exception as e:
            logger.debug("MLflow log_task_attempt failed: %s", e)

        logger.info(
            "Attempt %d: task=%s tribe=%d verdict=%s reward=%.3f "
            "(pass_rate=%.1f%%)",
            self._total_attempts, task.task_id, tribe_idx,
            result.verdict.name, result.reward,
            100.0 * self._total_passes / max(self._total_attempts, 1),
        )

    def _save_state(self) -> None:
        """Persist all state via the state store (Redis or local fallback)."""
        stats = {
            "total_attempts": self._total_attempts,
            "total_passes": self._total_passes,
            "total_failures": self._total_failures,
            "total_deep_audits": self._total_deep_audits,
        }
        try:
            self.state_store.save_state(
                bandit=self.bandit,
                encoder=self.failure_encoder,
                patch_registry=self.patch_registry,
                stats=stats,
            )
        except Exception as e:
            logger.warning("Failed to save state: %s", e)

    # ------------------------------------------------------------------
    # Cooling period management
    # ------------------------------------------------------------------

    def trigger_cooling(self, tribe_idx: int) -> None:
        """Put a tribe into cooling after a patch application."""
        if 0 <= tribe_idx < self.n_tribes:
            self._cooling_remaining[tribe_idx] = self._cooling_k
            logger.info(
                "Tribe %d entering cooling for %d rounds",
                tribe_idx, self._cooling_k,
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return diagnostic statistics for the council."""
        return {
            "total_attempts": self._total_attempts,
            "total_passes": self._total_passes,
            "total_failures": self._total_failures,
            "total_deep_audits": self._total_deep_audits,
            "pass_rate": (
                self._total_passes / max(self._total_attempts, 1)
            ),
            "bandit_info": repr(self.bandit),
            "sandbox_pool_free": len(self.sandbox_pool),
            "sandbox_pool_total": self._sandbox_pool_size,
            "cooling_remaining": self._cooling_remaining[:],
            "encoder_samples": self.failure_encoder.metric_scaler.count,
            "patch_registry": repr(self.patch_registry),
            "patches_stored": self.patch_registry.n_patches,
            "tasks_tracked": len(self.patch_registry._attempt_counts),
            "state_backend": "redis" if self.state_store.has_redis else "local",
        }

    def get_bandit_weights(self) -> list[list[float]]:
        """Return the LinUCB weight vectors for each tribe."""
        return [
            self.bandit.get_theta(i).tolist()
            for i in range(self.n_tribes)
        ]

    def get_routing_history(self) -> dict:
        """Return per-tribe pull counts (for specialization analysis)."""
        return {
            f"tribe_{i}": int(self.bandit.counts[i])
            for i in range(self.n_tribes)
        }

    def save_state(self) -> None:
        """Persist full state to disk (or Redis)."""
        self._save_state()
        logger.info("Council state saved via %s", self.state_store)
