"""MCN-v0.1 MLflow Experiment Tracker.

Wraps MLflow client for structured experiment tracking with
graceful degradation — all methods are no-ops when disabled.

Tracks:
    - Per-task metrics: reward, pass/fail, tribe selection, timing, tokens
    - Per-experiment summary: pass rate, routing distribution, totals
    - Artifacts: runs.jsonl, bandit state snapshots, config dumps
    - Parameters: full config snapshot at experiment start

MLflow UI is accessible at http://localhost:5000 when running in Docker.

Usage:
    tracker = MCNTracker(
        tracking_uri="http://mlflow:5000",
        experiment_name="mcn-experiments",
        enabled=True,
    )
    tracker.start_run(
        run_name="live-36tasks",
        config={"model": "Qwen-7B", "n_tribes": 3},
    )

    # Per-task logging
    tracker.log_task_attempt(
        run_number=1, task_id="abc", tribe_idx=0,
        verdict="PASS", reward=0.95, elapsed_seconds=2.3,
    )

    # End of experiment
    tracker.log_summary({"pass_rate": 0.83, "total_tasks": 36})
    tracker.log_artifact("/results/runs.jsonl")
    tracker.end_run()
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MCNTracker:
    """MLflow experiment tracker with graceful degradation.

    When disabled (enabled=False or no tracking_uri), all methods
    are silent no-ops. This allows the same council code to work
    with or without MLflow.

    Attributes:
        enabled: Whether tracking is active.
    """

    def __init__(
        self,
        tracking_uri: str = "",
        experiment_name: str = "mcn-experiments",
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled and bool(tracking_uri)
        self._mlflow: Any = None
        self._run: Any = None

        if self._enabled:
            try:
                import mlflow
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
                self._mlflow = mlflow
                logger.info(
                    "MLflow tracking enabled: uri=%s, experiment=%s",
                    tracking_uri, experiment_name,
                )
            except Exception as e:
                logger.warning(
                    "MLflow initialization failed (%s), tracking disabled", e,
                )
                self._enabled = False

    @property
    def enabled(self) -> bool:
        """True if MLflow tracking is active."""
        return self._enabled

    def start_run(
        self,
        run_name: str = "",
        config: Optional[dict] = None,
    ) -> None:
        """Start a new MLflow run.

        Logs all config values as MLflow parameters.
        """
        if not self._enabled:
            return

        try:
            self._run = self._mlflow.start_run(run_name=run_name or None)

            # Log config as parameters
            if config:
                # MLflow params must be strings, max 250 chars
                params = {
                    k: str(v)[:250] for k, v in config.items()
                }
                self._mlflow.log_params(params)

            logger.info("MLflow run started: %s", run_name)
        except Exception as e:
            logger.warning("MLflow start_run failed: %s", e)

    def log_task_attempt(
        self,
        run_number: int,
        task_id: str,
        tribe_idx: int,
        verdict: str,
        reward: float,
        elapsed_seconds: float,
        tokens_used: int = 0,
        ucb_scores: Optional[list[float]] = None,
        mutation_score: float = -1.0,
        patches_used: Optional[list[str]] = None,
    ) -> None:
        """Log a single task attempt as a step metric.

        Uses run_number as the step index for time-series plotting.
        """
        if not self._enabled:
            return

        try:
            step = run_number

            # Core metrics
            self._mlflow.log_metrics(
                {
                    "reward": reward,
                    "passed": 1.0 if verdict == "PASS" else 0.0,
                    "tribe_idx": float(tribe_idx),
                    "elapsed_seconds": elapsed_seconds,
                    "tokens_used": float(tokens_used),
                },
                step=step,
            )

            # Per-tribe UCB scores (for routing analysis)
            if ucb_scores:
                for i, score in enumerate(ucb_scores):
                    if score is not None and score != float("-inf"):
                        self._mlflow.log_metric(
                            f"ucb_score_tribe_{i}", score, step=step,
                        )

            # Mutation score (deep audit)
            if mutation_score >= 0:
                self._mlflow.log_metric(
                    "mutation_score", mutation_score, step=step,
                )

            # Running pass rate
            # (computed from cumulative metrics, tracked as a metric)

        except Exception as e:
            logger.warning("MLflow log_task_attempt failed: %s", e)

    def log_summary(self, summary: dict) -> None:
        """Log experiment-level summary metrics.

        Called once at the end of an experiment.
        """
        if not self._enabled:
            return

        try:
            metrics = {}
            if "pass_rate" in summary:
                metrics["final_pass_rate"] = float(summary["pass_rate"])
            if "total_tasks" in summary:
                metrics["total_tasks"] = float(summary["total_tasks"])
            if "total_passes" in summary:
                metrics["total_passes"] = float(summary["total_passes"])
            if "total_tokens" in summary:
                metrics["total_tokens"] = float(summary["total_tokens"])
            if "total_time" in summary:
                metrics["total_time_seconds"] = float(summary["total_time"])

            if metrics:
                self._mlflow.log_metrics(metrics)
                logger.info("MLflow summary logged: %s", metrics)
        except Exception as e:
            logger.warning("MLflow log_summary failed: %s", e)

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str = "",
    ) -> None:
        """Log a file as an MLflow artifact.

        Args:
            local_path: Path to the file on disk.
            artifact_path: Optional subdirectory in MLflow artifacts.
        """
        if not self._enabled:
            return

        try:
            self._mlflow.log_artifact(local_path, artifact_path or None)
            logger.info("MLflow artifact logged: %s", local_path)
        except Exception as e:
            logger.warning("MLflow log_artifact failed: %s", e)

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Log a single metric (convenience method)."""
        if not self._enabled:
            return
        try:
            self._mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.warning("MLflow log_metric failed: %s", e)

    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self._enabled:
            return

        try:
            self._mlflow.end_run()
            logger.info("MLflow run ended")
        except Exception as e:
            logger.warning("MLflow end_run failed: %s", e)
        finally:
            self._run = None

    def __repr__(self) -> str:
        status = "enabled" if self._enabled else "disabled"
        return f"MCNTracker({status})"
