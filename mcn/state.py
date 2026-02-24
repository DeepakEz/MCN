"""MCN-v0.1 State Store — Redis Persistence with Local Fallback.

Provides a unified interface for persisting MCN state (bandit matrices,
failure encoder, patch registry, run logs) to either Redis or local files.

When Redis is available and MCN_USE_REDIS=true:
    - Bandit state  -> Redis key "mcn:bandit" (JSON)
    - Encoder state -> Redis key "mcn:encoder" (JSON)
    - Patch registry -> Redis key "mcn:patches" (JSON)
    - Stats         -> Redis hash "mcn:stats"
    - Run logs      -> Redis stream "mcn:runs" + local JSONL backup

When Redis is unavailable (graceful fallback):
    - Bandit + encoder + patches -> bandit.pkl (pickle)
    - Run logs -> runs.jsonl

Usage:
    from mcn.state import MCNStateStore
    store = MCNStateStore(redis_url="redis://localhost:6379/0", log_dir="/results")

    # Save state (every 10 tasks)
    store.save_state(bandit, encoder, patch_registry, stats)

    # Load state (on startup)
    state = store.load_state()

    # Append run log (every task)
    store.append_run_log({"task_id": "abc", "verdict": "PASS", ...})
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NumPy-aware JSON serialization
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy arrays and types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Handle enums (GateVerdict, FailureCategory, etc.)
        if hasattr(obj, "name"):
            return obj.name
        return super().default(obj)


def _json_dumps(obj: Any) -> str:
    """Serialize to JSON with NumPy and enum support."""
    return json.dumps(obj, cls=_NumpyEncoder)


# ---------------------------------------------------------------------------
# State Store
# ---------------------------------------------------------------------------

class MCNStateStore:
    """Persistent state backed by Redis (primary) or local files (fallback).

    Redis keys:
        mcn:bandit    — JSON string: LinUCB state_dict
        mcn:encoder   — JSON string: FailureEncoder state_dict
        mcn:patches   — JSON string: PatchRegistry state_dict
        mcn:stats     — Redis hash: total_attempts, total_passes, etc.
        mcn:runs      — Redis stream: per-task attempt records

    Local files (fallback):
        bandit.pkl    — Pickled state dict (bandit + encoder + patches + stats)
        runs.jsonl    — JSONL run log
    """

    # Redis key prefixes
    _KEY_BANDIT = "mcn:bandit"
    _KEY_ENCODER = "mcn:encoder"
    _KEY_PATCHES = "mcn:patches"
    _KEY_STATS = "mcn:stats"
    _KEY_RUNS = "mcn:runs"

    def __init__(
        self,
        redis_url: str = "",
        log_dir: str = ".",
        use_redis: bool = True,
    ) -> None:
        self._redis = None
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Attempt Redis connection
        if redis_url and use_redis:
            try:
                import redis as redis_lib
                self._redis = redis_lib.from_url(
                    redis_url, decode_responses=True,
                )
                self._redis.ping()
                logger.info("Redis connected: %s", redis_url)
            except Exception as e:
                logger.warning(
                    "Redis unavailable (%s), using local file fallback", e,
                )
                self._redis = None

    @property
    def has_redis(self) -> bool:
        """True if Redis is connected and responsive."""
        if self._redis is None:
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Save state
    # ------------------------------------------------------------------

    def save_state(
        self,
        bandit: Any,
        encoder: Any,
        patch_registry: Any,
        stats: dict,
    ) -> None:
        """Persist all MCN state.

        Args:
            bandit: LinUCB instance (has .state_dict())
            encoder: FailureEncoder instance (has .state_dict())
            patch_registry: PatchRegistry instance (has .state_dict())
            stats: Dict of council stats (total_attempts, passes, etc.)
        """
        bandit_state = bandit.state_dict()
        encoder_state = encoder.state_dict()
        patch_state = patch_registry.state_dict()

        if self.has_redis:
            try:
                pipe = self._redis.pipeline()
                pipe.set(self._KEY_BANDIT, _json_dumps(bandit_state))
                pipe.set(self._KEY_ENCODER, _json_dumps(encoder_state))
                pipe.set(self._KEY_PATCHES, _json_dumps(patch_state))
                # Store stats as hash
                for k, v in stats.items():
                    pipe.hset(self._KEY_STATS, k, str(v))
                pipe.execute()
                logger.info("State saved to Redis")
                return
            except Exception as e:
                logger.warning("Redis save failed (%s), falling back to pickle", e)

        # Local file fallback: pickle
        self._save_pickle(bandit_state, encoder_state, patch_state, stats)

    def _save_pickle(
        self,
        bandit_state: dict,
        encoder_state: dict,
        patch_state: dict,
        stats: dict,
    ) -> None:
        """Save state to local pickle file."""
        pkl_path = self._log_dir / "bandit.pkl"
        state = {
            "bandit": bandit_state,
            "failure_encoder": encoder_state,
            "patch_registry": patch_state,
            "stats": stats,
        }
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(
                "State saved to %s (%d bytes)",
                pkl_path, pkl_path.stat().st_size,
            )
        except Exception as e:
            logger.warning("Failed to save pickle: %s", e)

    # ------------------------------------------------------------------
    # Load state
    # ------------------------------------------------------------------

    def load_state(self) -> Optional[dict]:
        """Load persisted state (Redis first, then pickle fallback).

        Returns:
            Dict with keys: "bandit", "failure_encoder", "patch_registry", "stats"
            or None if no state is found.
        """
        if self.has_redis:
            try:
                bandit_json = self._redis.get(self._KEY_BANDIT)
                if bandit_json:
                    state = {
                        "bandit": json.loads(bandit_json),
                        "failure_encoder": json.loads(
                            self._redis.get(self._KEY_ENCODER) or "{}",
                        ),
                        "patch_registry": json.loads(
                            self._redis.get(self._KEY_PATCHES) or "{}",
                        ),
                        "stats": self._redis.hgetall(self._KEY_STATS),
                    }
                    logger.info("State restored from Redis")
                    return state
            except Exception as e:
                logger.warning("Redis load failed (%s), trying pickle", e)

        # Local file fallback: pickle
        return self._load_pickle()

    def _load_pickle(self) -> Optional[dict]:
        """Load state from local pickle file."""
        pkl_path = self._log_dir / "bandit.pkl"
        if not pkl_path.exists():
            return None
        try:
            with open(pkl_path, "rb") as f:
                state = pickle.load(f)
            logger.info("State restored from %s", pkl_path)
            return state
        except Exception as e:
            logger.warning("Failed to load pickle: %s", e)
            return None

    # ------------------------------------------------------------------
    # Run log
    # ------------------------------------------------------------------

    def append_run_log(self, record: dict) -> None:
        """Append a single run record to the log.

        Writes to Redis stream (if available) AND local JSONL (always).
        The JSONL file serves as a human-readable backup.
        """
        # Always write JSONL (backup / human-readable)
        self._append_jsonl(record)

        # Also write to Redis stream if available
        if self.has_redis:
            try:
                # Flatten record for Redis stream (all values must be strings)
                flat = {k: _json_dumps(v) for k, v in record.items()}
                self._redis.xadd(
                    self._KEY_RUNS, flat,
                    maxlen=10000,  # bounded stream
                )
            except Exception as e:
                logger.warning("Redis XADD failed: %s", e)

    def _append_jsonl(self, record: dict) -> None:
        """Append a JSON record to the JSONL log file."""
        jsonl_path = self._log_dir / "runs.jsonl"
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(_json_dumps(record) + "\n")
        except Exception as e:
            logger.warning("Failed to write runs.jsonl: %s", e)

    def get_run_history(self, count: int = 100) -> list[dict]:
        """Retrieve recent run records.

        Reads from Redis stream if available, otherwise from JSONL.
        """
        if self.has_redis:
            try:
                # XREVRANGE: newest first, limited by count
                entries = self._redis.xrevrange(
                    self._KEY_RUNS, count=count,
                )
                results = []
                for entry_id, fields in entries:
                    record = {
                        k: json.loads(v) for k, v in fields.items()
                    }
                    record["_stream_id"] = entry_id
                    results.append(record)
                return results
            except Exception as e:
                logger.warning("Redis XREVRANGE failed: %s", e)

        # JSONL fallback
        return self._read_jsonl_tail(count)

    def _read_jsonl_tail(self, count: int) -> list[dict]:
        """Read the last N records from the JSONL file."""
        jsonl_path = self._log_dir / "runs.jsonl"
        if not jsonl_path.exists():
            return []
        try:
            lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
            tail = lines[-count:] if count < len(lines) else lines
            return [json.loads(line) for line in tail if line.strip()]
        except Exception as e:
            logger.warning("Failed to read runs.jsonl: %s", e)
            return []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear_all(self) -> None:
        """Clear all stored state (use with caution)."""
        if self.has_redis:
            try:
                self._redis.delete(
                    self._KEY_BANDIT,
                    self._KEY_ENCODER,
                    self._KEY_PATCHES,
                    self._KEY_STATS,
                    self._KEY_RUNS,
                )
                logger.info("Redis state cleared")
            except Exception as e:
                logger.warning("Redis clear failed: %s", e)

    def get_stats_summary(self) -> dict:
        """Get a summary of stored state."""
        info = {
            "backend": "redis" if self.has_redis else "local",
            "log_dir": str(self._log_dir),
        }
        if self.has_redis:
            try:
                info["redis_stats"] = self._redis.hgetall(self._KEY_STATS)
                info["run_count"] = self._redis.xlen(self._KEY_RUNS)
            except Exception:
                pass
        return info

    def __repr__(self) -> str:
        backend = "redis" if self.has_redis else "local"
        return f"MCNStateStore(backend={backend}, log_dir={self._log_dir})"
