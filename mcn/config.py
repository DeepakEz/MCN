"""MCN-v0.1 Configuration — Environment-variable-driven config.

All MCN configuration is centralized here, read from environment variables
with sensible defaults. This replaces hardcoded values scattered across
the codebase and enables Docker Compose orchestration.

Environment variables use the MCN_ prefix to avoid collisions.

Usage:
    from mcn.config import MCNConfig

    # Read a config value (resolved at import time from env):
    url = MCNConfig.VLLM_BASE_URL

    # Override via environment:
    #   MCN_VLLM_URL=http://vllm:8000/v1 python main.py --live
"""

from __future__ import annotations

import os


class MCNConfig:
    """Static configuration namespace — all values resolved from env at import time."""

    # ------------------------------------------------------------------
    # vLLM / LLM inference
    # ------------------------------------------------------------------
    VLLM_BASE_URL: str = os.getenv("MCN_VLLM_URL", "http://localhost:8000/v1")
    VLLM_MODEL: str = os.getenv(
        "MCN_VLLM_MODEL", "deepseek-ai/deepseek-coder-6.7b-instruct",
    )
    VLLM_API_KEY: str = os.getenv("MCN_VLLM_API_KEY", "EMPTY")

    # ------------------------------------------------------------------
    # Redis state store
    # ------------------------------------------------------------------
    REDIS_URL: str = os.getenv("MCN_REDIS_URL", "redis://localhost:6379/0")
    USE_REDIS: bool = os.getenv("MCN_USE_REDIS", "false").lower() == "true"

    # ------------------------------------------------------------------
    # Tribes
    # ------------------------------------------------------------------
    NUM_TRIBES: int = int(os.getenv("MCN_NUM_TRIBES", "3"))
    TRIBE_TEMPERATURE: float = float(os.getenv("MCN_TRIBE_TEMPERATURE", "0.3"))
    TRIBE_MAX_TOKENS: int = int(os.getenv("MCN_TRIBE_MAX_TOKENS", "2048"))

    # Per-tribe temperatures (comma-separated, length must equal NUM_TRIBES).
    # When set, each tribe is initialised with a different sampling temperature,
    # creating genuine behavioural diversity that the router can exploit.
    # Example: MCN_TRIBE_TEMPERATURES=0.1,0.5,0.9
    # If unset (or wrong length), all tribes fall back to TRIBE_TEMPERATURE.
    TRIBE_TEMPERATURES: list[float] = [
        float(t.strip())
        for t in os.getenv("MCN_TRIBE_TEMPERATURES", "").split(",")
        if t.strip()
    ]

    # Tribe system prompts (default set for 3 tribes)
    # These must be explicit about output format for small models (3B-7B)
    _PROMPT_SUFFIX: str = (
        "\n\nCRITICAL RULES:\n"
        "1. Output ONLY a single ```python fenced code block.\n"
        "2. Do NOT include any explanation, commentary, or text outside the code block.\n"
        "3. The code block must define the requested function and nothing else.\n"
        "4. Do NOT include import statements unless absolutely necessary.\n"
        "5. Do NOT include test code or example usage.\n"
        "6. The function must handle edge cases (empty lists, None, etc).\n"
    )
    TRIBE_PROMPTS: list[str] = [
        "You are a reliable Python coder. Prioritize correctness and robustness." + _PROMPT_SUFFIX,
        "You are a fast Python coder. Write efficient, performant code." + _PROMPT_SUFFIX,
        "You are a creative Python coder. Find elegant, idiomatic solutions." + _PROMPT_SUFFIX,
    ]

    # ------------------------------------------------------------------
    # Sandbox
    # ------------------------------------------------------------------
    NUM_SANDBOXES: int = int(os.getenv("MCN_NUM_SANDBOXES", "4"))
    SANDBOX_IMAGE: str = os.getenv("MCN_SANDBOX_IMAGE", "mcn-sandbox:v0.1")
    SANDBOX_TMPDIR: str = os.getenv("MCN_SANDBOX_TMPDIR", "")
    # Empty string = system default tmpdir; "/dev/shm" = ramdisk

    SANDBOX_MEM_LIMIT: str = os.getenv("MCN_SANDBOX_MEM_LIMIT", "256m")
    SANDBOX_TIMEOUT: float = float(os.getenv("MCN_SANDBOX_TIMEOUT", "10.0"))

    # ------------------------------------------------------------------
    # Council / Bandit
    # ------------------------------------------------------------------
    BANDIT_ALPHA: float = float(os.getenv("MCN_BANDIT_ALPHA", "1.5"))
    DEEP_AUDIT_RATE: float = float(os.getenv("MCN_DEEP_AUDIT_RATE", "0.1"))
    # Minimum attempts before a passing solution is stored as a patch.
    # Default 1 = store every passing solution; 2 = only hard-won solutions.
    PATCH_MIN_ATTEMPTS: int = int(os.getenv("MCN_PATCH_MIN_ATTEMPTS", "1"))

    # Epsilon-greedy warm-up for LinUCB (prevents cold-start routing collapse).
    # At t=0 all UCB scores are identical (theta=0, A=I), so hard argmax always
    # picks arm 0, causing degenerate 100/0/0 routing.  Epsilon > 0 forces
    # random exploration during warm-up; epsilon decays to epsilon_min over time.
    # Set MCN_BANDIT_EPSILON=0 to disable (pure UCB, reproduces collapse).
    BANDIT_EPSILON: float = float(os.getenv("MCN_BANDIT_EPSILON", "0.3"))
    BANDIT_EPSILON_MIN: float = float(os.getenv("MCN_BANDIT_EPSILON_MIN", "0.05"))
    BANDIT_EPSILON_DECAY: float = float(os.getenv("MCN_BANDIT_EPSILON_DECAY", "0.99"))

    # ------------------------------------------------------------------
    # Persistence / logging
    # ------------------------------------------------------------------
    LOG_DIR: str = os.getenv("MCN_LOG_DIR", "/results")

    # ------------------------------------------------------------------
    # Ray
    # ------------------------------------------------------------------
    RAY_NUM_CPUS: int = int(os.getenv("MCN_RAY_NUM_CPUS", "0"))
    # 0 = auto-detect available CPUs

    # ------------------------------------------------------------------
    # Phase 5: GNN Graph Router
    # ------------------------------------------------------------------
    USE_GNN_ROUTER: bool = os.getenv("MCN_USE_GNN_ROUTER", "false").lower() == "true"

    # ------------------------------------------------------------------
    # Category-aware Thompson Sampling router
    # Addresses oracle gap by maintaining per-(category, arm) Beta posteriors.
    # Mutually exclusive with USE_GNN_ROUTER; takes priority if both set.
    # Set MCN_USE_THOMPSON_SAMPLING=true to enable.
    # ------------------------------------------------------------------
    USE_THOMPSON_SAMPLING: bool = (
        os.getenv("MCN_USE_THOMPSON_SAMPLING", "false").lower() == "true"
    )
    TS_ALPHA_PRIOR: float = float(os.getenv("MCN_TS_ALPHA_PRIOR", "1.0"))
    GNN_HIDDEN_DIM: int = int(os.getenv("MCN_GNN_HIDDEN_DIM", "32"))
    GNN_LR: float = float(os.getenv("MCN_GNN_LR", "0.01"))
    GNN_BUFFER_SIZE: int = int(os.getenv("MCN_GNN_BUFFER_SIZE", "64"))
    GNN_BATCH_SIZE: int = int(os.getenv("MCN_GNN_BATCH_SIZE", "8"))

    # ------------------------------------------------------------------
    # Phase 5: ChromaDB vector store
    # ------------------------------------------------------------------
    USE_CHROMADB: bool = os.getenv("MCN_USE_CHROMADB", "false").lower() == "true"
    CHROMADB_URL: str = os.getenv("MCN_CHROMADB_URL", "")
    # Empty = local persistent client; "http://chromadb:8000" = Docker service
    CHROMADB_PERSIST_DIR: str = os.getenv("MCN_CHROMADB_PERSIST_DIR", "/results/chroma")
    CHROMADB_COLLECTION: str = os.getenv("MCN_CHROMADB_COLLECTION", "mcn_patches")

    # ------------------------------------------------------------------
    # Phase 5: MLflow experiment tracking
    # ------------------------------------------------------------------
    USE_MLFLOW: bool = os.getenv("MCN_USE_MLFLOW", "false").lower() == "true"
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MCN_MLFLOW_TRACKING_URI", "http://localhost:5000",
    )
    MLFLOW_EXPERIMENT_NAME: str = os.getenv(
        "MCN_MLFLOW_EXPERIMENT_NAME", "mcn-experiments",
    )
