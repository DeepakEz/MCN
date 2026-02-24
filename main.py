"""MCN-v0.1 Entry Point — Tribal Specialization Experiment.

Supports two modes:
  --mock : Mock LLM + subprocess sandbox (no GPU/Docker needed)
  --live : Real vLLM inference + Docker sandbox (requires vLLM + Docker)

Mock mode (original Phase 1-3 behavior):
    - Tribe 0 ("reliable"): failure_bias="memory"  -> fails on memory tasks
    - Tribe 1 ("fast"):     failure_bias="timeout"  -> fails on timeout tasks
    - Tribe 2 ("creative"): failure_bias="index"    -> fails on index tasks

Live mode (Phase 4 — real LLM):
    - Tribes use vLLM for code generation (OpenAI-compatible API)
    - Sandboxes use Docker containers for isolated execution
    - State persisted to Redis (if configured) or local files
    - Config from environment variables (see mcn/config.py)

Usage:
    python main.py --mock               # 50-task experiment, mock sandbox
    python main.py --mock -n 100        # 100 tasks for stronger convergence
    python main.py --mock -v            # with debug logging
    python main.py --live -n 50         # real vLLM + Docker sandbox
    python main.py --live --log-dir /results  # custom log directory
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import ray

from mcn.config import MCNConfig
from mcn.protocol import Task


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    fmt = "[%(asctime)s] %(levelname)-7s %(name)-20s %(message)s"
    logging.basicConfig(
        level=level, format=fmt, datefmt="%H:%M:%S", stream=sys.stdout,
    )
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("docker").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Task factory: 3 task types that trigger different failure biases
# ---------------------------------------------------------------------------

def make_memory_task() -> Task:
    """Task whose function name contains 'memory' -> triggers memory bias."""
    return Task(
        description="Sort a list, optimized for memory-constrained environments.",
        function_name="sort_memory_safe",
        input_signature="def sort_memory_safe(xs: list[int]) -> list[int]",
        timeout_seconds=10.0,
        unit_tests=(
            "def test_basic():\n"
            "    assert sort_memory_safe([3, 1, 2]) == [1, 2, 3]\n\n"
            "def test_empty():\n"
            "    assert sort_memory_safe([]) == []\n"
        ),
    )


def make_timeout_task() -> Task:
    """Task whose function name contains 'timeout' -> triggers timeout bias."""
    return Task(
        description="Sort a list with strict timeout constraints.",
        function_name="sort_timeout_proof",
        input_signature="def sort_timeout_proof(xs: list[int]) -> list[int]",
        timeout_seconds=3.0,   # short timeout to detect infinite loops quickly
        unit_tests=(
            "def test_basic():\n"
            "    assert sort_timeout_proof([3, 1, 2]) == [1, 2, 3]\n\n"
            "def test_empty():\n"
            "    assert sort_timeout_proof([]) == []\n"
        ),
    )


def make_index_task() -> Task:
    """Task whose function name contains 'index' -> triggers index bias."""
    return Task(
        description="Sort a list with careful index handling.",
        function_name="sort_index_safe",
        input_signature="def sort_index_safe(xs: list[int]) -> list[int]",
        timeout_seconds=10.0,
        unit_tests=(
            "def test_basic():\n"
            "    assert sort_index_safe([3, 1, 2]) == [1, 2, 3]\n\n"
            "def test_empty():\n"
            "    assert sort_index_safe([]) == []\n"
        ),
    )


TASK_FACTORIES = {
    "memory":  make_memory_task,
    "timeout": make_timeout_task,
    "index":   make_index_task,
}


def make_task_sequence(n: int) -> list[tuple[str, Task]]:
    """Generate a round-robin sequence of n tasks across all types."""
    types = list(TASK_FACTORIES.keys())
    seq = []
    for i in range(n):
        task_type = types[i % len(types)]
        task = TASK_FACTORIES[task_type]()
        seq.append((task_type, task))
    return seq


# ---------------------------------------------------------------------------
# Mock sandbox (for testing without Docker)
# ---------------------------------------------------------------------------

@ray.remote
class MockSandboxExecutor:
    """Local subprocess sandbox (no Docker). Development use only."""

    def __init__(self) -> None:
        pass

    def execute(
        self,
        code: str,
        test_source: str,
        timeout_seconds: float = 10.0,
        mem_limit: str = "256m",
    ):
        import subprocess
        import tempfile
        from pathlib import Path as P

        from mcn.sandbox import SandboxResult, _parse_pytest_output

        result = SandboxResult()
        start = time.monotonic()

        # Use ramdisk if configured
        tmpdir_base = MCNConfig.SANDBOX_TMPDIR or None
        with tempfile.TemporaryDirectory(prefix="mcn_mock_", dir=tmpdir_base) as tmpdir:
            code_path = P(tmpdir) / "solution.py"
            test_path = P(tmpdir) / "test_solution.py"

            code_path.write_text(code, encoding="utf-8")

            test_preamble = (
                "import sys, os\n"
                f"sys.path.insert(0, {tmpdir!r})\n"
                "from solution import *\n\n"
            )
            test_path.write_text(
                test_preamble + test_source, encoding="utf-8",
            )

            try:
                proc = subprocess.run(
                    [
                        sys.executable, "-m", "pytest",
                        str(test_path),
                        "-v", "--tb=short", "--no-header", "-q",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=tmpdir,
                )
                result.exit_code = proc.returncode
                result.stdout = proc.stdout
                result.stderr = proc.stderr

            except subprocess.TimeoutExpired:
                result.timed_out = True
                result.exit_code = -1
                result.stderr = f"TimeoutError: Exceeded {timeout_seconds}s"

            except Exception as e:
                result.exit_code = -1
                result.stderr = f"Mock sandbox error: {e}"

        result.elapsed_seconds = time.monotonic() - start

        parsed = _parse_pytest_output(result.stdout, result.stderr)
        result.tests_passed = parsed["tests_passed"]
        result.tests_failed = parsed["tests_failed"]
        result.tests_errored = parsed["tests_errored"]
        result.tests_total = parsed["tests_total"]
        result.exception_type = parsed["exception_type"]
        result.exception_message = parsed["exception_message"]
        result.failed_test_names = parsed["failed_test_names"]

        if result.timed_out:
            result.passed = False
            result.exception_type = result.exception_type or "TimeoutError"
        elif (
            result.exit_code == 0
            and result.tests_failed == 0
            and result.tests_errored == 0
        ):
            result.passed = True
        else:
            result.passed = False

        return result

    def run_deep_audit(
        self,
        code: str,
        test_source: str,
        timeout_seconds: float = 10.0,
        mem_limit: str = "256m",
    ):
        """Run mutation testing on passing code (subprocess-based)."""
        from mcn.sandbox import SandboxResult, _generate_mutants

        mutants = _generate_mutants(code)
        result = SandboxResult(passed=True, mutation_score=1.0)

        if not mutants:
            return result

        killed = 0
        for mutant_code in mutants:
            mutant_result = self.execute(
                code=mutant_code,
                test_source=test_source,
                timeout_seconds=timeout_seconds,
                mem_limit=mem_limit,
            )
            if not mutant_result.passed:
                killed += 1

        result.mutation_score = killed / len(mutants)
        return result

    def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results(
    results: list[dict],
    n_tribes: int = 3,
) -> None:
    """Print convergence analysis showing tribal specialization."""
    n = len(results)
    n_early = min(10, n)
    n_late = min(10, n)

    early = results[:n_early]
    late = results[-n_late:]

    print(f"\n{'=' * 70}")
    print(f"  SPECIALIZATION ANALYSIS ({n} total runs)")
    print(f"{'=' * 70}")

    # Overall stats
    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    print(f"\n  Overall pass rate: {n_pass}/{n} ({100*n_pass/max(n,1):.0f}%)")

    # Per-window routing distribution
    for window_name, window in [("EARLY (first 10)", early), ("LATE (last 10)", late)]:
        print(f"\n  {window_name}:")
        by_type = defaultdict(lambda: defaultdict(int))
        for r in window:
            task_type = r.get("task_type", "unknown")
            tribe = r["tribe_idx"]
            by_type[task_type][tribe] += 1

        for task_type in sorted(by_type):
            counts = by_type[task_type]
            total = sum(counts.values())
            dist = " ".join(
                f"T{i}={counts.get(i, 0)}"
                for i in range(n_tribes)
            )
            print(f"    {task_type:10s}: {dist}  (n={total})")

    # Avoidance check
    print(f"\n  AVOIDANCE CHECK (last {n_late} runs):")
    bias_map = {
        "memory": 0,    # tribe 0 fails memory
        "timeout": 1,   # tribe 1 fails timeout
        "index": 2,     # tribe 2 fails index
    }
    all_avoided = True
    for task_type, bad_tribe in bias_map.items():
        late_type = [r for r in late if r.get("task_type") == task_type]
        n_routed_bad = sum(1 for r in late_type if r["tribe_idx"] == bad_tribe)
        n_type = len(late_type)
        if n_type == 0:
            status = "SKIP (no tasks)"
        elif n_routed_bad == 0:
            status = "PASS (fully avoided)"
        elif n_routed_bad < n_type * 0.5:
            status = f"PARTIAL ({n_routed_bad}/{n_type} still routed to bad tribe)"
        else:
            status = f"FAIL ({n_routed_bad}/{n_type} routed to bad tribe)"
            all_avoided = False
        print(f"    {task_type:10s} -> tribe_{bad_tribe} should be avoided: {status}")

    if all_avoided:
        print(f"\n  >>> SPECIALIZATION LEARNED: Council routes around failures <<<")
    else:
        print(f"\n  >>> STILL LEARNING: Try more tasks (-n 100) for convergence <<<")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCN-v0.1 Tribal Specialization Experiment",
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--mock", action="store_true",
        help="Use mock LLM + subprocess sandbox (no GPU/Docker needed)",
    )
    mode_group.add_argument(
        "--live", action="store_true",
        help="Use live vLLM + Docker sandbox (requires vLLM service running)",
    )
    parser.add_argument(
        "-n", "--num-tasks", type=int, default=50,
        help="Number of tasks to run (default 50)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="",
        help="Directory for state and logs (default: from MCN_LOG_DIR or auto)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Debug logging",
    )
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("mcn.main")

    # Resolve log directory
    import tempfile
    if args.log_dir:
        log_dir = args.log_dir
    elif MCNConfig.LOG_DIR and MCNConfig.LOG_DIR != "/results":
        log_dir = MCNConfig.LOG_DIR
    else:
        log_dir = tempfile.mkdtemp(prefix="mcn_exp_")

    # --- Init Ray ---
    mode_str = "MOCK" if args.mock else "LIVE (vLLM)"
    print(f"\n{'=' * 70}")
    print(f"  MCN-v0.1 TRIBAL SPECIALIZATION EXPERIMENT [{mode_str}]")
    print(f"{'=' * 70}")
    ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)

    # --- Tribes ---
    from mcn.tribe import TribeActor

    if args.mock:
        # Mock mode: tribes with failure biases (original behavior)
        failure_biases = ["memory", "timeout", "index"]
        tribe_labels = ["reliable", "fast", "creative"]
        tribes = []
        for i in range(MCNConfig.NUM_TRIBES):
            bias = failure_biases[i] if i < len(failure_biases) else ""
            label = tribe_labels[i] if i < len(tribe_labels) else f"tribe_{i}"
            prompt = MCNConfig.TRIBE_PROMPTS[i] if i < len(MCNConfig.TRIBE_PROMPTS) else f"You are tribe {i}."
            tribes.append(
                TribeActor.remote(
                    tribe_id=f"tribe_{i}",
                    system_prompt=prompt,
                    use_mock=True,
                    failure_bias=bias,
                )
            )
            print(f"  Tribe {i} ({label}):  bias={bias or 'none'}  [MOCK]")
    else:
        # Live mode: real vLLM inference
        tribes = []
        for i in range(MCNConfig.NUM_TRIBES):
            prompt = MCNConfig.TRIBE_PROMPTS[i] if i < len(MCNConfig.TRIBE_PROMPTS) else f"You are tribe {i}."
            tribes.append(
                TribeActor.remote(
                    tribe_id=f"tribe_{i}",
                    system_prompt=prompt,
                    model=MCNConfig.VLLM_MODEL,
                    temperature=MCNConfig.TRIBE_TEMPERATURE,
                    max_tokens=MCNConfig.TRIBE_MAX_TOKENS,
                    use_mock=False,
                )
            )
            print(f"  Tribe {i}:  model={MCNConfig.VLLM_MODEL}  [LIVE]")

    # --- Overseer ---
    from mcn.overseer import OverseerActor
    overseer = OverseerActor.remote()

    # --- Sandboxes ---
    # Both mock and live mode use subprocess sandboxes (MockSandboxExecutor).
    # When running inside the mcn-runner Docker container, the container
    # itself provides isolation. Docker-in-Docker (SandboxExecutor) is only
    # needed for bare-metal live runs outside Docker.
    n_sandboxes = MCNConfig.NUM_SANDBOXES if args.live else 2
    sandboxes = [MockSandboxExecutor.remote() for _ in range(n_sandboxes)]
    print(f"  {n_sandboxes} subprocess sandboxes (pytest)")

    # --- Council ---
    from mcn.council import CouncilActor
    council = CouncilActor.remote(
        tribe_handles=tribes,
        overseer_handles=[overseer],
        sandbox_handles=sandboxes,
        alpha=MCNConfig.BANDIT_ALPHA,
        log_dir=log_dir,
    )
    print(f"  Council (LinUCB alpha={MCNConfig.BANDIT_ALPHA}, dim=18)")
    print(f"  Log dir: {log_dir}")
    if args.live:
        print(f"  vLLM:  {MCNConfig.VLLM_BASE_URL}")
        print(f"  Redis: {'enabled' if MCNConfig.USE_REDIS else 'disabled'}")
        if MCNConfig.SANDBOX_TMPDIR:
            print(f"  Ramdisk: {MCNConfig.SANDBOX_TMPDIR}")

    # --- Generate task sequence ---
    task_seq = make_task_sequence(args.num_tasks)
    print(f"\n  Running {args.num_tasks} tasks (round-robin: memory, timeout, index)")
    print(f"  {'='*66}")

    # --- Execute ---
    all_results: list[dict] = []
    t_start = time.time()

    for i, (task_type, task) in enumerate(task_seq):
        result = ray.get(council.run_task.remote(task))

        record = {
            "run": i + 1,
            "task_type": task_type,
            "tribe_idx": int(result.tribe_id.split("_")[1]),
            "tribe_id": result.tribe_id,
            "verdict": result.verdict.name,
            "reward": result.reward,
            "exception": result.failure_info.exception_type,
        }
        all_results.append(record)

        v = "+" if record["verdict"] == "PASS" else "-"
        print(
            f"  [{i+1:3d}/{args.num_tasks}] "
            f"{task_type:8s} -> T{record['tribe_idx']} "
            f"[{v}] r={record['reward']:+.3f}"
            f"{'  ' + record['exception'] if record['exception'] else ''}"
        )

    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time:.1f}s ({total_time/args.num_tasks:.2f}s/task)")

    # --- Analysis ---
    analyze_results(all_results, n_tribes=MCNConfig.NUM_TRIBES)

    # --- Check artifacts ---
    pkl_path = Path(log_dir) / "bandit.pkl"
    runs_path = Path(log_dir) / "runs.jsonl"

    print(f"\n  ARTIFACTS:")
    if pkl_path.exists():
        print(f"    bandit.pkl:  {pkl_path.stat().st_size:,} bytes")
    else:
        print(f"    bandit.pkl:  NOT FOUND (saves every 10 tasks)")
    if runs_path.exists():
        n_lines = sum(1 for _ in open(runs_path))
        print(f"    runs.jsonl:  {n_lines} entries, {runs_path.stat().st_size:,} bytes")

    # --- Final council stats ---
    stats = ray.get(council.get_stats.remote())
    routing = ray.get(council.get_routing_history.remote())
    print(f"\n  COUNCIL FINAL STATE:")
    print(f"    Pass rate:      {stats['pass_rate']:.1%}")
    print(f"    Routing totals: {routing}")
    print(f"    Encoder samples: {stats['encoder_samples']}")
    print(f"    State backend:  {stats.get('state_backend', 'local')}")

    ray.shutdown()
    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
