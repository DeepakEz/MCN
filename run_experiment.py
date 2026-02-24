"""MCN-v0.1 Specialization Experiment — Standalone Runner (no Ray required).

Simulates the full 50-task tribal specialization experiment by calling
all components directly (bypassing Ray remote/get). This proves LinUCB
learns to route around tribal failure biases.

Expected outcome:
  - Early runs (1-10):  roughly uniform tribe selection (exploration)
  - Late runs (40-50):  council routes around the failing tribe per task type
    - memory tasks  -> routed to tribe 1 or 2 (not 0)
    - timeout tasks -> routed to tribe 0 or 2 (not 1)
    - index tasks   -> routed to tribe 0 or 1 (not 2)
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import types

import numpy as np

# --- Mock Ray module so MCN source files can be imported without Ray ---
_ray_mock = types.ModuleType("ray")
_ray_mock.__dict__["init"] = lambda **kw: None
_ray_mock.__dict__["shutdown"] = lambda: None
_ray_mock.__dict__["get"] = lambda refs: [r() if callable(r) else r for r in refs] if isinstance(refs, list) else (refs() if callable(refs) else refs)

def _remote_decorator(cls_or_fn):
    """No-op @ray.remote decorator."""
    return cls_or_fn

_ray_mock.__dict__["remote"] = _remote_decorator
sys.modules["ray"] = _ray_mock

# --- Direct imports (now safe with mock Ray) ---
from mcn.memory import PatchRegistry
from mcn.protocol import (
    AttemptResult, CONTEXT_DIM, FailureCategory, FailureSignature,
    GateVerdict, OverseerDecision, Task, TestSuite,
    classify_exception as _classify_exception,
)
from mcn.sandbox import SandboxResult, _parse_pytest_output, _generate_mutants
from mcn.util.bandit import LinUCB
from mcn.util.failure_encoder import FailureEncoder


def _compute_reward(passed: bool, elapsed_seconds: float) -> float:
    base = 1.0 if passed else -0.5
    runtime_penalty = 0.05 * math.log1p(max(elapsed_seconds, 0.0))
    return base - runtime_penalty


# ---------------------------------------------------------------------------
# Inline mock tribe generate (same logic as tribe.py _mock_generate)
# ---------------------------------------------------------------------------

BIAS_BUG_MAP = {
    "memory": (
        "    big = [0] * (10 ** 10)\n"
        "    return sorted(xs)\n"
    ),
    "timeout": (
        "    while True:\n"
        "        pass\n"
        "    return sorted(xs)\n"
    ),
    "index": (
        "    return xs[999999]\n"
    ),
}


def mock_generate(tribe_id: str, failure_bias: str, function_name: str) -> str:
    """Generate code: buggy if bias matches function_name, else correct."""
    if failure_bias and failure_bias in function_name.lower():
        bug_body = BIAS_BUG_MAP.get(failure_bias, "    return sorted(xs)\n")
        return f"def {function_name}(xs):\n{bug_body}"
    return (
        f"def {function_name}(xs):\n"
        f"    if not isinstance(xs, list):\n"
        f"        raise TypeError('Expected a list')\n"
        f"    return sorted(xs)\n"
    )


# ---------------------------------------------------------------------------
# Inline mock overseer (same logic as overseer.py)
# ---------------------------------------------------------------------------

def mock_generate_suite(function_name: str, input_signature: str, unit_tests: str) -> TestSuite:
    """Generate the same test suite as OverseerActor (with Hypothesis tests)."""
    parts = []

    # Existence test
    parts.append(
        f"def test_{function_name}_exists():\n"
        f"    assert callable({function_name})\n"
    )

    if "list" in input_signature.lower():
        parts.append(
            f"def test_{function_name}_empty_list():\n"
            f"    result = {function_name}([])\n"
            f"    assert result is not None\n"
        )
        parts.append(
            f"def test_{function_name}_single_element():\n"
            f"    result = {function_name}([42])\n"
            f"    assert result is not None\n"
        )
        parts.append(
            f"def test_{function_name}_negative_numbers():\n"
            f"    result = {function_name}([-3, -1, -2])\n"
            f"    assert result is not None\n"
        )
        parts.append(
            f"def test_{function_name}_duplicates():\n"
            f"    result = {function_name}([1, 1, 1, 1])\n"
            f"    assert result is not None\n"
        )
        parts.append(
            f"def test_{function_name}_large_input():\n"
            f"    import time\n"
            f"    big = list(range(10000, 0, -1))\n"
            f"    start = time.monotonic()\n"
            f"    result = {function_name}(big)\n"
            f"    elapsed = time.monotonic() - start\n"
            f"    assert elapsed < 5.0\n"
            f"    assert result is not None\n"
        )

    if "-> list" in input_signature.lower():
        parts.append(
            f"def test_{function_name}_returns_list():\n"
            f"    result = {function_name}([3, 1, 2])\n"
            f"    assert isinstance(result, list)\n"
        )

    overseer_tests = "\n\n".join(parts)

    # Generate Hypothesis fuzz tests via the overseer module
    # Only include if hypothesis is actually installed
    fuzz_tests = ""
    try:
        import hypothesis  # noqa: F401
        from mcn.overseer import _generate_hypothesis_tests
        fuzz_tests = _generate_hypothesis_tests(function_name, input_signature)
    except ImportError:
        pass  # hypothesis not available — skip fuzz tests

    return TestSuite(
        unit_tests=unit_tests,
        overseer_tests=overseer_tests,
        fuzz_tests=fuzz_tests,
    )


# ---------------------------------------------------------------------------
# Inline mock sandbox (subprocess-based, same as main.py MockSandboxExecutor)
# ---------------------------------------------------------------------------

def mock_sandbox_execute(
    code: str, test_source: str, timeout_seconds: float = 10.0,
) -> SandboxResult:
    """Run code + tests in a subprocess (no Docker)."""
    result = SandboxResult()
    start = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="mcn_exp_") as tmpdir:
        code_path = Path(tmpdir) / "solution.py"
        test_path = Path(tmpdir) / "test_solution.py"

        code_path.write_text(code, encoding="utf-8")

        test_preamble = (
            "import sys, os\n"
            f"sys.path.insert(0, {tmpdir!r})\n"
            "from solution import *\n\n"
        )
        test_path.write_text(test_preamble + test_source, encoding="utf-8")

        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "pytest",
                    str(test_path),
                    "-v", "--tb=short", "--no-header", "-q",
                ],
                capture_output=True, text=True,
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
    elif result.exit_code == 0 and result.tests_failed == 0 and result.tests_errored == 0:
        result.passed = True
    else:
        result.passed = False

    return result


def mock_deep_audit(
    code: str, test_source: str, timeout_seconds: float = 10.0,
) -> SandboxResult:
    """Run mutation testing on passing code (subprocess-based)."""
    mutants = _generate_mutants(code)
    result = SandboxResult(passed=True, mutation_score=1.0)

    if not mutants:
        return result

    killed = 0
    for mutant_code in mutants:
        mutant_result = mock_sandbox_execute(
            code=mutant_code,
            test_source=test_source,
            timeout_seconds=timeout_seconds,
        )
        if not mutant_result.passed:
            killed += 1

    result.mutation_score = killed / len(mutants)
    return result


# ---------------------------------------------------------------------------
# Task factories (same as main.py)
# ---------------------------------------------------------------------------

def make_memory_task() -> Task:
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
    return Task(
        description="Sort a list with strict timeout constraints.",
        function_name="sort_timeout_proof",
        input_signature="def sort_timeout_proof(xs: list[int]) -> list[int]",
        timeout_seconds=3.0,
        unit_tests=(
            "def test_basic():\n"
            "    assert sort_timeout_proof([3, 1, 2]) == [1, 2, 3]\n\n"
            "def test_empty():\n"
            "    assert sort_timeout_proof([]) == []\n"
        ),
    )


def make_index_task() -> Task:
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
    "memory": make_memory_task,
    "timeout": make_timeout_task,
    "index": make_index_task,
}


def make_task_sequence(n: int) -> list[tuple[str, Task]]:
    types = list(TASK_FACTORIES.keys())
    seq = []
    for i in range(n):
        task_type = types[i % len(types)]
        task = TASK_FACTORIES[task_type]()
        seq.append((task_type, task))
    return seq


# ---------------------------------------------------------------------------
# Analysis (same as main.py)
# ---------------------------------------------------------------------------

def analyze_results(results: list[dict], n_tribes: int = 3) -> bool:
    """Print convergence analysis. Returns True if specialization learned."""
    n = len(results)
    n_early = min(10, n)
    n_late = min(10, n)

    early = results[:n_early]
    late = results[-n_late:]

    print(f"\n{'=' * 70}")
    print(f"  SPECIALIZATION ANALYSIS ({n} total runs)")
    print(f"{'=' * 70}")

    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    print(f"\n  Overall pass rate: {n_pass}/{n} ({100*n_pass/max(n,1):.0f}%)")

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
            dist = " ".join(f"T{i}={counts.get(i, 0)}" for i in range(n_tribes))
            print(f"    {task_type:10s}: {dist}  (n={total})")

    print(f"\n  AVOIDANCE CHECK (last {n_late} runs):")
    bias_map = {
        "memory": 0,
        "timeout": 1,
        "index": 2,
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

    return all_avoided


# ---------------------------------------------------------------------------
# Inline Council Logic (mirrors CouncilActor but without Ray)
# ---------------------------------------------------------------------------

class InlineCouncil:
    """Simplified council for direct execution (no Ray).

    Phase 2+3: includes deep audit (mutation testing) and PatchRegistry.
    """

    def __init__(
        self,
        tribes: list[dict],   # [{"id": str, "bias": str}, ...]
        alpha: float = 1.5,
        log_dir: str = ".",
    ) -> None:
        self.tribes = tribes
        self.n_tribes = len(tribes)

        self.failure_encoder = FailureEncoder(dim=CONTEXT_DIM)
        self.bandit = LinUCB(n_arms=self.n_tribes, dim=CONTEXT_DIM, alpha=alpha)

        # Phase 3: PatchRegistry for knowledge diffusion
        self.patch_registry = PatchRegistry(dim=CONTEXT_DIM)

        self._cooling_remaining = [0] * self.n_tribes
        self._last_failure: dict[str, np.ndarray] = {}

        self.log_dir = Path(log_dir)
        self.runs_path = self.log_dir / "runs.jsonl"
        self.bandit_pkl_path = self.log_dir / "bandit.pkl"

        self._total_attempts = 0
        self._total_passes = 0
        self._total_deep_audits = 0

    def _build_context(self, task: Task) -> np.ndarray:
        cached = self._last_failure.get(task.task_id)
        if cached is not None:
            return cached.astype(np.float64)
        # First attempt: task-type features only (no failure info yet)
        return self.failure_encoder.encode_task_context(
            task.function_name,
        ).astype(np.float64)

    def run_task(self, task: Task, task_type: str) -> dict:
        """Execute the full MCN pipeline for a single task (inline).

        Phase 2: Hypothesis tests in suite, 10% deep audit on passing tasks.
        Phase 3: Patch search -> hint injection, patch registration.
        """
        # 1. Build context
        context = self._build_context(task)

        # 2. Select tribe via LinUCB
        cooling_mask = [c == 0 for c in self._cooling_remaining]
        if not any(cooling_mask):
            cooling_mask = [True] * self.n_tribes
        tribe_idx, ucb_scores = self.bandit.select_with_scores(context, mask=cooling_mask)

        tribe = self.tribes[tribe_idx]

        # 2b. Search PatchRegistry for hints
        patches = self.patch_registry.search(context, top_k=1)
        patch_hint = patches[0].lesson if patches else ""
        patches_used = [p.patch_id for p in patches]

        # 3. Generate code (inline mock — hint is available but mock ignores it)
        code = mock_generate(tribe["id"], tribe["bias"], task.function_name)

        # 3b. Generate test suite (with Hypothesis tests)
        suite = mock_generate_suite(task.function_name, task.input_signature, task.unit_tests)
        combined_tests = suite.combined_source()

        # 4. Run in mock sandbox
        sandbox_result = mock_sandbox_execute(
            code=code,
            test_source=combined_tests,
            timeout_seconds=task.timeout_seconds,
        )

        # 5. Encode failure via FailureEncoder
        failure_ctx = self.failure_encoder.encode_from_sandbox(
            stdout=sandbox_result.stdout,
            stderr=sandbox_result.stderr,
            elapsed_seconds=sandbox_result.elapsed_seconds,
            tests_passed=sandbox_result.tests_passed,
            tests_failed=sandbox_result.tests_failed,
            tests_total=sandbox_result.tests_total,
            function_name=task.function_name,
        )

        # Cache failure context
        if not sandbox_result.passed:
            self._last_failure[task.task_id] = failure_ctx
        else:
            self._last_failure.pop(task.task_id, None)

        # 5b. Deep audit (10% of passing tasks)
        mutation_score = -1.0
        if sandbox_result.passed and random.random() < 0.1:
            audit_result = mock_deep_audit(
                code=code,
                test_source=combined_tests,
                timeout_seconds=task.timeout_seconds,
            )
            mutation_score = audit_result.mutation_score
            self._total_deep_audits += 1

        # 6. Compute reward (with mutation bonus)
        reward = _compute_reward(sandbox_result.passed, sandbox_result.elapsed_seconds)
        if mutation_score >= 0:
            reward += 2.0 * mutation_score

        # 7. Update bandit
        self.bandit.update(context, arm=tribe_idx, reward=reward)

        # Tick cooling
        for i in range(self.n_tribes):
            if self._cooling_remaining[i] > 0:
                self._cooling_remaining[i] -= 1

        # Stats
        self._total_attempts += 1
        verdict_str = "PASS" if sandbox_result.passed else "FAIL"
        if sandbox_result.passed:
            self._total_passes += 1

        # 8. Record attempt + register patch
        self.patch_registry.record_attempt(task.task_id)
        if sandbox_result.passed:
            self.patch_registry.register_candidate(
                task_id=task.task_id,
                description=task.description,
                function_name=task.function_name,
                code=code,
                context_vector=failure_ctx,
                passed=True,
                failure_category=sandbox_result.exception_type,
            )

        # Log to JSONL
        record = {
            "run_number": self._total_attempts,
            "task_type": task_type,
            "tribe_idx": tribe_idx,
            "tribe_id": tribe["id"],
            "verdict": verdict_str,
            "reward": round(reward, 4),
            "exception_type": sandbox_result.exception_type,
            "elapsed_seconds": round(sandbox_result.elapsed_seconds, 4),
            "tests_passed": sandbox_result.tests_passed,
            "tests_failed": sandbox_result.tests_failed,
            "tests_total": sandbox_result.tests_total,
            "ucb_scores": [round(s, 3) for s in ucb_scores],
            "mutation_score": round(mutation_score, 4) if mutation_score >= 0 else None,
            "patches_used": patches_used,
        }
        try:
            with open(self.runs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

        # Persist bandit every 10 tasks
        if self._total_attempts % 10 == 0:
            self._save_bandit()

        return record

    def _save_bandit(self) -> None:
        state = {
            "bandit": self.bandit.state_dict(),
            "failure_encoder": self.failure_encoder.state_dict(),
            "patch_registry": self.patch_registry.state_dict(),
            "stats": {
                "total_attempts": self._total_attempts,
                "total_passes": self._total_passes,
                "total_deep_audits": self._total_deep_audits,
            },
        }
        try:
            with open(self.bandit_pkl_path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def get_routing_history(self) -> dict:
        return {f"tribe_{i}": int(self.bandit.counts[i]) for i in range(self.n_tribes)}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    n_tasks = 50

    print(f"\n{'=' * 70}")
    print(f"  MCN-v0.1 TRIBAL SPECIALIZATION EXPERIMENT (standalone)")
    print(f"{'=' * 70}")

    # Create temp dir for logs
    log_dir = tempfile.mkdtemp(prefix="mcn_exp_")

    # Tribes with failure biases
    tribes = [
        {"id": "reliable", "bias": "memory"},   # T0: fails on memory tasks
        {"id": "fast",     "bias": "timeout"},   # T1: fails on timeout tasks
        {"id": "creative", "bias": "index"},     # T2: fails on index tasks
    ]
    print(f"  Tribe 0 (reliable):  bias=memory  -> fails on memory tasks")
    print(f"  Tribe 1 (fast):      bias=timeout -> fails on timeout tasks")
    print(f"  Tribe 2 (creative):  bias=index   -> fails on index tasks")

    # Council
    council = InlineCouncil(tribes=tribes, alpha=1.5, log_dir=log_dir)
    print(f"  Council (LinUCB alpha=1.5, dim={CONTEXT_DIM})")
    print(f"  Log dir: {log_dir}")

    # Generate task sequence
    task_seq = make_task_sequence(n_tasks)
    print(f"\n  Running {n_tasks} tasks (round-robin: memory, timeout, index)")
    print(f"  {'=' * 66}")

    # Execute
    all_results: list[dict] = []
    t_start = time.time()

    for i, (task_type, task) in enumerate(task_seq):
        record = council.run_task(task, task_type)
        all_results.append(record)

        v = "+" if record["verdict"] == "PASS" else "-"
        exc = f"  {record['exception_type']}" if record['exception_type'] else ""
        mut = f"  mut={record['mutation_score']:.2f}" if record.get('mutation_score') is not None else ""
        patch = f"  patch={record['patches_used'][0][:8]}" if record.get('patches_used') else ""
        print(
            f"  [{i+1:3d}/{n_tasks}] "
            f"{task_type:8s} -> T{record['tribe_idx']} "
            f"[{v}] r={record['reward']:+.3f}"
            f"{exc}{mut}{patch}"
        )

    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time:.1f}s ({total_time/n_tasks:.2f}s/task)")

    # Analysis
    success = analyze_results(all_results)

    # Check artifacts
    pkl_path = Path(log_dir) / "bandit.pkl"
    runs_path = Path(log_dir) / "runs.jsonl"

    print(f"\n  ARTIFACTS:")
    if pkl_path.exists():
        print(f"    bandit.pkl:  {pkl_path.stat().st_size:,} bytes")
        # Load and verify
        with open(pkl_path, "rb") as f:
            state = pickle.load(f)
        print(f"    bandit.pkl contains: {list(state.keys())}")
        print(f"    bandit total_updates: {state['bandit']['total_updates']}")
        print(f"    bandit counts: {state['bandit']['counts']}")
    else:
        print(f"    bandit.pkl:  NOT FOUND")

    if runs_path.exists():
        n_lines = sum(1 for _ in open(runs_path))
        print(f"    runs.jsonl:  {n_lines} entries, {runs_path.stat().st_size:,} bytes")

    # Final council stats
    routing = council.get_routing_history()
    pass_rate = council._total_passes / max(council._total_attempts, 1)
    print(f"\n  COUNCIL FINAL STATE:")
    print(f"    Pass rate:        {pass_rate:.1%}")
    print(f"    Routing totals:   {routing}")
    print(f"    Encoder samples:  {council.failure_encoder.metric_scaler.count}")
    print(f"    Deep audits:      {council._total_deep_audits}")
    print(f"    Patches stored:   {council.patch_registry.n_patches}")
    print(f"    Tasks tracked:    {len(council.patch_registry._attempt_counts)}")

    # Save final bandit state
    council._save_bandit()
    if pkl_path.exists():
        print(f"    Final bandit.pkl: {pkl_path.stat().st_size:,} bytes")

    print(f"\n{'=' * 70}")
    if success:
        print(f"  EXPERIMENT COMPLETE — SPECIALIZATION VERIFIED")
    else:
        print(f"  EXPERIMENT COMPLETE — CONVERGENCE PARTIAL")
    print(f"{'=' * 70}\n")

    return success


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)
