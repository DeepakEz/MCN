"""MCN-v0.1 Sandbox Executor — Deterministic Evidence Gate.

Ray actor that runs generated code + tests inside an isolated Docker
container and returns structured results.

Evidence gate:
    E(y, tau) = 1[ Exec(p, T(tau), C) = PASS ]

Security model:
    - network_mode='none'  : no network access
    - remove=True          : container deleted after execution
    - mem_limit            : enforced by Docker
    - timeout              : enforced by container stop + wall-clock
    - read_only rootfs     : prevents writes outside /tmp

Usage (via Ray):
    sandbox = SandboxExecutor.remote()
    result = ray.get(sandbox.execute.remote(code, tests, timeout=10.0))
"""

from __future__ import annotations

import ast
import copy
import logging
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ray

from mcn.config import MCNConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass (internal to sandbox — converted to protocol types upstream)
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """Raw output from a sandbox execution.

    Upstream code converts this into AttemptResult + FailureSignature.
    """
    passed: bool = False
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = 0.0
    timed_out: bool = False

    # parsed test counts (from pytest output)
    tests_passed: int = 0
    tests_failed: int = 0
    tests_errored: int = 0
    tests_total: int = 0

    # failure details parsed from output
    exception_type: str = ""
    exception_message: str = ""
    failed_test_names: list[str] = field(default_factory=list)

    # mutation testing score (-1.0 = not audited, 0.0..1.0 = audited)
    mutation_score: float = -1.0


# ---------------------------------------------------------------------------
# Pytest output parser
# ---------------------------------------------------------------------------

# Matches: "===== 3 passed, 1 failed, 2 error in 0.45s ====="
_PYTEST_SUMMARY_RE = re.compile(
    r"(?P<passed>\d+)\s+passed"
    r"|(?P<failed>\d+)\s+failed"
    r"|(?P<error>\d+)\s+error"
)

# Matches two pytest failure formats:
#   Verbose: "FAILED test_foo.py::test_bar - IndexError: list index out of range"
#   Quiet:   "test_foo.py::test_bar FAILED"
_PYTEST_FAILED_RE = re.compile(
    r"^(?:"
    r"FAILED\s+(?P<name1>\S+::\S+)"       # verbose: FAILED name
    r"|"
    r"(?P<name2>\S+::\S+)\s+FAILED"       # quiet:   name FAILED
    r")"
    r"(?:\s*-\s*(?P<exc_type>\w+Error|Exception):\s*(?P<exc_msg>.+))?",
    re.MULTILINE,
)

# Matches: "E   IndexError: list index out of range"
_PYTEST_ERROR_LINE_RE = re.compile(
    r"^E\s+(?P<exc_type>\w+(?:Error|Exception)):\s*(?P<exc_msg>.+)$",
    re.MULTILINE,
)


def _parse_pytest_output(stdout: str, stderr: str) -> dict:
    """Extract structured test results from pytest output.

    Returns a dict with keys matching SandboxResult fields.
    """
    combined = stdout + "\n" + stderr
    result: dict = {
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_errored": 0,
        "exception_type": "",
        "exception_message": "",
        "failed_test_names": [],
    }

    # Parse summary line
    for match in _PYTEST_SUMMARY_RE.finditer(combined):
        if match.group("passed"):
            result["tests_passed"] = int(match.group("passed"))
        if match.group("failed"):
            result["tests_failed"] = int(match.group("failed"))
        if match.group("error"):
            result["tests_errored"] = int(match.group("error"))

    result["tests_total"] = (
        result["tests_passed"]
        + result["tests_failed"]
        + result["tests_errored"]
    )

    # Parse individual failures (handles both verbose and quiet formats)
    for match in _PYTEST_FAILED_RE.finditer(combined):
        name = match.group("name1") or match.group("name2")
        if name:
            result["failed_test_names"].append(name)
        if match.group("exc_type") and not result["exception_type"]:
            result["exception_type"] = match.group("exc_type")
            result["exception_message"] = match.group("exc_msg") or ""

    # Fallback: grab first "E   SomeError:" line
    if not result["exception_type"]:
        err_match = _PYTEST_ERROR_LINE_RE.search(combined)
        if err_match:
            result["exception_type"] = err_match.group("exc_type")
            result["exception_message"] = err_match.group("exc_msg")

    return result


# ---------------------------------------------------------------------------
# AST Mutation Engine — operator-level mutations for deep audit
# ---------------------------------------------------------------------------

# Comparison operator flips
_COMPARISON_FLIPS: dict[type, type] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}

# Binary operator flips
_BINOP_FLIPS: dict[type, type] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
}


class _MutationVisitor(ast.NodeTransformer):
    """Apply a single mutation at a specific target index.

    Walks the AST, counting mutable nodes. When the counter matches
    target_idx, applies the mutation. Otherwise, nodes pass through.
    """

    def __init__(self, target_idx: int) -> None:
        super().__init__()
        self.target_idx = target_idx
        self._counter = 0
        self.mutated = False

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        """Flip comparison operators (e.g., < -> <=)."""
        new_ops = []
        for op in node.ops:
            if type(op) in _COMPARISON_FLIPS:
                if self._counter == self.target_idx:
                    new_ops.append(_COMPARISON_FLIPS[type(op)]())
                    self.mutated = True
                else:
                    new_ops.append(op)
                self._counter += 1
            else:
                new_ops.append(op)
        node.ops = new_ops
        self.generic_visit(node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        """Flip binary operators (e.g., + -> -)."""
        if type(node.op) in _BINOP_FLIPS:
            if self._counter == self.target_idx:
                node.op = _BINOP_FLIPS[type(node.op)]()
                self.mutated = True
            self._counter += 1
        self.generic_visit(node)
        return node


def _count_mutable_nodes(tree: ast.AST) -> int:
    """Count the number of flippable operators in an AST."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if type(op) in _COMPARISON_FLIPS:
                    count += 1
        elif isinstance(node, ast.BinOp):
            if type(node.op) in _BINOP_FLIPS:
                count += 1
    return count


def _generate_mutants(code: str) -> list[str]:
    """Generate all single-mutation source variants of the given code.

    Parses the code into an AST, then for each mutable operator,
    creates a copy with that single operator flipped.

    Returns a list of mutated source strings. Empty list if code
    can't be parsed or has no mutable operators.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    n_mutable = _count_mutable_nodes(tree)
    if n_mutable == 0:
        return []

    mutants: list[str] = []
    for idx in range(n_mutable):
        tree_copy = copy.deepcopy(tree)
        visitor = _MutationVisitor(target_idx=idx)
        mutated_tree = visitor.visit(tree_copy)
        if visitor.mutated:
            try:
                ast.fix_missing_locations(mutated_tree)
                mutant_code = ast.unparse(mutated_tree)
                mutants.append(mutant_code)
            except Exception:
                continue

    return mutants


# ---------------------------------------------------------------------------
# Ray Actor
# ---------------------------------------------------------------------------

@ray.remote
class SandboxExecutor:
    """Isolated Docker-based code executor.

    Each instance holds a Docker client. Execution is synchronous
    inside the actor — Ray handles concurrency across actors.

    Constructor args:
        docker_image: Base image for execution (must have python + pytest).
        default_mem_limit: Default memory limit string (e.g. "256m").
    """

    def __init__(
        self,
        docker_image: str = "mcn-sandbox:v0.1",
        default_mem_limit: str = "256m",
    ) -> None:
        # Lazy import: docker may not be installed in all environments
        import docker
        self.client = docker.from_env()
        self.docker_image = docker_image
        self.default_mem_limit = default_mem_limit
        self._ensure_image()

    def _ensure_image(self) -> None:
        """Pull the base image if not already present."""
        try:
            self.client.images.get(self.docker_image)
        except Exception:
            logger.info("Pulling Docker image: %s", self.docker_image)
            self.client.images.pull(self.docker_image)

    def execute(
        self,
        code: str,
        test_source: str,
        timeout_seconds: float = 10.0,
        mem_limit: Optional[str] = None,
    ) -> SandboxResult:
        """Run code + tests in an isolated container.

        Steps:
          1. Write code to a temp file (solution.py)
          2. Write tests to a temp file (test_solution.py)
          3. Mount both into a Docker container
          4. Run: python -m pytest test_solution.py -v --tb=short
          5. Capture stdout/stderr, parse results
          6. Container is auto-removed (remove=True)

        Args:
            code: The generated Python source code.
            test_source: Pytest-compatible test source (combined test suite).
            timeout_seconds: Max execution time before kill.
            mem_limit: Docker memory limit (e.g. "256m"). Falls back to default.

        Returns:
            SandboxResult with structured pass/fail information.
        """
        if mem_limit is None:
            mem_limit = self.default_mem_limit

        result = SandboxResult()
        start_time = time.monotonic()

        # Write code and tests to a temp directory
        # Use ramdisk (/dev/shm) if configured for faster I/O
        tmpdir_base = MCNConfig.SANDBOX_TMPDIR or None
        with tempfile.TemporaryDirectory(prefix="mcn_sandbox_", dir=tmpdir_base) as tmpdir:
            code_path = Path(tmpdir) / "solution.py"
            test_path = Path(tmpdir) / "test_solution.py"

            code_path.write_text(code, encoding="utf-8")

            # Prepend import of solution module to tests
            test_preamble = (
                "import sys, os\n"
                "sys.path.insert(0, '/workspace')\n"
                "from solution import *\n\n"
            )
            test_path.write_text(
                test_preamble + test_source, encoding="utf-8"
            )

            # Also write a conftest.py to set import paths cleanly
            conftest_path = Path(tmpdir) / "conftest.py"
            conftest_path.write_text(
                "import sys\nsys.path.insert(0, '/workspace')\n",
                encoding="utf-8",
            )

            try:
                container = self.client.containers.run(
                    image=self.docker_image,
                    command=(
                        "python -m pytest /workspace/test_solution.py "
                        "-v --tb=short --no-header -q"
                    ),
                    volumes={
                        tmpdir: {"bind": "/workspace", "mode": "ro"},
                    },
                    working_dir="/workspace",
                    mem_limit=mem_limit,
                    network_mode="none",
                    remove=False,       # we remove manually after log capture
                    read_only=True,     # read-only rootfs
                    tmpfs={"/tmp": "size=64m"},  # writable /tmp for pytest
                    detach=True,
                    stderr=True,
                    stdout=True,
                )

                # Wait with timeout
                try:
                    wait_result = container.wait(
                        timeout=timeout_seconds
                    )
                    result.exit_code = wait_result.get("StatusCode", -1)
                except Exception:
                    # Timeout — kill the container
                    result.timed_out = True
                    result.exit_code = -1
                    try:
                        container.kill()
                    except Exception:
                        pass

                # Capture logs
                try:
                    result.stdout = container.logs(
                        stdout=True, stderr=False
                    ).decode("utf-8", errors="replace")
                    result.stderr = container.logs(
                        stdout=False, stderr=True
                    ).decode("utf-8", errors="replace")
                except Exception as e:
                    result.stderr = f"[MCN] Failed to capture logs: {e}"

                # Clean up container
                try:
                    container.remove(force=True)
                except Exception:
                    pass

            except Exception as e:
                result.stderr = f"[MCN] Docker execution error: {e}"
                result.exit_code = -1

        result.elapsed_seconds = time.monotonic() - start_time

        # Parse pytest output into structured fields
        parsed = _parse_pytest_output(result.stdout, result.stderr)
        result.tests_passed = parsed["tests_passed"]
        result.tests_failed = parsed["tests_failed"]
        result.tests_errored = parsed["tests_errored"]
        result.tests_total = parsed["tests_total"]
        result.exception_type = parsed["exception_type"]
        result.exception_message = parsed["exception_message"]
        result.failed_test_names = parsed["failed_test_names"]

        # Determine pass/fail
        if result.timed_out:
            result.passed = False
            result.exception_type = result.exception_type or "TimeoutError"
            result.exception_message = (
                result.exception_message
                or f"Exceeded {timeout_seconds}s limit"
            )
        elif result.exit_code == 0 and result.tests_failed == 0 and result.tests_errored == 0:
            result.passed = True
        else:
            result.passed = False

        return result

    def run_deep_audit(
        self,
        code: str,
        test_source: str,
        timeout_seconds: float = 10.0,
        mem_limit: Optional[str] = None,
    ) -> SandboxResult:
        """Run mutation testing: generate mutants, test each, compute kill rate.

        Generates all single-operator mutations of the code, runs tests
        against each mutant. A mutant is "killed" if tests fail.

        mutation_score = killed / total_mutants

        If no mutants can be generated, returns score=1.0 (vacuously robust).
        The returned SandboxResult has mutation_score set.

        Args:
            code: The original (passing) code to audit.
            test_source: Combined test suite source.
            timeout_seconds: Per-mutant execution timeout.
            mem_limit: Docker memory limit.

        Returns:
            SandboxResult with mutation_score populated.
        """
        mutants = _generate_mutants(code)
        result = SandboxResult(passed=True, mutation_score=1.0)

        if not mutants:
            # No mutable operators -> vacuously robust
            return result

        killed = 0
        for mutant_code in mutants:
            mutant_result = self.execute(
                code=mutant_code,
                test_source=test_source,
                timeout_seconds=timeout_seconds,
                mem_limit=mem_limit,
            )
            # Mutant is killed if tests fail (good — tests caught the mutation)
            if not mutant_result.passed:
                killed += 1

        result.mutation_score = killed / len(mutants)
        return result

    def health_check(self) -> bool:
        """Verify Docker connectivity and image availability."""
        try:
            self.client.ping()
            self.client.images.get(self.docker_image)
            return True
        except Exception:
            return False
