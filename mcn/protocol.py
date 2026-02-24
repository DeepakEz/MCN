"""MCN-v0.1 Protocol Definitions.

Strict dataclass contracts for all inter-module messages.
Every field is typed; no dicts-as-messages allowed.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GateVerdict(Enum):
    """Deterministic evidence-gate outcome."""
    PASS = auto()
    FAIL = auto()
    TIMEOUT = auto()
    ERROR = auto()          # sandbox-level error (not the code's fault)


class OverseerDecision(Enum):
    """Overseer triage after gate evaluation."""
    ACCEPT = auto()
    REVISE = auto()         # known failure signature -> retry with hint
    ESCALATE = auto()       # unknown / repeated failure -> council decides


class FailureCategory(Enum):
    """Top-level exception family for structured failure signatures.

    Maps to F_exc in the formal spec:
    f in F = F_exc x F_loc x F_resource x F_test
    """
    NONE = auto()           # no failure (success)
    TYPE_ERROR = auto()
    INDEX_ERROR = auto()
    VALUE_ERROR = auto()
    KEY_ERROR = auto()
    ATTRIBUTE_ERROR = auto()
    RECURSION_ERROR = auto()
    TIMEOUT = auto()
    MEMORY_ERROR = auto()
    ASSERTION_ERROR = auto()  # test assertion failed
    IMPORT_ERROR = auto()
    SYNTAX_ERROR = auto()
    RUNTIME_OTHER = auto()
    UNKNOWN = auto()


# ---------------------------------------------------------------------------
# Context vector dimension constants
# Single source of truth: encoder and bandit both depend on these.
#   [0..N_TASK_TYPES)   one-hot task-type keywords (memory, timeout, index)
#   [N_TASK_TYPES..N_TASK_TYPES+N_EXC)  one-hot exception type (10 slots)
#   [N_TASK_TYPES+N_EXC..)  Z-scored resource metrics (5 slots)
# ---------------------------------------------------------------------------

N_TASK_TYPES: int = 3   # task-type keyword one-hot slots
N_EXC: int = 10         # exception-type one-hot slots
N_METRICS: int = 5      # Z-scored metric slots
CONTEXT_DIM: int = N_TASK_TYPES + N_EXC + N_METRICS  # 18


# ---------------------------------------------------------------------------
# Exception-type -> FailureCategory mapping  (single source of truth)
# ---------------------------------------------------------------------------

EXC_TO_CATEGORY: dict[str, FailureCategory] = {
    "TypeError": FailureCategory.TYPE_ERROR,
    "IndexError": FailureCategory.INDEX_ERROR,
    "ValueError": FailureCategory.VALUE_ERROR,
    "KeyError": FailureCategory.KEY_ERROR,
    "AttributeError": FailureCategory.ATTRIBUTE_ERROR,
    "RecursionError": FailureCategory.RECURSION_ERROR,
    "TimeoutError": FailureCategory.TIMEOUT,
    "MemoryError": FailureCategory.MEMORY_ERROR,
    "AssertionError": FailureCategory.ASSERTION_ERROR,
    "ImportError": FailureCategory.IMPORT_ERROR,
    "ModuleNotFoundError": FailureCategory.IMPORT_ERROR,
    "SyntaxError": FailureCategory.SYNTAX_ERROR,
}


def classify_exception(exc_type: str) -> FailureCategory:
    """Map an exception type string to a FailureCategory enum."""
    return EXC_TO_CATEGORY.get(exc_type, FailureCategory.RUNTIME_OTHER)


# ---------------------------------------------------------------------------
# Core message types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """A single code-synthesis task instance.

    Maps to x = (spec, C, R) in the formal spec.

    Attributes:
        task_id: Unique identifier.
        description: Natural-language specification.
        function_name: Entry-point function name expected in output.
        input_signature: Type signature string, e.g. "def f(xs: list[int]) -> int".
        reference_solution: Oracle solution (if available).
        timeout_seconds: Max wall-clock time for sandbox execution.
        memory_limit_mb: Max memory for sandbox container.
        forbidden_imports: Modules the generated code must not use.
        style_rules: Soft constraints for code style.
        unit_tests: Pytest-compatible test source from the task spec.
        property_tests: Hypothesis-compatible property test source.
    """
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # --- spec ---
    description: str = ""
    function_name: str = ""
    input_signature: str = ""
    reference_solution: Optional[str] = None

    # --- constraints C ---
    timeout_seconds: float = 10.0
    memory_limit_mb: int = 256
    forbidden_imports: tuple[str, ...] = ()
    style_rules: tuple[str, ...] = ()

    # --- reference checks R ---
    unit_tests: str = ""
    property_tests: str = ""


@dataclass(frozen=True)
class TestSuite:
    """Aggregated test battery sent to the sandbox.

    T(tau) = T_self U T_overseer U T_fuzz U T_unit
    """
    self_tests: str = ""           # tests written by the tribe itself
    overseer_tests: str = ""       # adversarial edge-case tests
    fuzz_tests: str = ""           # property-based / hypothesis tests
    unit_tests: str = ""           # from the task spec (R)

    def combined_source(self) -> str:
        """Concatenate all test sources into a single pytest-runnable module."""
        sections = [
            s for s in (
                self.unit_tests,
                self.self_tests,
                self.overseer_tests,
                self.fuzz_tests,
            ) if s.strip()
        ]
        return "\n\n".join(sections)


@dataclass(frozen=True)
class FailureSignature:
    """Structured failure descriptor.

    f in F = F_exc x F_loc x F_resource x F_test

    The embedding Phi(f) -> R^d is computed by to_feature_vector().
    v0.1 uses a hand-crafted feature map; learned calibration comes later.
    """
    category: FailureCategory = FailureCategory.NONE
    exception_type: str = ""          # raw class name, e.g. "IndexError"
    exception_message: str = ""
    location_file: str = ""
    location_line: int = -1
    location_ast_node: str = ""       # e.g. "Subscript", "Call"

    # resource usage at failure
    elapsed_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    recursion_depth: int = 0

    # test-level info
    tests_passed: int = 0
    tests_failed: int = 0
    tests_errored: int = 0
    failed_test_names: tuple[str, ...] = ()

    @property
    def is_success(self) -> bool:
        return self.category == FailureCategory.NONE

    def to_feature_vector(self, dim: int = 32) -> NDArray[np.float32]:
        """Hand-crafted Phi: structured fields -> R^d.

        Feature layout:
          [0..N_cat)    one-hot failure category
          [N_cat]       normalized elapsed time
          [N_cat+1]     normalized peak memory
          [N_cat+2]     normalized recursion depth
          [N_cat+3]     test pass ratio
          [N_cat+4]     test fail ratio
          [N_cat+5]     test error ratio
          [rest]        zero-padded
        """
        features: list[float] = []

        # one-hot for category
        n_cats = len(FailureCategory)
        cat_vec = [0.0] * n_cats
        cat_vec[self.category.value - 1] = 1.0
        features.extend(cat_vec)

        # resource features (rough normalization — RunningScaler refines later)
        features.append(min(self.elapsed_seconds / 30.0, 1.0))
        features.append(min(self.peak_memory_mb / 512.0, 1.0))
        features.append(min(self.recursion_depth / 1000.0, 1.0))

        # test outcome ratios
        total = self.tests_passed + self.tests_failed + self.tests_errored
        if total > 0:
            features.append(self.tests_passed / total)
            features.append(self.tests_failed / total)
            features.append(self.tests_errored / total)
        else:
            features.extend([0.0, 0.0, 0.0])

        # pad or truncate to target dim
        vec = np.array(features, dtype=np.float32)
        if len(vec) < dim:
            vec = np.concatenate([vec, np.zeros(dim - len(vec), dtype=np.float32)])
        else:
            vec = vec[:dim]

        return vec


@dataclass(frozen=True)
class AttemptResult:
    """Full record of a single tribe attempt.

    Stored in Campfire buffer as zeta = (x, p, T, r, f).
    This is the primary message flowing through the system.
    """
    task_id: str
    tribe_id: str
    attempt_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    # --- generated artefacts ---
    plan: str = ""                  # chain-of-thought / plan text
    code: str = ""                  # generated program p
    self_tests: str = ""            # tests the tribe wrote for itself

    # --- gate outcome ---
    verdict: GateVerdict = GateVerdict.ERROR
    overseer_decision: OverseerDecision = OverseerDecision.ESCALATE

    # --- failure signature (embedded) ---
    failure_signature: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(32, dtype=np.float32),
    )
    cluster_id: int = -1            # assigned by failure-space clustering

    # --- structured failure info (pre-embedding) ---
    failure_info: FailureSignature = field(default_factory=FailureSignature)

    # --- resource usage ---
    elapsed_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    generation_tokens: int = 0      # LLM tokens consumed

    # --- reward ---
    reward: float = 0.0             # R = 1[PASS] - c_t*time - c_m*mem - c_s*steps
    sandbox_log: str = ""           # raw stdout/stderr from sandbox


@dataclass
class PatchRecord:
    """A distilled knowledge patch: few-shot context injection.

    P_ij = verified trajectories from Tribe i applicable to Tribe j.
    Norm bound ||P_ij|| <= P_max is enforced as token_count <= L_max.
    """
    patch_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_tribe_id: str = ""
    target_tribe_id: str = ""
    cluster_id: int = -1

    # the actual patch content (few-shot exemplars)
    exemplars: list[str] = field(default_factory=list)
    token_count: int = 0            # ||P_ij|| in token space
    created_at: float = field(default_factory=time.time)
    times_applied: int = 0
    verified_successes: int = 0     # successes after this patch was active

    @property
    def utility(self) -> float:
        """Patch utility for LRU-weighted eviction.

        Higher utility -> survives longer in PatchStore.
        """
        if self.times_applied == 0:
            return 0.0
        return self.verified_successes / self.times_applied
