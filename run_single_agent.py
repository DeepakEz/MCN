"""MCN — Experiment A: Single-Agent Ablation Baseline.

Runs a single tribe (no routing, no council, no bandit) on the same task
sequence used in the main MCN experiment.  This establishes the ablation
baseline: if MCN's multi-tribe routing does NOT beat this, the routing adds
no value.

Methodology
-----------
1. We pick ONE tribe (default: Tribe 1, 0-indexed, the "fast" coder).
   You can choose any tribe with --tribe-idx.
2. We run it on the same 100-task round-robin sequence from run_live_experiment.py.
3. We record pass/fail per task-type (same schema as runs.jsonl).
4. We print a side-by-side report for direct comparison with MCN runs.

Note: uses the same TribeActor + OverseerActor + Sandbox infrastructure
as the full MCN pipeline, so the only variable is routing (none here).

Usage:
    # Inside docker compose (recommended):
    docker compose run --rm mcn-runner python run_single_agent.py -n 100 --tribe-idx 1

    # Standalone:
    python run_single_agent.py -n 100 --tribe-idx 1 --log-dir /results/single_agent
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

# Reuse the exact same task list and factory from the main experiment
from run_live_experiment import LIVE_TASKS, make_live_task, make_live_task_sequence


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
# Report
# ---------------------------------------------------------------------------

def print_report(
    results: list[dict],
    total_time: float,
    n_tasks: int,
    tribe_idx: int,
) -> None:
    """Print ablation experiment report."""
    n = len(results)
    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    n_fail = n - n_pass
    total_tokens = sum(r.get("tokens", 0) for r in results)

    print(f"\n{'=' * 70}")
    print(f"  SINGLE-AGENT ABLATION REPORT  (Tribe {tribe_idx} only, no routing)")
    print(f"{'=' * 70}")
    print(f"\n  Results:")
    print(f"    Total tasks:  {n}")
    print(f"    Passed:       {n_pass} ({100*n_pass/max(n,1):.1f}%)")
    print(f"    Failed:       {n_fail} ({100*n_fail/max(n,1):.1f}%)")
    print(f"\n  Performance:")
    print(f"    Total time:   {total_time:.1f}s")
    print(f"    Avg per task: {total_time/max(n,1):.2f}s")
    if total_tokens > 0:
        print(f"    Total tokens: {total_tokens:,}")
        print(f"    Avg tokens:   {total_tokens/max(n,1):.0f}")

    # Per-task-type breakdown
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0})
    for r in results:
        key = "pass" if r["verdict"] == "PASS" else "fail"
        by_type[r["task_type"]][key] += 1

    print(f"\n  Per-task breakdown:")
    for task_type in sorted(by_type):
        stats = by_type[task_type]
        total = stats["pass"] + stats["fail"]
        rate = 100 * stats["pass"] / max(total, 1)
        print(f"    {task_type:20s}: {stats['pass']}/{total} ({rate:.0f}%)")

    # Error distribution
    errors = defaultdict(int)
    for r in results:
        if r["verdict"] != "PASS" and r.get("exception"):
            errors[r["exception"]] += 1
    if errors:
        print(f"\n  Error distribution:")
        for exc, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"    {exc}: {count}")

    print(f"\n{'=' * 70}")
    print(f"  COMPARISON HINT:")
    print(f"    If MCN (multi-tribe) pass rate > {100*n_pass/max(n,1):.1f}%, routing adds value.")
    print(f"    If MCN pass rate <= {100*n_pass/max(n,1):.1f}%, routing provides no benefit.")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCN Experiment A: Single-Agent Ablation",
    )
    parser.add_argument(
        "--tribe-idx", type=int, default=1,
        help="Which tribe to use (0=reliable, 1=fast, 2=creative). Default: 1",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock mode (no vLLM needed)",
    )
    parser.add_argument(
        "-n", "--num-tasks", type=int, default=100,
        help="Number of tasks to run (default 100, same as MCN experiment)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="",
        help="Directory for results.jsonl output",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Debug logging",
    )
    parser.add_argument(
        "--skip-checks", action="store_true",
        help="Skip infrastructure validation",
    )
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger("mcn.ablation")

    # Resolve log directory
    import tempfile
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = Path(tempfile.mkdtemp(prefix="mcn_ablation_"))
    log_dir.mkdir(parents=True, exist_ok=True)

    use_mock = args.mock
    tribe_idx = args.tribe_idx

    print(f"\n{'=' * 70}")
    print(f"  MCN EXPERIMENT A — SINGLE-AGENT ABLATION")
    print(f"  Tribe: {tribe_idx} ({'mock' if use_mock else MCNConfig.VLLM_MODEL})")
    print(f"  Tasks: {args.num_tasks}")
    print(f"  Log dir: {log_dir}")
    print(f"{'=' * 70}")

    # --- Infrastructure checks ---
    if not use_mock and not args.skip_checks:
        from run_live_experiment import check_vllm
        vllm_ok = check_vllm(MCNConfig.VLLM_BASE_URL)
        print(f"  vLLM: {'OK' if vllm_ok else 'FAIL'}")
        if not vllm_ok:
            print("  ERROR: vLLM not available")
            sys.exit(1)

    # --- Init Ray ---
    ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)

    # --- Build ONE tribe ---
    from mcn.tribe import TribeActor
    from mcn.overseer import OverseerActor

    if tribe_idx >= MCNConfig.NUM_TRIBES:
        print(f"  ERROR: tribe_idx={tribe_idx} out of range (NUM_TRIBES={MCNConfig.NUM_TRIBES})")
        sys.exit(1)

    prompt = (
        MCNConfig.TRIBE_PROMPTS[tribe_idx]
        if tribe_idx < len(MCNConfig.TRIBE_PROMPTS)
        else f"You are tribe {tribe_idx}."
    )
    # Use the per-tribe temperature if TRIBE_TEMPERATURES is configured,
    # otherwise fall back to the global TRIBE_TEMPERATURE.
    # This ensures the ablation exactly mirrors the temperature that
    # tribe_idx uses in the full MCN run (e.g. tribe 2 = T=0.9 in Run 10).
    _temps = MCNConfig.TRIBE_TEMPERATURES
    temp = (
        _temps[tribe_idx]
        if _temps and tribe_idx < len(_temps)
        else MCNConfig.TRIBE_TEMPERATURE
    )
    tribe = TribeActor.remote(
        tribe_id=f"tribe_{tribe_idx}",
        system_prompt=prompt,
        model=MCNConfig.VLLM_MODEL,
        temperature=temp,
        max_tokens=MCNConfig.TRIBE_MAX_TOKENS,
        use_mock=use_mock,
        failure_bias="",
    )
    overseer = OverseerActor.remote()

    from main import MockSandboxExecutor
    sandbox = MockSandboxExecutor.remote()

    print(f"  Tribe {tribe_idx}: {MCNConfig.VLLM_MODEL if not use_mock else 'MOCK'}")
    print(f"  Running {args.num_tasks} tasks (no routing, no council)...")
    print(f"  {'='*66}")

    # --- Task sequence ---
    task_seq = make_live_task_sequence(args.num_tasks)

    all_results: list[dict] = []
    t_start = time.time()
    results_path = log_dir / "single_agent_runs.jsonl"

    with open(results_path, "w") as out:
        for i, (task_type, task) in enumerate(task_seq):
            step_start = time.time()

            # 1. PARALLEL: tribe generates solution + overseer generates test suite
            #    (exact same pattern as council.py, minus routing and patch memory)
            gen_ref = tribe.generate.remote(
                task_description=task.description,
                function_name=task.function_name,
                input_signature=task.input_signature,
                unit_tests=task.unit_tests,
                hint=None,                               # no patch memory in ablation
                reference_solution=task.reference_solution,
            )
            suite_ref = overseer.generate_suite.remote(
                task_description=task.description,
                function_name=task.function_name,
                input_signature=task.input_signature,
                unit_tests=task.unit_tests,
            )

            gen_result, test_suite = ray.get([gen_ref, suite_ref])

            # Handle generation failure
            if gen_result.error:
                record = {
                    "run": i + 1,
                    "task_type": task_type,
                    "tribe_idx": tribe_idx,
                    "tribe_id": f"tribe_{tribe_idx}",
                    "verdict": "FAIL",
                    "exception": "GenerationError",
                    "tokens": 0,
                    "elapsed": round(time.time() - step_start, 3),
                }
                all_results.append(record)
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(
                    f"  [{i+1:3d}/{args.num_tasks}] "
                    f"{task_type:20s} -> T{tribe_idx} "
                    f"[-]  GenerationError"
                )
                continue

            # 2. Run combined tests in sandbox
            combined_tests = test_suite.combined_source()
            sandbox_result = ray.get(
                sandbox.execute.remote(
                    code=gen_result.code,
                    test_source=combined_tests,
                    timeout_seconds=task.timeout_seconds,
                    mem_limit=f"{task.memory_limit_mb}m",
                )
            )

            step_time = time.time() - step_start
            verdict = "PASS" if sandbox_result.passed else "FAIL"

            record = {
                "run": i + 1,
                "task_type": task_type,
                "tribe_idx": tribe_idx,
                "tribe_id": f"tribe_{tribe_idx}",
                "verdict": verdict,
                "exception": sandbox_result.exception_type,
                "tokens": gen_result.tokens_used,
                "elapsed": round(step_time, 3),
            }
            all_results.append(record)
            out.write(json.dumps(record) + "\n")
            out.flush()

            v = "+" if verdict == "PASS" else "-"
            print(
                f"  [{i+1:3d}/{args.num_tasks}] "
                f"{task_type:20s} -> T{tribe_idx} "
                f"[{v}]"
                f"{'  ' + record['exception'] if record['exception'] else ''}"
            )

    total_time = time.time() - t_start

    print_report(all_results, total_time, args.num_tasks, tribe_idx)
    print(f"\n  Results saved to: {results_path}")

    ray.shutdown()
    print(f"\n{'=' * 70}")
    print(f"  ABLATION COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
