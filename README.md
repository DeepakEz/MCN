# Mycelial Council Network (MCN)

> **Does a learned router beat the best single agent when all agents share the same base model?**
>
> Short answer: **No — temperature selection dominates routing strategy.**

MCN is a multi-agent LLM system that routes Python code-synthesis tasks across a *council* of three tribe agents using either a contextual LinUCB bandit or a lightweight GNN. This repository contains the full research codebase, experimental infrastructure, results data, and a generated IEEE-format academic paper.

---

## Table of Contents

- [Key Findings](#key-findings)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Experimental Phases](#experimental-phases)
- [Results](#results)
- [Reproducing the Paper](#reproducing-the-paper)
- [Monitoring](#monitoring)

---

## Key Findings

| Condition | Pass Rate | vs. Best Single Agent |
|---|---|---|
| T=0.3 single agent **(best)** | **91%** | — |
| MCN-GNN (T=0.1/0.5/0.9) | 88% | −3 pp |
| MCN-LinUCB (T=0.3 × 3) | 86% | −5 pp |
| T=0.1 single agent | 86% | −5 pp |
| T=0.9 single agent | 88% | −3 pp |

**Phase 1B (400-task stratified evaluation, 8 categories × 50 tasks):**

| Category | Pass Rate | Capability |
|---|---|---|
| string | 100% | Full |
| math | 86% | High |
| data_structures | 84% | High |
| dynamic_programming | 76% | Moderate |
| parsing | 66% | Moderate |
| iterative | 43% | Low |
| recursive | 42% | Low |
| graph | 0% | None (model limit) |
| **Overall** | **61.2%** | |

Oracle gap: **12.8 pp** — split equally between exploration cost (4.1 pp), tie-breaking noise (4.7 pp), and exploitation error (4.0 pp). Chi-squared routing independence: **p = 0.396** (no specialisation).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   MCN Stack (Docker Compose)         │
│                                                     │
│  ┌──────────┐   ┌──────────────────────────────┐   │
│  │  vLLM    │   │         mcn-runner           │   │
│  │ :8000    │◄──│                              │   │
│  │ Qwen2.5- │   │  CouncilActor (Ray)          │   │
│  │ Coder-7B │   │    ├─ TribeActor × 3         │   │
│  └──────────┘   │    │    T0 / T1 / T2         │   │
│                 │    ├─ OverseerActor           │   │
│  ┌──────────┐   │    ├─ SandboxExecutor × 4    │   │
│  │  Redis   │◄──│    └─ LinUCB / GNN Router    │   │
│  │ :6379    │   │                              │   │
│  └──────────┘   │  run_live_experiment.py       │   │
│                 └──────────────────────────────┘   │
│  ┌──────────┐   ┌──────────┐                       │
│  │ ChromaDB │   │  MLflow  │                       │
│  │ :8000    │   │  :5000   │                       │
│  └──────────┘   └──────────┘                       │
└─────────────────────────────────────────────────────┘
```

### Components

| Component | File | Role |
|---|---|---|
| **CouncilActor** | `mcn/council.py` | Orchestrates routing decisions and result aggregation |
| **TribeActor** | `mcn/tribe.py` | Single LLM agent; calls vLLM with its system prompt + temperature |
| **OverseerActor** | `mcn/overseer.py` | Generates test suites: Hypothesis fuzzing, adversarial edge cases, mutation tests |
| **SandboxExecutor** | `mcn/sandbox.py` | Runs generated code in isolated subprocess with timeout |
| **LinUCB Bandit** | `mcn/util/bandit.py` | Contextual bandit, α=2.5, 18-dim context, disjoint per-arm matrices |
| **GNN Router** | `mcn/util/gnn_router.py` | 2-layer MLP (36→32→16→1), online Adam, 64-entry replay buffer |
| **PatchRegistry** | `mcn/memory.py` / `mcn/chroma_registry.py` | ChromaDB or in-memory store for few-shot hints from verified solutions |
| **StateStore** | `mcn/state.py` | Redis stream writer for all task results |
| **Tracker** | `mcn/tracking.py` | MLflow run tracking wrapper |

### Routing Context Vector (18 dimensions)

| Dims | Feature |
|---|---|
| 0–2 | Task-type one-hot |
| 3–12 | Exception type one-hot (ValueError, TypeError, …) |
| 13–17 | Z-scored execution metrics: runtime, tests_passed, tests_failed, test_count, failure_density |

---

## Prerequisites

- **Docker** with Compose v2 (`docker compose` command)
- **NVIDIA GPU** with ≥ 12 GB VRAM (tested: RTX 3080/4090, A100)
- **NVIDIA Container Toolkit** (`nvidia-docker2`)
- Python 3.12+ (for running analysis scripts on the host)
- ~25 GB free disk (model weights + Docker images)

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/DeepakEz/MCN.git
cd MCN
cp .env.example .env
# Edit .env: set MCN_VLLM_MODEL, HF_TOKEN, etc.

# 2. Start the stack (downloads model on first run — takes ~10 min)
docker compose up -d

# 3. Verify everything is healthy
docker compose ps
docker exec mcn-vllm-1 curl -s http://localhost:8000/v1/models | python -m json.tool

# 4. Run the Phase 1B stratified experiment (400 tasks, ~45 min)
docker compose run --rm mcn-runner \
  python run_live_experiment.py --stratified --tasks-per-category 50

# 5. Extract results from Redis
docker run --rm --network mcn_default \
  -v "$(pwd)/extract_redis.py:/extract_redis.py" \
  -v mcn-results:/results \
  python:3.12-slim bash -c \
    "pip install redis -q && python /extract_redis.py"

# 6. Generate academic paper + reports
pip install reportlab python-docx
python generate_paper.py
python generate_live_report.py
```

---

## Running Experiments

### Stratified Live Evaluation (Recommended)

```bash
# 400 tasks — 8 categories × 50 each (~45 min)
docker compose run --rm mcn-runner \
  python run_live_experiment.py --stratified --tasks-per-category 50

# 2000 tasks — 8 categories × 250 each (~5.5 hours, tests convergence)
docker compose run --rm mcn-runner \
  python run_live_experiment.py --stratified --tasks-per-category 250
```

### Round-Robin (original Phase 1–3 style)

```bash
# 100 tasks across all task types
docker compose run --rm mcn-runner \
  python run_live_experiment.py --num-tasks 100

# With GNN router instead of LinUCB
MCN_USE_GNN_ROUTER=true docker compose run --rm mcn-runner \
  python run_live_experiment.py --num-tasks 100

# Single-agent ablation (T=0.3)
docker compose run --rm mcn-runner \
  python run_single_agent.py --temperature 0.3 --num-tasks 100
```

### Mock Mode (no GPU required, for testing)

```bash
python run_live_experiment.py --stratified --tasks-per-category 50 --mock
```

### Detached (long runs)

```bash
# Run detached, inspect logs later
docker compose run -d mcn-runner \
  python -u run_live_experiment.py --stratified --tasks-per-category 250

# Monitor progress
docker exec mcn-redis-1 redis-cli XLEN mcn:runs

# Stream task logs
docker logs -f <container-name>
```

---

## Project Structure

```
MCN/
├── mcn/                          # Core package
│   ├── config.py                 # All configuration (MCN_* env vars)
│   ├── council.py                # CouncilActor: routing + orchestration
│   ├── tribe.py                  # TribeActor: LLM inference via vLLM
│   ├── overseer.py               # OverseerActor: test generation
│   ├── sandbox.py                # SandboxExecutor: isolated code execution
│   ├── state.py                  # Redis stream writer
│   ├── memory.py                 # In-memory patch registry
│   ├── chroma_registry.py        # ChromaDB patch registry
│   ├── tracking.py               # MLflow tracker wrapper
│   ├── protocol.py               # Shared dataclasses (Task, Result, etc.)
│   ├── util/
│   │   ├── bandit.py             # LinUCB contextual bandit
│   │   ├── gnn_router.py         # GNN router (2-layer MLP + replay)
│   │   ├── failure_encoder.py    # Exception → one-hot context
│   │   └── normalization.py      # Z-score context normalisation
│   └── tests/                    # Unit tests (pytest)
│
├── run_live_experiment.py        # Main experiment runner (live + mock)
├── run_experiment.py             # Phase 1–3 runner (historical)
├── run_phase2_experiment.py      # Phase 2 runner (historical)
├── run_oracle_experiment.py      # Oracle upper-bound experiment
├── run_single_agent.py           # Single-agent ablation runner
├── analyze_routing.py            # Routing analysis + visualisation
├── extract_redis.py              # Extract Redis stream → JSONL
├── generate_paper.py             # IEEE paper + progress report generator
├── generate_live_report.py       # Phase 1B PDF report with charts
├── generate_report.py            # Phase 1–3 report generator
│
├── categorized_runs.jsonl        # Phase 1B results (400 tasks, raw)
├── runs_n100.jsonl               # Phase 1 sample results
│
├── MCN_Academic_Paper.pdf        # Generated IEEE paper (Tables I–V)
├── MCN_Academic_Paper.docx       # Generated IEEE paper (DOCX)
├── MCN_Live_Experiment_Report.pdf# Phase 1B detailed report with charts
├── MCN_Progress_Report.pdf       # All-runs progress report
├── MCN_Progress_Report.docx      # All-runs progress report (DOCX)
│
├── docker-compose.yml            # Stack: vLLM, Redis, ChromaDB, MLflow, runner
├── Dockerfile.mcn                # mcn-runner image
├── Dockerfile.sandbox            # Sandbox isolation image
├── .env.example                  # Configuration template
├── pytest.ini                    # Test configuration
└── .github/workflows/            # CI: unit tests + integration
```

---

## Configuration Reference

Copy `.env.example` to `.env` and adjust. All variables use the `MCN_` prefix.

| Variable | Default | Description |
|---|---|---|
| `MCN_VLLM_MODEL` | `deepseek-ai/deepseek-coder-6.7b-instruct` | Model to load in vLLM |
| `MCN_VLLM_URL` | `http://vllm:8000/v1` | vLLM endpoint (use `http://localhost:8000/v1` outside Docker) |
| `MCN_NUM_TRIBES` | `3` | Number of tribe agents |
| `MCN_TRIBE_TEMPERATURE` | `0.3` | Shared temperature (homogeneous mode) |
| `MCN_TRIBE_TEMPERATURES` | `` | Per-tribe temperatures, comma-separated e.g. `0.1,0.5,0.9` |
| `MCN_BANDIT_ALPHA` | `1.5` | LinUCB exploration coefficient |
| `MCN_USE_GNN_ROUTER` | `false` | `true` to use GNN instead of LinUCB |
| `MCN_USE_REDIS` | `false` | `true` to stream results to Redis |
| `MCN_USE_CHROMADB` | `false` | `true` to use ChromaDB for patch registry |
| `MCN_USE_MLFLOW` | `false` | `true` to enable MLflow run tracking |
| `MCN_NUM_SANDBOXES` | `4` | Parallel sandbox executors |
| `MCN_SANDBOX_TIMEOUT` | `10.0` | Per-test timeout in seconds |
| `HF_TOKEN` | — | HuggingFace token (required for gated models) |

### Recommended configurations

**Phase 1B replication (homogeneous LinUCB):**
```env
MCN_VLLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
MCN_NUM_TRIBES=3
MCN_TRIBE_TEMPERATURE=0.3
MCN_BANDIT_ALPHA=2.5
MCN_USE_REDIS=true
MCN_USE_MLFLOW=true
```

**Heterogeneous GNN (Phase 3 Run 10 style):**
```env
MCN_USE_GNN_ROUTER=true
MCN_TRIBE_TEMPERATURES=0.1,0.5,0.9
MCN_USE_CHROMADB=true
```

---

## Experimental Phases

### Phase 1 (Runs 1–6): Infrastructure Validation
- **Task set:** 12 Python functions
- **Router:** GNN, then LinUCB
- **Key finding:** Three critical infrastructure defects (unbounded Hypothesis, `sys.maxsize` hang, `PATCH_MIN_ATTEMPTS=2`) caused pass-rate distortion. After fixes, GNN reached **98%**.
- **Defect:** LinUCB cold-start collapse → 100/0/0 routing.

### Phase 2 (Runs 7–8): Task Expansion
- **Task set:** Extended to 16 types (added `has_cycle`, `permutations`, `search_insert`, `unique_paths`)
- **Key finding:** 14 pp pass-rate drop from task difficulty. Three more reference-solution bugs found.

### Phase 3 (Runs 9–10 + 3 ablations): Controlled Comparison
- **Five conditions:** T=0.1, T=0.3, T=0.9 single agents + MCN-LinUCB + MCN-GNN
- **Key finding:** T=0.3 single agent (**91%**) beats all MCN variants. Temperature dominates routing.

### Phase 1B (400-task Stratified Evaluation)
- **Task set:** 8 categories × 50 tasks, live vLLM inference
- **Key finding:** Extreme category heterogeneity (0%–100%). Oracle gap **12.8 pp**, evenly split. Bandit not converged at 400 tasks.
- **Results file:** `categorized_runs.jsonl`

### Phase 1C (2000-task Scale-Up — in progress)
- **Goal:** Test whether bandit converges at 2,000 tasks (250/category × 8)
- **Hypothesis:** At 2,000 tasks, per-category UCB confidence intervals should narrow enough for reliable arm selection — *if tribes are genuinely diverse*
- **Status:** Running (`mcn-runs` Redis stream, container `mcn-mcn-runner-run-38ee170d2f1c`)

---

## Results

All raw results are stored in Redis (`mcn:runs` stream, JSON-encoded fields) and extracted to JSONL:

```bash
# Extract current Redis stream to JSONL
docker run --rm --network mcn_default \
  -v "$(pwd)/extract_redis.py:/extract_redis.py" \
  python:3.12-slim bash -c "pip install redis -q && python /extract_redis.py"
# Output: /results/categorized_runs.jsonl (via mcn-results volume)
```

### Record schema

```json
{
  "run": 42,
  "task_type": "fibonacci",
  "category": "iterative",
  "tribe_idx": 1,
  "tribe_id": "tribe_1",
  "verdict": "PASS",
  "reward": 0.964,
  "exception": null,
  "tokens": 335
}
```

### Generate analysis reports

```bash
# PDF report with 4 charts (category pass rates, bandit drift, oracle gap, tribe heatmap)
python generate_live_report.py

# IEEE paper (PDF + DOCX) with Tables I–V
python generate_paper.py

# Routing analysis plots
python analyze_routing.py --input categorized_runs.jsonl
```

---

## Reproducing the Paper

The IEEE-format paper (`MCN_Academic_Paper.pdf`) is fully generated from `generate_paper.py`. No manual editing.

```bash
# Install dependencies (host Python)
pip install reportlab python-docx

# Regenerate all four documents
python generate_paper.py
# Writes:
#   MCN_Academic_Paper.pdf   — IEEE two-column, Letter, Times-Roman
#   MCN_Academic_Paper.docx  — IEEE two-column DOCX
#   MCN_Progress_Report.pdf  — All-runs technical log
#   MCN_Progress_Report.docx — All-runs technical log (DOCX)
```

All content (sections, tables, abstract, keywords) is defined as Python constants at the top of `generate_paper.py` — easy to update when new experimental results arrive.

---

## Monitoring

```bash
# Task progress (count)
docker exec mcn-redis-1 redis-cli XLEN mcn:runs

# Pass rate (live)
docker exec mcn-redis-1 redis-cli XRANGE mcn:runs - + | \
  python -c "
import sys, json
lines = sys.stdin.read().split('\n')
vals = [json.loads(l.split()[-1]) for l in lines if '\"verdict\"' in l]
n = len(vals); p = sum(1 for v in vals if v == 'PASS')
print(f'{n} done  |  {p} passed  |  {100*p/n:.1f}%' if n else 'no data')
"

# Container logs (streaming)
docker logs -f mcn-mcn-runner-run-38ee170d2f1c

# MLflow UI
open http://localhost:5000
```

---

## Running Tests

```bash
# Host (requires mcn package to be importable)
pip install pytest numpy
pytest mcn/tests/ -v

# Inside Docker
docker compose run --rm mcn-runner pytest mcn/tests/ -v
```

---

## Citation

```bibtex
@misc{mcn2026,
  title  = {Mycelial Council Network: Temperature Dominates Routing in
             Multi-Agent LLM Code Synthesis --- A Longitudinal Study
             with Category-Level Analysis},
  author = {MCN Research Initiative},
  year   = {2026},
  note   = {Anonymous Submission, February 2026}
}
```

---

## License

Research code — see `LICENSE` if present, otherwise all rights reserved.
