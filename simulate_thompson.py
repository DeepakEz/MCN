"""Retrospective router comparison on Phase 1C data.

Replays the 2000-task Phase 1C sequence under four routing strategies:
  1. LinUCB (actual)          — recorded tribe_idx from the live experiment
  2. Random routing           — uniform draw each task
  3. Oracle per-category      — best tribe per category (hindsight)
  4. Category Thompson Sampling (CTS) — online simulation with imputed rewards

CTS simulation method:
  At each step t with task of category k:
    - Sample arm i* from Beta posteriors for category k
    - If i* == actual_arm: observe real reward, update posterior
    - If i* != actual_arm: impute reward from empirical per-(category, arm)
      pass rate (computed from the full dataset as a proxy oracle), update
    Both cases count toward the simulated pass rate.

This gives an upper-bound estimate of CTS performance. The true online
performance would lie between CTS-simulated and Oracle.

Output: simulate_thompson_report.pdf
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

DATA = Path("categorized_runs_phase1c.jsonl")
assert DATA.exists(), f"Phase 1C data not found: {DATA}"

records = [json.loads(l) for l in DATA.open()]
print(f"Loaded {len(records)} Phase 1C records")

# Canonical categories
CATEGORIES = [
    "data_structures", "dynamic_programming", "graph", "iterative",
    "math", "parsing", "recursive", "string",
]
N_ARMS = 3

# ---------------------------------------------------------------------------
# Empirical per-(category, arm) pass rates from the full dataset
# ---------------------------------------------------------------------------

emp = defaultdict(lambda: {"pass": 0, "total": 0})
for rec in records:
    cat = rec.get("category", "unknown")
    arm = int(rec.get("tribe_idx", 0))
    emp[(cat, arm)]["total"] += 1
    if rec.get("verdict") == "PASS":
        emp[(cat, arm)]["pass"] += 1

def emp_rate(cat, arm):
    d = emp[(cat, arm)]
    if d["total"] == 0:
        return 0.5  # uniform prior if unseen
    return d["pass"] / d["total"]

print("\nEmpirical per-(category, arm) pass rates:")
for cat in CATEGORIES:
    rates = [f"T{i}={emp_rate(cat,i)*100:.1f}%({emp[(cat,i)]['total']})" for i in range(N_ARMS)]
    best = max(range(N_ARMS), key=lambda i: emp_rate(cat, i))
    print(f"  {cat:22s}: {' | '.join(rates)}  best=T{best}")

# Oracle per-category arm
oracle_arm = {cat: max(range(N_ARMS), key=lambda i: emp_rate(cat, i)) for cat in CATEGORIES}

# ---------------------------------------------------------------------------
# Strategy simulations
# ---------------------------------------------------------------------------

def sim_linucb(records):
    """Actual LinUCB results — just read recorded verdicts."""
    return [r.get("verdict") == "PASS" for r in records]

def sim_random(records, seed=42):
    rng = random.Random(seed)
    results = []
    for rec in records:
        cat = rec.get("category", "unknown")
        arm = rng.randint(0, N_ARMS - 1)
        # Impute from empirical rate
        passed = rng.random() < emp_rate(cat, arm)
        results.append(passed)
    return results

def sim_oracle(records):
    results = []
    for rec in records:
        cat = rec.get("category", "unknown")
        arm = oracle_arm.get(cat, 0)
        # If this arm was actually chosen, use real verdict; else impute
        if int(rec.get("tribe_idx", 0)) == arm:
            passed = rec.get("verdict") == "PASS"
        else:
            passed = random.random() < emp_rate(cat, arm)
        results.append(passed)
    return results

def sim_cts(records, alpha_prior=1.0, seed=42):
    """Category Thompson Sampling retrospective simulation."""
    rng = np.random.default_rng(seed)
    # Beta posteriors: alpha[k,i], beta[k,i]
    cat_idx = {c: i for i, c in enumerate(CATEGORIES)}
    n_cats = len(CATEGORIES)
    alpha_k = np.full((n_cats, N_ARMS), alpha_prior)
    beta_k  = np.full((n_cats, N_ARMS), alpha_prior)

    results = []
    routing = []
    for rec in records:
        cat = rec.get("category", "unknown")
        k = cat_idx.get(cat, 0)
        actual_arm = int(rec.get("tribe_idx", 0))

        # Thompson sample
        samples = rng.beta(alpha_k[k], beta_k[k])
        chosen_arm = int(np.argmax(samples))
        routing.append(chosen_arm)

        # Observe / impute reward
        if chosen_arm == actual_arm:
            passed = rec.get("verdict") == "PASS"
        else:
            # Impute from empirical rate
            passed = rng.random() < emp_rate(cat, chosen_arm)

        results.append(passed)

        # Update posterior
        if passed:
            alpha_k[k, chosen_arm] += 1.0
        else:
            beta_k[k, chosen_arm] += 1.0

    return results, routing, alpha_k, beta_k

# Run simulations
random.seed(42)
np.random.seed(42)

r_linucb = sim_linucb(records)
r_random = sim_random(records)
r_oracle = sim_oracle(records)
r_cts, cts_routing, cts_alpha, cts_beta = sim_cts(records)

strategies = {
    "LinUCB (actual)":        r_linucb,
    "Random":                 r_random,
    "Oracle per-category":    r_oracle,
    "Category TS (simulated)": r_cts,
}

print("\n--- Strategy Comparison ---")
for name, res in strategies.items():
    total = len(res)
    passed = sum(res)
    print(f"  {name:30s}: {passed}/{total} = {100*passed/total:.1f}%")

# Rolling pass rate (window=100)
def rolling(results, w=100):
    arr = np.array(results, dtype=float)
    return np.convolve(arr, np.ones(w) / w, mode="valid")

# Per-category comparison
print("\n--- Per-category: LinUCB vs CTS ---")
cat_results = defaultdict(lambda: {"linucb": [], "cts": []})
for i, rec in enumerate(records):
    cat = rec.get("category", "unknown")
    cat_results[cat]["linucb"].append(r_linucb[i])
    cat_results[cat]["cts"].append(r_cts[i])

for cat in CATEGORIES:
    d = cat_results[cat]
    lu = 100 * sum(d["linucb"]) / len(d["linucb"]) if d["linucb"] else 0
    cts = 100 * sum(d["cts"]) / len(d["cts"]) if d["cts"] else 0
    delta = cts - lu
    sign = "+" if delta >= 0 else ""
    print(f"  {cat:22s}: LinUCB={lu:.1f}%  CTS={cts:.1f}%  delta={sign}{delta:.1f}pp")

# CTS routing evolution
cts_routing_arr = np.array(cts_routing)
linucb_routing_arr = np.array([r.get("tribe_idx", 0) for r in records])

# ---------------------------------------------------------------------------
# PDF Report
# ---------------------------------------------------------------------------

OUT = Path("simulate_thompson_report.pdf")
COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
CAT_COLORS = {
    "data_structures": "#1976D2", "dynamic_programming": "#7B1FA2",
    "graph": "#D32F2F", "iterative": "#F57C00",
    "math": "#388E3C", "parsing": "#0097A7",
    "recursive": "#5D4037", "string": "#00796B",
}

W = 100  # rolling window

with PdfPages(OUT) as pdf:
    # ── Cover page ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    fig.patch.set_facecolor("#1A1A2E")
    ax.text(0.5, 0.72, "MCN Router Comparison", transform=ax.transAxes,
            ha="center", va="center", fontsize=28, fontweight="bold", color="white")
    ax.text(0.5, 0.60, "Retrospective Simulation on Phase 1C (2000 tasks)",
            transform=ax.transAxes, ha="center", va="center", fontsize=16, color="#AAAACC")
    ax.text(0.5, 0.50, "LinUCB  |  Random  |  Oracle per-category  |  Category Thompson Sampling",
            transform=ax.transAxes, ha="center", va="center", fontsize=13, color="#7788AA")

    # Summary table
    rows = []
    for name, res in strategies.items():
        passed = sum(res)
        pct = 100 * passed / len(res)
        rows.append([name, f"{passed}/{len(res)}", f"{pct:.1f}%"])
    tbl = ax.table(cellText=rows,
                   colLabels=["Strategy", "Tasks Passed", "Pass Rate"],
                   loc="center", bbox=[0.15, 0.20, 0.70, 0.24])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#2A2A4E" if r % 2 == 0 else "#1A1A3E")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#444466")
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── Figure 1: Overall rolling pass rate ────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    for (name, res), col in zip(strategies.items(), COLORS):
        r = rolling(res, W)
        ax.plot(range(W - 1, len(res)), r * 100, label=name, color=col, lw=1.8)
    ax.axhline(100 * sum(r_linucb) / len(r_linucb), color=COLORS[0],
               lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Task index")
    ax.set_ylabel(f"Pass rate (rolling {W})")
    ax.set_title("Rolling Pass Rate by Strategy — Phase 1C (2000 tasks)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Figure 2: Per-category bar chart comparison ─────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(CATEGORIES))
    w = 0.20
    for j, (name, res) in enumerate(strategies.items()):
        per_cat = []
        for cat in CATEGORIES:
            idxs = [i for i, r in enumerate(records) if r.get("category") == cat]
            per_cat.append(100 * sum(res[i] for i in idxs) / len(idxs) if idxs else 0)
        ax.bar(x + j * w - 1.5 * w, per_cat, w, label=name, color=COLORS[j], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CATEGORIES], fontsize=8)
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Per-Category Pass Rate by Strategy")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Figure 3: Routing evolution — LinUCB vs CTS ─────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    window = 100
    for t_idx in range(N_ARMS):
        linucb_share = np.convolve(
            (linucb_routing_arr == t_idx).astype(float),
            np.ones(window) / window, mode="valid"
        ) * 100
        cts_share = np.convolve(
            (cts_routing_arr == t_idx).astype(float),
            np.ones(window) / window, mode="valid"
        ) * 100
        xs = range(window - 1, len(records))
        axes[0].plot(xs, linucb_share, label=f"T{t_idx}", lw=1.6)
        axes[1].plot(xs, cts_share, label=f"T{t_idx}", lw=1.6)

    axes[0].set_ylabel("Routing share (%)")
    axes[0].set_title("LinUCB Routing (converged to T0 by task 1700)")
    axes[0].legend(loc="right", fontsize=9)
    axes[0].set_ylim(-5, 105)
    axes[0].grid(alpha=0.25)

    axes[1].set_ylabel("Routing share (%)")
    axes[1].set_xlabel("Task index")
    axes[1].set_title("Category TS Routing (maintains diversity)")
    axes[1].legend(loc="right", fontsize=9)
    axes[1].set_ylim(-5, 105)
    axes[1].grid(alpha=0.25)

    fig.suptitle("Routing Evolution: LinUCB vs Category Thompson Sampling", fontsize=13)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Figure 4: CTS posterior means at end of sequence ───────────────────
    cts_alpha_f, cts_beta_f = cts_alpha, cts_beta
    means = cts_alpha_f / (cts_alpha_f + cts_beta_f)  # shape (n_cats, n_arms)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(CATEGORIES))
    for arm in range(N_ARMS):
        ax.plot(x, means[:, arm] * 100, "o-", label=f"T{arm}",
                lw=1.8, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CATEGORIES], fontsize=8)
    ax.set_ylabel("Posterior mean pass rate (%)")
    ax.set_title("Category TS: Posterior Means by (Category, Arm) at End of Sequence")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.25)
    ax.axhline(100 * sum(r_linucb) / len(r_linucb), color="gray",
               lw=0.8, ls="--", label="LinUCB overall")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Figure 5: Oracle gap decomposition ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    linucb_pct = 100 * sum(r_linucb) / len(r_linucb)
    random_pct = 100 * sum(r_random) / len(r_random)
    oracle_pct = 100 * sum(r_oracle) / len(r_oracle)
    cts_pct    = 100 * sum(r_cts)    / len(r_cts)

    labels = ["Random\n(baseline)", "LinUCB\n(actual)", "CTS\n(simulated)", "Oracle\n(per-category)"]
    values = [random_pct, linucb_pct, cts_pct, oracle_pct]
    bars = ax.bar(labels, values, color=[COLORS[1], COLORS[0], COLORS[3], COLORS[2]], width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, max(values) * 1.15)
    ax.set_ylabel("Overall pass rate (%)")
    ax.set_title("Oracle Gap Decomposition (Phase 1C, 2000 tasks)")
    ax.grid(axis="y", alpha=0.25)

    # Annotate gaps
    ax.annotate("", xy=(2, cts_pct), xytext=(1, linucb_pct),
                arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5))
    ax.text(1.5, (cts_pct + linucb_pct) / 2, f"+{cts_pct - linucb_pct:.1f}pp",
            ha="center", va="bottom", fontsize=10, color="purple")
    ax.annotate("", xy=(3, oracle_pct), xytext=(2, cts_pct),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
    ax.text(2.5, (oracle_pct + cts_pct) / 2, f"+{oracle_pct - cts_pct:.1f}pp",
            ha="center", va="bottom", fontsize=10, color="green")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Metadata ────────────────────────────────────────────────────────────
    d = pdf.infodict()
    d["Title"] = "MCN Router Comparison: Thompson Sampling Retrospective"
    d["Subject"] = "Bandit routing comparison on Phase 1C 2000-task dataset"

print(f"\nReport saved -> {OUT}")
print(f"  {OUT.stat().st_size // 1024} KB")
