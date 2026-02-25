"""Router comparison report: LinUCB (Phase 1C) vs GNN (Phase 1D).

Run after extract_redis_gnn.py has produced categorized_runs_gnn.jsonl.
Generates MCN_Router_Comparison.pdf with:
  - Overall pass rate comparison
  - Per-category pass rate comparison (bar chart)
  - Routing evolution (LinUCB vs GNN)
  - Tribe utilization comparison
  - Oracle gap analysis

Usage:
    python compare_routers_report.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

F_1C = Path("categorized_runs_phase1c.jsonl")
F_GNN = Path("categorized_runs_gnn.jsonl")

if not F_GNN.exists():
    print(f"ERROR: {F_GNN} not found. Run extract_redis_gnn.py first.")
    raise SystemExit(1)

recs_1c  = [json.loads(l) for l in F_1C.open()]
recs_gnn = [json.loads(l) for l in F_GNN.open()]
print(f"LinUCB (1C): {len(recs_1c)} records")
print(f"GNN:         {len(recs_gnn)} records")

CATEGORIES = [
    "data_structures", "dynamic_programming", "graph", "iterative",
    "math", "parsing", "recursive", "string",
]
N_ARMS = 3

def pass_rate(recs):
    if not recs:
        return 0.0
    return sum(1 for r in recs if r.get("verdict") == "PASS") / len(recs)

def per_category(recs, categories):
    return {
        cat: pass_rate([r for r in recs if r.get("category") == cat]) * 100
        for cat in categories
    }

def per_tribe(recs):
    return {
        i: {
            "count": sum(1 for r in recs if int(r.get("tribe_idx", 0)) == i),
            "pass":  sum(1 for r in recs if int(r.get("tribe_idx", 0)) == i and r.get("verdict") == "PASS"),
        }
        for i in range(N_ARMS)
    }

def rolling(recs, w=100):
    arr = np.array([r.get("verdict") == "PASS" for r in recs], dtype=float)
    return np.convolve(arr, np.ones(w) / w, mode="valid")

# ---------------------------------------------------------------------------
# Compute stats
# ---------------------------------------------------------------------------

overall_1c  = pass_rate(recs_1c)  * 100
overall_gnn = pass_rate(recs_gnn) * 100

cat_1c  = per_category(recs_1c,  CATEGORIES)
cat_gnn = per_category(recs_gnn, CATEGORIES)

tribe_1c  = per_tribe(recs_1c)
tribe_gnn = per_tribe(recs_gnn)

print(f"\nOverall: LinUCB={overall_1c:.1f}%  GNN={overall_gnn:.1f}%  delta={overall_gnn-overall_1c:+.1f}pp")
print("\nPer-category:")
for cat in CATEGORIES:
    d = cat_gnn[cat] - cat_1c[cat]
    sign = "+" if d >= 0 else ""
    print(f"  {cat:22s}: LinUCB={cat_1c[cat]:.1f}%  GNN={cat_gnn[cat]:.1f}%  {sign}{d:.1f}pp")

print("\nPer-tribe:")
for arm in range(N_ARMS):
    t1c  = tribe_1c[arm]
    tgnn = tribe_gnn[arm]
    pr1c  = 100 * t1c["pass"] / t1c["count"] if t1c["count"] else 0
    prgnn = 100 * tgnn["pass"] / tgnn["count"] if tgnn["count"] else 0
    n1c   = t1c["count"]
    ngnn  = tgnn["count"]
    print(f"  T{arm}: LinUCB={n1c}({n1c*100//len(recs_1c)}%,{pr1c:.0f}%pass)  "
          f"GNN={ngnn}({ngnn*100//len(recs_gnn)}%,{prgnn:.0f}%pass)")

# ---------------------------------------------------------------------------
# PDF Report
# ---------------------------------------------------------------------------

OUT = Path("MCN_Router_Comparison.pdf")
COLORS = {"linucb": "#2196F3", "gnn": "#FF5722"}
TRIBE_COLORS = ["#1976D2", "#E53935", "#43A047"]

with PdfPages(OUT) as pdf:

    # Cover
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    fig.patch.set_facecolor("#0D1117")
    ax.text(0.5, 0.80, "MCN Router Comparison", transform=ax.transAxes,
            ha="center", fontsize=28, fontweight="bold", color="white")
    ax.text(0.5, 0.70, "LinUCB (Phase 1C) vs GNN Router (Phase 1D)",
            transform=ax.transAxes, ha="center", fontsize=16, color="#AAAACC")
    ax.text(0.5, 0.60, f"LinUCB: {len(recs_1c)} tasks  |  GNN: {len(recs_gnn)} tasks",
            transform=ax.transAxes, ha="center", fontsize=12, color="#7788AA")

    rows = [
        ["LinUCB", f"{sum(r.get('verdict')=='PASS' for r in recs_1c)}/{len(recs_1c)}", f"{overall_1c:.1f}%"],
        ["GNN",    f"{sum(r.get('verdict')=='PASS' for r in recs_gnn)}/{len(recs_gnn)}", f"{overall_gnn:.1f}%"],
    ]
    tbl = ax.table(cellText=rows, colLabels=["Router", "Tasks Passed", "Pass Rate"],
                   loc="center", bbox=[0.20, 0.35, 0.60, 0.16])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1A1A2E" if r % 2 == 0 else "#0D1117")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#333355")
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Per-category bar chart
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(CATEGORIES))
    w = 0.35
    bars1 = ax.bar(x - w/2, [cat_1c[c] for c in CATEGORIES],  w, label="LinUCB", color=COLORS["linucb"], alpha=0.85)
    bars2 = ax.bar(x + w/2, [cat_gnn[c] for c in CATEGORIES], w, label="GNN",    color=COLORS["gnn"],    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CATEGORIES], fontsize=8)
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Per-Category Pass Rate: LinUCB vs GNN Router")
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    for b1, b2, cat in zip(bars1, bars2, CATEGORIES):
        delta = cat_gnn[cat] - cat_1c[cat]
        clr = "#4CAF50" if delta > 0 else "#F44336" if delta < 0 else "gray"
        sign = "+" if delta >= 0 else ""
        ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.5,
                f"{sign}{delta:.0f}", ha="center", va="bottom", fontsize=8, color=clr)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Rolling pass rate
    fig, ax = plt.subplots(figsize=(13, 5))
    W = 100
    r1c  = rolling(recs_1c,  W)
    rgnn = rolling(recs_gnn, W)
    ax.plot(range(W-1, len(recs_1c)),  r1c  * 100, label="LinUCB", color=COLORS["linucb"], lw=1.8)
    ax.plot(range(W-1, len(recs_gnn)), rgnn * 100, label="GNN",    color=COLORS["gnn"],    lw=1.8)
    ax.axhline(overall_1c,  color=COLORS["linucb"], lw=0.8, ls=":", alpha=0.6)
    ax.axhline(overall_gnn, color=COLORS["gnn"],    lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Task index")
    ax.set_ylabel(f"Pass rate (rolling {W})")
    ax.set_title("Rolling Pass Rate: LinUCB vs GNN Router")
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Routing evolution
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    W2 = 100
    for arm in range(N_ARMS):
        arr1c  = np.array([int(r.get("tribe_idx", 0)) == arm for r in recs_1c],  dtype=float)
        arrgnn = np.array([int(r.get("tribe_idx", 0)) == arm for r in recs_gnn], dtype=float)
        s1c  = np.convolve(arr1c,  np.ones(W2)/W2, mode="valid") * 100
        sgnn = np.convolve(arrgnn, np.ones(W2)/W2, mode="valid") * 100
        axes[0].plot(range(W2-1, len(recs_1c)),  s1c,  label=f"T{arm}", color=TRIBE_COLORS[arm], lw=1.6)
        axes[1].plot(range(W2-1, len(recs_gnn)), sgnn, label=f"T{arm}", color=TRIBE_COLORS[arm], lw=1.6)

    axes[0].set_title("LinUCB Routing (Phase 1C)")
    axes[0].set_ylabel("Routing share (%)")
    axes[0].legend(loc="right", fontsize=9)
    axes[0].set_ylim(-5, 105)
    axes[0].grid(alpha=0.25)

    axes[1].set_title("GNN Router Routing (Phase 1D)")
    axes[1].set_ylabel("Routing share (%)")
    axes[1].set_xlabel("Task index")
    axes[1].legend(loc="right", fontsize=9)
    axes[1].set_ylim(-5, 105)
    axes[1].grid(alpha=0.25)

    fig.suptitle("Routing Evolution: LinUCB vs GNN", fontsize=13)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    d = pdf.infodict()
    d["Title"] = "MCN Router Comparison: LinUCB vs GNN"

print(f"\nReport saved -> {OUT}")
print(f"  {OUT.stat().st_size // 1024} KB")
