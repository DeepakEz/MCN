"""
Phase 2 Analysis: Heterogeneous Temperature Tribes + CTS Router

Tribes: T0=0.1 (deterministic), T1=0.5 (balanced), T2=0.9 (creative)
Router: CategoryThompsonSampling

Key questions:
  1. Does routing converge to a single tribe, or maintain diversity?
  2. Does each category develop a preferred temperature?
  3. What is the oracle gap vs. actual CTS performance?
  4. How does Phase 2 compare to Phase 1C (homogeneous + LinUCB)?

Requires: categorized_runs_phase2.jsonl (produced by parse_phase2_dump.py)
Produces: MCN_Phase2_Report.pdf
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph, Spacer, SimpleDocTemplate, Table, TableStyle, HRFlowable,
)
from reportlab.lib import colors as rl_colors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PHASE1C_FILE = Path("categorized_runs_phase1c.jsonl")
PHASE2_FILE  = Path("categorized_runs_phase2.jsonl")
OUTPUT_PDF   = Path("MCN_Phase2_Report.pdf")

TRIBE_TEMPS = {0: 0.1, 1: 0.5, 2: 0.9}
CATEGORIES  = [
    "string", "math", "data_structures", "dynamic_programming",
    "parsing", "iterative", "recursive", "graph",
]
CAT_COLORS = {
    "string": "#4CAF50", "math": "#2196F3", "data_structures": "#9C27B0",
    "dynamic_programming": "#FF9800", "parsing": "#F44336",
    "iterative": "#00BCD4", "recursive": "#795548", "graph": "#607D8B",
}
TRIBE_COLORS = ["#1565C0", "#2E7D32", "#B71C1C"]  # T0/T1/T2

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_runs(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[warn] {path} not found — returning empty")
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _is_pass(r: dict) -> bool:
    """Normalise pass/fail across Phase 1C (verdict field) and Phase 2 (passed field)."""
    if "passed" in r:
        return bool(r["passed"])
    return r.get("verdict") == "PASS"


def compute_stats(runs: list[dict]) -> dict:
    """Aggregate stats for a run set."""
    total = len(runs)
    passed = sum(1 for r in runs if _is_pass(r))

    # Per-category
    cat_total:  dict[str, int] = defaultdict(int)
    cat_pass:   dict[str, int] = defaultdict(int)

    # Per-tribe
    tribe_total: dict[int, int] = defaultdict(int)
    tribe_pass:  dict[int, int] = defaultdict(int)

    # Per (category, tribe)
    cat_tribe_total: dict[tuple, int] = defaultdict(int)
    cat_tribe_pass:  dict[tuple, int] = defaultdict(int)

    # Routing share over time (for convergence plot)
    routing_time: list[int] = []

    for r in runs:
        cat   = r.get("category", "unknown")
        t_idx = int(r.get("tribe_idx", 0))
        ok    = _is_pass(r)

        cat_total[cat]  += 1
        cat_pass[cat]   += ok
        tribe_total[t_idx] += 1
        tribe_pass[t_idx]  += ok
        cat_tribe_total[(cat, t_idx)] += 1
        cat_tribe_pass[(cat, t_idx)]  += ok
        routing_time.append(t_idx)

    # Oracle: best tribe per category
    oracle_wins = 0
    oracle_detail: dict[str, tuple] = {}
    for cat in CATEGORIES:
        best_rate = -1.0
        best_tribe = 0
        for t in range(3):
            n = cat_tribe_total.get((cat, t), 0)
            w = cat_tribe_pass.get((cat, t), 0)
            rate = w / n if n > 0 else 0.0
            if rate > best_rate:
                best_rate, best_tribe = rate, t
        oracle_wins += cat_pass.get(cat, 0)  # approximate: total cat passes
        # Actual oracle: for each task, would best tribe have passed?
        # Use imputed per-task reward from aggregate rates
        oracle_detail[cat] = (best_tribe, best_rate)

    # Compute actual oracle (per-task counterfactual from aggregate rates)
    actual_oracle = 0
    for cat in CATEGORIES:
        _, best_rate = oracle_detail[cat]
        n = cat_total.get(cat, 0)
        actual_oracle += round(best_rate * n)

    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "cat_total": dict(cat_total),
        "cat_pass": dict(cat_pass),
        "tribe_total": dict(tribe_total),
        "tribe_pass": dict(tribe_pass),
        "cat_tribe_total": {str(k): v for k, v in cat_tribe_total.items()},
        "cat_tribe_pass":  {str(k): v for k, v in cat_tribe_pass.items()},
        "oracle_passes": actual_oracle,
        "oracle_rate": actual_oracle / total if total else 0.0,
        "oracle_gap": (actual_oracle / total - passed / total) if total else 0.0,
        "routing_time": routing_time,
        "oracle_detail": oracle_detail,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_routing_convergence(runs2: list[dict], runs1c: list[dict]) -> str:
    """Rolling 200-task routing share for each tribe — Phase 2 vs Phase 1C."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    def _rolling_shares(runs, window=200):
        n = len(runs)
        shares = {0: [], 1: [], 2: []}
        for i in range(window, n + 1):
            chunk = runs[i - window:i]
            for t in range(3):
                shares[t].append(sum(1 for r in chunk if int(r.get("tribe_idx", 0)) == t) / window)
        return shares, list(range(window, n + 1))

    for ax, runs, title in [
        (axes[0], runs1c, "Phase 1C: LinUCB + Homogeneous (T=0.3)"),
        (axes[1], runs2,  "Phase 2: CTS + Heterogeneous (T=0.1/0.5/0.9)"),
    ]:
        shares, xs = _rolling_shares(runs)
        for t in range(3):
            lbl = f"T{t} (temp={TRIBE_TEMPS[t]})" if runs is runs2 else f"T{t}"
            ax.plot(xs, shares[t], color=TRIBE_COLORS[t], lw=1.8, label=lbl)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("Task index")
        ax.set_ylabel("Routing share (rolling 200)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Router Convergence: Homogeneous vs Heterogeneous Tribes", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = "/tmp/fig_routing_conv.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def fig_per_category_tribe(stats2: dict) -> str:
    """Stacked bar: pass rate by (category, tribe) for Phase 2."""
    cats = [c for c in CATEGORIES if stats2["cat_total"].get(c, 0) > 0]
    x = np.arange(len(cats))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    for j, t in enumerate([0, 1, 2]):
        rates = []
        for cat in cats:
            key = str((cat, t))
            n = stats2["cat_tribe_total"].get(key, stats2["cat_tribe_total"].get(f"('{cat}', {t})", 0))
            w = stats2["cat_tribe_pass"].get(key, stats2["cat_tribe_pass"].get(f"('{cat}', {t})", 0))
            rates.append(w / n if n > 0 else 0.0)
        bars = ax.bar(x + j * width, rates, width, label=f"T{t} (T={TRIBE_TEMPS[t]})",
                      color=TRIBE_COLORS[j], alpha=0.85)
        for bar, rate in zip(bars, rates):
            if rate > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{rate:.0%}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Pass rate")
    ax.set_title("Pass Rate by Category and Tribe Temperature (Phase 2)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = "/tmp/fig_cat_tribe.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def fig_oracle_gap_comparison(stats1c: dict, stats2: dict) -> str:
    """Bar: overall pass rate vs oracle — Phase 1C vs Phase 2."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels  = ["Phase 1C\n(LinUCB+Homo)", "Phase 2\n(CTS+Hetero)"]
    actuals = [stats1c["pass_rate"], stats2["pass_rate"]]
    oracles = [stats1c["oracle_rate"], stats2["oracle_rate"]]

    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, actuals, w, label="Actual router", color=["#1565C0", "#B71C1C"], alpha=0.85)
    b2 = ax.bar(x + w / 2, oracles, w, label="Oracle (per-cat best tribe)", color=["#90CAF9", "#FFCDD2"], alpha=0.85)

    for bar, v in zip(list(b1) + list(b2), actuals + oracles):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Oracle gap annotations
    for i in range(2):
        gap = oracles[i] - actuals[i]
        y   = max(actuals[i], oracles[i]) + 0.03
        ax.annotate(f"gap={gap:.1%}", xy=(x[i], y), ha="center", fontsize=9,
                    color="#555", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Pass rate")
    ax.set_title("Oracle Gap: Homogeneous vs Heterogeneous Tribes", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = "/tmp/fig_oracle_gap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def fig_preferred_temps(stats2: dict) -> str:
    """Heatmap: pass rate per (category, tribe) — shows temperature preferences."""
    cats = CATEGORIES
    data = np.zeros((3, len(cats)))
    for j, t in enumerate(range(3)):
        for i, cat in enumerate(cats):
            key_a = f"('{cat}', {t})"
            key_b = str((cat, t))
            n = stats2["cat_tribe_total"].get(key_a, stats2["cat_tribe_total"].get(key_b, 0))
            w = stats2["cat_tribe_pass"].get(key_a, stats2["cat_tribe_pass"].get(key_b, 0))
            data[j, i] = w / n if n > 0 else np.nan

    fig, ax = plt.subplots(figsize=(11, 3.5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"T{t} (temp={TRIBE_TEMPS[t]})" for t in range(3)], fontsize=9)
    plt.colorbar(im, ax=ax, label="Pass rate")

    for j in range(3):
        for i in range(len(cats)):
            val = data[j, i]
            if not np.isnan(val):
                ax.text(i, j, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color="black" if 0.3 < val < 0.7 else "white")

    ax.set_title("Temperature Preference Heatmap: Pass Rate by (Category, Tribe)", fontweight="bold")
    plt.tight_layout()
    out = "/tmp/fig_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

def build_report(stats1c: dict, stats2: dict, runs1c: list, runs2: list) -> None:
    doc  = SimpleDocTemplate(str(OUTPUT_PDF), pagesize=letter,
                             leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                             topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    S    = getSampleStyleSheet()
    body = []

    def h1(t):  body.append(Paragraph(t, S["h1"]));  body.append(Spacer(1, 6))
    def h2(t):  body.append(Paragraph(t, S["h2"]));  body.append(Spacer(1, 4))
    def p(t):   body.append(Paragraph(t, S["Normal"])); body.append(Spacer(1, 6))
    def hr():   body.append(HRFlowable(width="100%", thickness=1, color=rl_colors.lightgrey)); body.append(Spacer(1, 6))
    def img(path, w=6.5):
        from reportlab.platypus import Image as RLImage
        body.append(RLImage(path, width=w * inch, height=3.5 * inch))
        body.append(Spacer(1, 8))

    def table(caption, header, rows, col_widths=None):
        body.append(Paragraph(f"<b>{caption}</b>", S["Normal"]))
        body.append(Spacer(1, 4))
        data = [header] + rows
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#37474F")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#F5F5F5")]),
            ("GRID",       (0, 0), (-1, -1), 0.5, rl_colors.lightgrey),
            ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        body.append(t)
        body.append(Spacer(1, 10))

    # ---- Title ----
    h1("MCN Phase 2: Heterogeneous Tribes — CTS Router Report")
    p(f"<b>Phase 2 setup:</b> T0=temp 0.1 (deterministic), T1=temp 0.5 (balanced), T2=temp 0.9 (creative). "
      f"Router: CategoryThompsonSampling. Tasks: {stats2['total']} (8×250 stratified). "
      f"Baseline: Phase 1C LinUCB + homogeneous tribes ({stats1c['total']} tasks).")
    hr()

    # ---- Summary table ----
    h2("1. Overall Results")
    table(
        "Experiment Comparison",
        ["Metric", "Phase 1C (LinUCB+Homo)", "Phase 2 (CTS+Hetero)", "Delta"],
        [
            ["Overall pass rate",
             f"{stats1c['pass_rate']:.1%}", f"{stats2['pass_rate']:.1%}",
             f"{(stats2['pass_rate'] - stats1c['pass_rate']) * 100:+.1f} pp"],
            ["Oracle (per-cat best)",
             f"{stats1c['oracle_rate']:.1%}", f"{stats2['oracle_rate']:.1%}",
             f"{(stats2['oracle_rate'] - stats1c['oracle_rate']) * 100:+.1f} pp"],
            ["Oracle gap",
             f"{stats1c['oracle_gap']:.1%}", f"{stats2['oracle_gap']:.1%}",
             f"{(stats2['oracle_gap'] - stats1c['oracle_gap']) * 100:+.1f} pp"],
        ],
        col_widths=[2.5 * inch, 1.8 * inch, 1.8 * inch, 1.0 * inch],
    )

    # ---- Routing convergence ----
    h2("2. Router Convergence")
    img(fig_routing_convergence(runs2, runs1c), w=7.0)
    p("Phase 1C LinUCB converges to T0 at ~task 1700 (55% share). "
      "Phase 2 CTS maintains per-category Beta posteriors — expect distributed routing if "
      "tribe temperatures create genuine category-level performance differences.")
    hr()

    # ---- Per-category tribe breakdown ----
    h2("3. Temperature Preference by Category")
    img(fig_preferred_temps(stats2), w=7.0)
    p("Green cells indicate high pass rate for that (category, temperature) combination. "
      "If routing is learning correctly, CTS should route high-confidence categories "
      "to the temperature with the highest per-category pass rate.")

    img(fig_per_category_tribe(stats2), w=7.5)

    # ---- Oracle gap comparison ----
    h2("4. Oracle Gap: Homogeneous vs Heterogeneous")
    img(fig_oracle_gap_comparison(stats1c, stats2), w=6.0)
    p("A larger oracle gap in Phase 2 indicates that tribe diversity creates a real "
      "routing opportunity. A smaller gap means CTS is successfully exploiting it.")
    hr()

    # ---- Per-category table ----
    h2("5. Per-Category Results (Phase 2)")
    rows = []
    for cat in CATEGORIES:
        n   = stats2["cat_total"].get(cat, 0)
        w   = stats2["cat_pass"].get(cat, 0)
        rate = w / n if n > 0 else 0.0
        best_t, best_r = stats2["oracle_detail"].get(cat, (0, 0.0))
        gap = best_r - rate
        rows.append([cat, str(n), f"{rate:.0%}", f"T{best_t} (T={TRIBE_TEMPS[best_t]})",
                     f"{best_r:.0%}", f"{gap:.1%}"])
    table(
        "Per-Category Pass Rate vs Oracle",
        ["Category", "N", "CTS Rate", "Best Tribe", "Oracle Rate", "Gap"],
        rows,
        col_widths=[1.8 * inch, 0.5 * inch, 0.9 * inch, 1.5 * inch, 1.0 * inch, 0.7 * inch],
    )

    # ---- Per-tribe summary ----
    h2("6. Per-Tribe Performance")
    tribe_rows = []
    for t in range(3):
        n    = stats2["tribe_total"].get(t, 0)
        w    = stats2["tribe_pass"].get(t, 0)
        rate = w / n if n > 0 else 0.0
        share = n / stats2["total"] if stats2["total"] else 0.0
        tribe_rows.append([f"T{t}", f"temp={TRIBE_TEMPS[t]}", str(n), f"{share:.0%}", f"{rate:.1%}"])
    table(
        "Per-Tribe Routing Share and Pass Rate",
        ["Tribe", "Temperature", "Tasks", "Share", "Pass Rate"],
        tribe_rows,
        col_widths=[0.8 * inch, 1.2 * inch, 0.8 * inch, 0.8 * inch, 1.0 * inch],
    )

    # ---- Conclusion ----
    hr()
    h2("7. Interpretation")
    p("<b>Hypothesis:</b> Heterogeneous temperatures create a non-flat reward surface, "
      "allowing CTS to learn meaningful routing preferences per category.")
    p("<b>Key question:</b> Is the oracle gap larger in Phase 2 (more room to route) "
      "AND is the actual CTS pass rate higher (gap is being exploited)?")
    p("If both are true: tribe diversity is a prerequisite for effective routing. "
      "If oracle gap is larger but CTS rate unchanged: 2000 tasks is insufficient for CTS to learn. "
      "If neither: even temperature diversity does not create exploitable differences.")

    doc.build(body)
    print(f"  Report saved -> {OUTPUT_PDF}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    runs1c = load_runs(PHASE1C_FILE)
    runs2  = load_runs(PHASE2_FILE)

    if not runs2:
        print(f"[error] {PHASE2_FILE} is empty or missing. Run parse_phase2_dump.py first.")
        raise SystemExit(1)

    print(f"  Phase 1C: {len(runs1c)} runs")
    print(f"  Phase 2:  {len(runs2)} runs")

    stats1c = compute_stats(runs1c)
    stats2  = compute_stats(runs2)

    print(f"\nPhase 1C: {stats1c['pass_rate']:.1%} actual, {stats1c['oracle_rate']:.1%} oracle "
          f"(gap={stats1c['oracle_gap']:.1%})")
    print(f"Phase 2:  {stats2['pass_rate']:.1%} actual, {stats2['oracle_rate']:.1%} oracle "
          f"(gap={stats2['oracle_gap']:.1%})")

    print("\nGenerating figures and report...")
    build_report(stats1c, stats2, runs1c, runs2)
    print("Done.")
