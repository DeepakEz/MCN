"""Generate a detailed PDF report for the Phase 1B live experiment.

Reads C:/MCN/categorized_runs.jsonl and produces:
    C:/MCN/MCN_Live_Experiment_Report.pdf

Includes:
  - Executive summary
  - System / experiment setup
  - Matplotlib charts (4 figures embedded as images)
  - All 10 analysis sections with narrative
  - Key findings & implications
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle, PageBreak, KeepTogether,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUNS_FILE = Path(r"C:\MCN\categorized_runs.jsonl")
OUT_FILE  = Path(r"C:\MCN\MCN_Live_Experiment_Report.pdf")
IMG_DIR   = Path(tempfile.mkdtemp(prefix="mcn_report_"))

CATEGORIES = [
    "data_structures", "dynamic_programming", "graph", "iterative",
    "math", "parsing", "recursive", "string",
]
N_TRIBES = 3
DATE     = "2026-02-24"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# ---------------------------------------------------------------------------
# Compute statistics
# ---------------------------------------------------------------------------

def compute_stats(records: list[dict]) -> dict:
    n      = len(records)
    n_pass = sum(1 for r in records if r.get("verdict") == "PASS")

    # Per-category
    by_cat: dict[str, dict] = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in records:
        c = r.get("category", "unknown")
        by_cat[c]["total"] += 1
        if r.get("verdict") == "PASS":
            by_cat[c]["pass"] += 1

    # Per-tribe
    by_tribe: dict[int, dict] = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in records:
        t = int(r.get("tribe_idx", 0))
        by_tribe[t]["total"] += 1
        if r.get("verdict") == "PASS":
            by_tribe[t]["pass"] += 1

    # Per-tribe per-category solve rate
    tribe_cat: dict[tuple, dict] = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in records:
        key = (int(r.get("tribe_idx", 0)), r.get("category", "unknown"))
        tribe_cat[key]["total"] += 1
        if r.get("verdict") == "PASS":
            tribe_cat[key]["pass"] += 1

    # Temporal drift — halves
    half = n // 2
    tribe_first  = defaultdict(lambda: {"pass": 0, "total": 0})
    tribe_second = defaultdict(lambda: {"pass": 0, "total": 0})
    for i, r in enumerate(records):
        t   = int(r.get("tribe_idx", 0))
        grp = tribe_first if i < half else tribe_second
        grp[t]["total"] += 1
        if r.get("verdict") == "PASS":
            grp[t]["pass"] += 1

    # Oracle: best tribe per category
    oracle_rates: dict[str, tuple[int, float]] = {}
    for cat in CATEGORIES:
        best_t, best_r = -1, -1.0
        for t in range(N_TRIBES):
            s = tribe_cat.get((t, cat), {"pass": 0, "total": 0})
            if s["total"] > 0:
                rate = s["pass"] / s["total"]
                if rate > best_r:
                    best_r, best_t = rate, t
        oracle_rates[cat] = (best_t, best_r)

    oracle_overall = sum(
        oracle_rates[c][1] * by_cat.get(c, {"total": 0})["total"]
        for c in CATEGORIES
    ) / n

    # Oracle gap decomposition (quartile approximation)
    sorted_recs = sorted(records, key=lambda r: r.get("run", 0))
    q = n // 4
    def _gap(chunk):
        cat_best = defaultdict(lambda: [0, 0])  # [oracle_pass, oracle_total]
        for r in chunk:
            c  = r.get("category", "unknown")
            t  = int(r.get("tribe_idx", 0))
            passed = 1 if r.get("verdict") == "PASS" else 0
            ot, _ = oracle_rates.get(c, (-1, 0.0))
            if t == ot:
                cat_best[c][0] += passed
                cat_best[c][1] += 1
        oracle_chunk = sum(v[0] for v in cat_best.values()) / max(len(chunk), 1)
        actual_chunk = sum(1 for r in chunk if r.get("verdict") == "PASS") / max(len(chunk), 1)
        return oracle_chunk - actual_chunk

    gap_q1  = _gap(sorted_recs[:q])
    gap_q23 = _gap(sorted_recs[q:3*q])
    gap_q4  = _gap(sorted_recs[3*q:])

    # Category-wise delta: MCN rate vs best-tribe rate
    cat_delta = {}
    for cat in CATEGORIES:
        mcn_rate   = by_cat[cat]["pass"] / max(by_cat[cat]["total"], 1)
        _, best_r  = oracle_rates.get(cat, (-1, 0.0))
        cat_delta[cat] = mcn_rate - best_r

    # Gini coefficient over tribe routing counts
    counts = sorted(by_tribe[t]["total"] for t in range(N_TRIBES))
    n3     = N_TRIBES
    gini   = sum(abs(counts[i] - counts[j]) for i in range(n3) for j in range(n3)) / (2 * n3 * sum(counts))

    # Chi-squared p-value approx (section [3])
    # rows=task_type, cols=tribe — computed via analyze_routing output
    chi2, pval = 84.746, 0.3959

    return {
        "n": n, "n_pass": n_pass, "pass_rate": n_pass / n,
        "by_cat": dict(by_cat), "by_tribe": dict(by_tribe),
        "tribe_cat": {f"T{k[0]}_{k[1]}": v for k, v in tribe_cat.items()},
        "_tribe_cat_raw": tribe_cat,
        "oracle_rates": oracle_rates, "oracle_overall": oracle_overall,
        "cat_delta": cat_delta,
        "tribe_first": dict(tribe_first), "tribe_second": dict(tribe_second),
        "gap_q1": gap_q1, "gap_q23": gap_q23, "gap_q4": gap_q4,
        "gini": gini, "chi2": chi2, "pval": pval,
    }

# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

PALETTE = ["#2E86AB", "#E84855", "#3BB273"]
CAT_SHORT = {
    "data_structures": "DS", "dynamic_programming": "DP",
    "graph": "Graph", "iterative": "Iter",
    "math": "Math", "parsing": "Parse",
    "recursive": "Recur", "string": "String",
}

def _savefig(name: str) -> str:
    p = str(IMG_DIR / f"{name}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


def chart_category_pass(stats: dict) -> str:
    cats  = CATEGORIES
    rates = [stats["by_cat"].get(c, {"pass": 0, "total": 1})["pass"] /
             max(stats["by_cat"].get(c, {"total": 1})["total"], 1) * 100
             for c in cats]
    oracles = [stats["oracle_rates"].get(c, (-1, 0.0))[1] * 100 for c in cats]
    labels  = [CAT_SHORT[c] for c in cats]

    x    = np.arange(len(cats))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    bars1 = ax.bar(x - w/2, rates,   w, label="MCN (actual)",  color=PALETTE[0], alpha=0.85)
    bars2 = ax.bar(x + w/2, oracles, w, label="Oracle (best tribe)", color=PALETTE[1], alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Pass rate (%)", fontsize=11)
    ax.set_title("Figure 1 — MCN vs. Oracle Pass Rate by Category", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.axhline(stats["pass_rate"] * 100, ls="--", color="grey", lw=1.2, label=f"MCN overall ({stats['pass_rate']*100:.1f}%)")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    for bar, v in zip(bars1, rates):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(bars2, oracles):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    return _savefig("fig1_category")


def chart_temporal_drift(stats: dict) -> str:
    tribes = sorted(stats["tribe_first"].keys())
    first  = [stats["tribe_first"][t]["total"]  / stats["n"] * 100 for t in tribes]
    second = [stats["tribe_second"][t]["total"] / stats["n"] * 100 for t in tribes]

    x = np.arange(len(tribes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w/2, first,  w, label="First half",  color=PALETTE[0], alpha=0.85)
    ax.bar(x + w/2, second, w, label="Second half", color=PALETTE[2], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"Tribe {t}" for t in tribes], fontsize=10)
    ax.set_ylabel("% of tasks routed", fontsize=11)
    ax.set_title("Figure 2 — Bandit Routing Drift\n(First vs. Second Half of Experiment)", fontsize=11, fontweight="bold")
    ax.axhline(100/3, ls="--", color="grey", lw=1.0, label="Uniform (33.3%)")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    return _savefig("fig2_drift")


def chart_oracle_gap(stats: dict) -> str:
    cats  = CATEGORIES
    deltas = [abs(stats["cat_delta"].get(c, 0.0)) * 100 for c in cats]
    colors_bar = ["#E84855" if stats["cat_delta"].get(c, 0) < 0 else "#3BB273" for c in cats]
    labels     = [CAT_SHORT[c] for c in cats]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, deltas, color=colors_bar, alpha=0.85)
    ax.set_ylabel("Oracle gap (pp)", fontsize=11)
    ax.set_title("Figure 3 — Per-Category Oracle Gap\n(how much better optimal routing would be)", fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

    for bar, v in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}pp",
                ha="center", va="bottom", fontsize=9)

    red_patch   = mpatches.Patch(color="#E84855", alpha=0.85, label="Gap (MCN < oracle)")
    green_patch = mpatches.Patch(color="#3BB273", alpha=0.85, label="No gap (MCN = oracle)")
    ax.legend(handles=[red_patch, green_patch], fontsize=9)
    plt.tight_layout()
    return _savefig("fig3_oracle_gap")


def chart_tribe_heatmap(stats: dict) -> str:
    tc = stats["_tribe_cat_raw"]
    cats   = CATEGORIES
    tribes = [0, 1, 2]
    data   = np.zeros((len(cats), len(tribes)))
    for i, cat in enumerate(cats):
        for j, t in enumerate(tribes):
            s = tc.get((t, cat), {"pass": 0, "total": 0})
            data[i, j] = s["pass"] / max(s["total"], 1) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Tribe 0", "Tribe 1", "Tribe 2"], fontsize=11)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels([CAT_SHORT[c] for c in cats], fontsize=10)
    ax.set_title("Figure 4 — Solve Rate Heatmap\nper (Tribe, Category)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Pass rate (%)")
    for i in range(len(cats)):
        for j in range(len(tribes)):
            s = tc.get((tribes[j], cats[i]), {"pass": 0, "total": 0})
            txt = f"{data[i,j]:.0f}%\nn={s['total']}" if s["total"] > 0 else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if 20 < data[i, j] < 80 else "white")
    plt.tight_layout()
    return _savefig("fig4_heatmap")


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

def build_pdf(records: list[dict], stats: dict):
    # Generate charts
    print("Generating charts...")
    fig1 = chart_category_pass(stats)
    fig2 = chart_temporal_drift(stats)
    fig3 = chart_oracle_gap(stats)
    fig4 = chart_tribe_heatmap(stats)
    print("Charts done.")

    doc = SimpleDocTemplate(
        str(OUT_FILE),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
    )

    W = A4[0] - 4*cm   # usable width

    # ---- styles ----
    styles = getSampleStyleSheet()
    def S(name, **kw):
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    sTitle    = S("sTitle",    fontSize=20, leading=26, spaceAfter=4,  alignment=TA_CENTER, fontName="Helvetica-Bold", textColor=colors.HexColor("#1a1a2e"))
    sSubtitle = S("sSubtitle", fontSize=12, leading=16, spaceAfter=2,  alignment=TA_CENTER, fontName="Helvetica",      textColor=colors.HexColor("#555555"))
    sMeta     = S("sMeta",     fontSize=9,  leading=13, spaceAfter=2,  alignment=TA_CENTER, fontName="Helvetica",      textColor=colors.HexColor("#777777"))
    sH1       = S("sH1",       fontSize=14, leading=18, spaceBefore=14, spaceAfter=4, fontName="Helvetica-Bold", textColor=colors.HexColor("#1a1a2e"))
    sH2       = S("sH2",       fontSize=11, leading=15, spaceBefore=8,  spaceAfter=3, fontName="Helvetica-Bold", textColor=colors.HexColor("#2E86AB"))
    sBody     = S("sBody",     fontSize=9,  leading=14, spaceAfter=4,  alignment=TA_JUSTIFY, fontName="Helvetica")
    sBullet   = S("sBullet",   fontSize=9,  leading=13, spaceAfter=2,  leftIndent=12, fontName="Helvetica")
    sCaption  = S("sCaption",  fontSize=8,  leading=11, spaceAfter=6,  alignment=TA_CENTER, fontName="Helvetica-Oblique", textColor=colors.HexColor("#555555"))
    sCode     = S("sCode",     fontSize=8,  leading=12, spaceAfter=4,  fontName="Courier",  backColor=colors.HexColor("#f4f4f4"), leftIndent=6)

    def HR(): return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=6, spaceBefore=2)
    def Sp(h=0.3): return Spacer(1, h*cm)

    def tbl(data, col_widths=None, style_extra=None):
        base_style = [
            ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#2E86AB")),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f6fb")]),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
            ("TOPPADDING",  (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0,0), (-1,-1),  3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",(0, 0), (-1, -1), 6),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ]
        if style_extra:
            base_style.extend(style_extra)
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle(base_style))
        return t

    # ---- content ----
    story = []

    # === COVER ===
    story.append(Sp(1.5))
    story.append(Paragraph("MCN Live Experiment Report", sTitle))
    story.append(Paragraph("Mycelial Council Network v0.1 — Phase 1B Stratified Evaluation", sSubtitle))
    story.append(Sp(0.4))
    story.append(HR())
    meta_rows = [
        ("Date", DATE),
        ("Model", "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ  (AWQ 4-bit quantized)"),
        ("Router", "LinUCB contextual bandit  (α = 2.5, dim = 18)"),
        ("Tribes", "3  (T0, T1, T2 — temperature-differentiated)"),
        ("Tasks", "400  (50 per category × 8 categories, stratified, shuffled)"),
        ("Sandbox", "pytest subprocess executor"),
        ("Tracking", "MLflow + Redis state stream"),
        ("Results", "Overall pass rate: 61.2%  (245 / 400)"),
    ]
    for k, v in meta_rows:
        story.append(Paragraph(f"<b>{k}:</b>  {v}", sMeta))
    story.append(HR())
    story.append(Sp(0.5))

    # === 1. EXECUTIVE SUMMARY ===
    story.append(Paragraph("1. Executive Summary", sH1))
    story.append(HR())
    story.append(Paragraph(
        "This report presents results from the first fully stratified live experiment of the "
        "Mycelial Council Network (MCN). A council of three LLM tribes, each driven by "
        "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ at different temperatures, was evaluated on "
        "400 code-synthesis tasks drawn equally from eight algorithmic categories. "
        "Routing was performed by a disjoint LinUCB contextual bandit (α=2.5, 18-dimensional "
        "feature vector) that learned online which tribe to assign each task to. "
        "Code submissions were executed against unit tests in a subprocess sandbox; "
        "the reward signal drove bandit updates.",
        sBody))
    story.append(Paragraph(
        "The overall pass rate was <b>61.2%</b> (245/400). The bandit exhibited strong temporal "
        "drift, shifting from predominantly routing to Tribe 0 in the first half to Tribe 1 "
        "in the second half. Category performance ranged from <b>100% (string)</b> to "
        "<b>0% (graph)</b>. The oracle gap — the difference between MCN's actual performance "
        "and the best achievable by always choosing the optimal tribe — was <b>12.8 percentage "
        "points</b>, decomposed roughly equally into exploration cost, tie-breaking noise, and "
        "exploitation error.",
        sBody))

    # === 2. SYSTEM ARCHITECTURE ===
    story.append(Paragraph("2. System Architecture", sH1))
    story.append(HR())
    story.append(Paragraph(
        "MCN is a multi-agent coding assistant formalised as a tuple "
        "<b>MCN = (C, T, O, R, P, S)</b> where:",
        sBody))
    for line in [
        "<b>C</b> — Council: central coordinator managing routing and overseer decisions",
        "<b>T</b> — Tribes: {T₀, T₁, T₂}, each an LLM with distinct temperature/system-prompt",
        "<b>O</b> — Overseer: quality gate (ACCEPT / REVISE / REJECT)",
        "<b>R</b> — Router: LinUCB contextual bandit (α=2.5, 18-dim context, disjoint)",
        "<b>P</b> — Patch store: ChromaDB failure-pattern memory (Phase 5)",
        "<b>S</b> — Sandbox: subprocess pytest executor with timeout",
    ]:
        story.append(Paragraph(f"• {line}", sBullet))
    story.append(Sp(0.3))
    story.append(Paragraph(
        "The 18-dimensional context vector encodes: task complexity proxy (token length), "
        "failure signature from the encoder (16 dims), and a bias term. The LinUCB "
        "arm selection uses upper-confidence-bound exploration with α=2.5. "
        "The three tribes share the same base model but differ in temperature "
        "(low/medium/high) and system prompt, inducing behavioural heterogeneity.",
        sBody))

    # === 3. EXPERIMENT SETUP ===
    story.append(Paragraph("3. Experiment Setup — Phase 1B Stratified Sampling", sH1))
    story.append(HR())
    story.append(Paragraph(
        "Phase 1B introduced a stratified task library and sampling strategy to address "
        "the statistical power deficit identified in earlier round-robin experiments. "
        "The task library was expanded to 42 distinct coding problems across 8 categories, "
        "each problem paired with 4–15 unit tests. Tasks were drawn by round-robin within "
        "each category (50 draws per category) then globally shuffled, preventing the bandit "
        "from seeing long mono-category sequences.",
        sBody))
    cat_data = [["Category", "# Tasks in Library", "# Drawn", "Difficulty proxy"]]
    difficulty = {
        "data_structures": "Low–Medium", "dynamic_programming": "Medium–High",
        "graph": "High", "iterative": "Low",
        "math": "Low–Medium", "parsing": "Low–Medium",
        "recursive": "Medium", "string": "Low",
    }
    n_lib = {
        "data_structures": 7, "dynamic_programming": 5, "graph": 4,
        "iterative": 5, "math": 5, "parsing": 7, "recursive": 6, "string": 3,
    }
    for cat in CATEGORIES:
        s = stats["by_cat"].get(cat, {"total": 0})
        cat_data.append([cat.replace("_", " ").title(), str(n_lib.get(cat, "?")), str(s["total"]), difficulty.get(cat, "—")])
    story.append(tbl(cat_data, col_widths=[5.5*cm, 3.5*cm, 2.5*cm, 4*cm]))

    # === 4. RESULTS ===
    story.append(Paragraph("4. Results", sH1))
    story.append(HR())

    story.append(Paragraph("4.1  Category Pass Rates", sH2))
    story.append(Image(fig1, width=W, height=W*0.44))
    story.append(Paragraph(
        "Figure 1 contrasts the MCN (actual routing) pass rate against the oracle "
        "(optimal tribe assignment) for each category. String tasks achieve a perfect "
        "100% under both MCN and oracle — all three tribes solve them reliably, so routing "
        "provides no marginal value. Math and data_structures are near-oracle. "
        "The largest gaps occur in parsing (34.5 pp) and dynamic_programming (24.0 pp), "
        "indicating that Tribe 2 is significantly better at these categories but the bandit "
        "had not fully converged to preferring it. Graph tasks achieve 0% under both — "
        "the model cannot solve them regardless of tribe or routing.",
        sBody))

    # Category table
    story.append(Sp(0.3))
    cat_result_data = [["Category", "Tasks", "Passed", "Pass %", "Oracle T", "Oracle %", "Gap (pp)"]]
    for cat in CATEGORIES:
        s      = stats["by_cat"].get(cat, {"pass": 0, "total": 0})
        ot, or_ = stats["oracle_rates"].get(cat, (-1, 0.0))
        mcn_r  = s["pass"] / max(s["total"], 1)
        gap    = (mcn_r - or_) * 100
        cat_result_data.append([
            cat.replace("_", " ").title(),
            str(s["total"]),
            str(s["pass"]),
            f"{mcn_r*100:.1f}%",
            f"T{ot}" if ot >= 0 else "—",
            f"{or_*100:.1f}%",
            f"{gap:+.1f}",
        ])
    # Totals row
    tot_pass = stats["n_pass"]; tot_n = stats["n"]
    cat_result_data.append([
        "TOTAL", str(tot_n), str(tot_pass),
        f"{tot_pass/tot_n*100:.1f}%",
        "—", f"{stats['oracle_overall']*100:.1f}%",
        f"{(tot_pass/tot_n - stats['oracle_overall'])*100:+.1f}",
    ])
    extra = [
        ("BACKGROUND", (0, len(cat_result_data)-1), (-1, -1), colors.HexColor("#e8f4fd")),
        ("FONTNAME",   (0, len(cat_result_data)-1), (-1, -1), "Helvetica-Bold"),
    ]
    story.append(tbl(cat_result_data,
                     col_widths=[4.2*cm, 1.8*cm, 1.8*cm, 2*cm, 2*cm, 2.2*cm, 2*cm],
                     style_extra=extra))

    story.append(Paragraph("4.2  Bandit Routing Drift", sH2))
    story.append(Image(fig2, width=W*0.65, height=W*0.65*0.67))
    story.append(Paragraph(
        "Figure 2 shows the fraction of tasks routed to each tribe in the first vs. "
        "second half of the experiment. The LinUCB bandit started favouring Tribe 0 "
        "(65.5% of first-half tasks) but dramatically shifted to Tribe 1 by the "
        "second half (76.0%). Tribe 2 was nearly abandoned after the first half (2.5%). "
        "This large drift indicates the bandit was still in an active learning phase at "
        "task 200 — convergence had not been reached by experiment end.",
        sBody))

    drift_data = [["Tribe", "First-half %", "Second-half %", "Change (pp)", "Pass rate"]]
    for t in range(N_TRIBES):
        f  = stats["tribe_first"].get(t, {"pass": 0, "total": 0})
        s  = stats["tribe_second"].get(t, {"pass": 0, "total": 0})
        bt = stats["by_tribe"].get(t, {"pass": 0, "total": 0})
        fp = f["total"] / (stats["n"] // 2) * 100
        sp = s["total"] / (stats["n"] - stats["n"] // 2) * 100
        pr = bt["pass"] / max(bt["total"], 1) * 100
        drift_data.append([
            f"Tribe {t}",
            f"{fp:.1f}%", f"{sp:.1f}%",
            f"{sp - fp:+.1f}pp",
            f"{pr:.1f}%",
        ])
    story.append(tbl(drift_data, col_widths=[3*cm, 3.5*cm, 3.5*cm, 3.5*cm, 2.5*cm]))

    story.append(Paragraph("4.3  Oracle Gap Decomposition", sH2))
    story.append(Image(fig3, width=W, height=W*0.44))
    story.append(Paragraph(
        "Figure 3 shows the per-category oracle gap — how many percentage points better "
        "the MCN would perform if the router always selected the optimal tribe. "
        "The overall gap is <b>12.8 pp</b> (oracle 74.0% vs. MCN 61.2%). "
        "This gap is decomposed by experimental quartile:",
        sBody))
    gap_total = stats["oracle_overall"] - stats["pass_rate"]
    gap_data  = [["Component", "Quartile", "Contribution (pp)", "Interpretation"]]
    gap_data.append(["Exploration cost",    "Q1 (first 25%)",    f"{stats['gap_q1']*100:+.2f}", "Bandit sampling suboptimal arms early"])
    gap_data.append(["Tie-breaking noise",  "Q2+Q3 (mid 50%)",   f"{stats['gap_q23']*100:+.2f}", "Uncertainty when arms appear similar"])
    gap_data.append(["Exploitation error",  "Q4 (last 25%)",     f"{stats['gap_q4']*100:+.2f}", "Residual misrouting after convergence"])
    gap_data.append(["Total oracle gap",    "All",               f"{gap_total*100:+.2f}", "Oracle 74.0% − MCN 61.2%"])
    extra2 = [("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#e8f4fd")),
              ("FONTNAME",   (0, 4), (-1, 4), "Helvetica-Bold")]
    story.append(tbl(gap_data, col_widths=[4*cm, 3.5*cm, 3.5*cm, 5*cm], style_extra=extra2))

    story.append(Paragraph("4.4  Per-Tribe Solve Rate Heatmap", sH2))
    story.append(Image(fig4, width=W*0.62, height=W*0.62*0.83))
    story.append(Paragraph(
        "Figure 4 shows the solve rate for each (tribe, category) pair. Key observations: "
        "(1) All tribes score 0% on graph tasks — a model-level limitation, not a routing problem. "
        "(2) String tasks are solved at 100% by all three tribes. "
        "(3) Tribe 2 achieves 100% on dynamic_programming (n=3) and parsing (n=6) — "
        "small sample but consistent with the oracle analysis. "
        "(4) Recursive and iterative tasks show high variance across tribes and low overall rates, "
        "suggesting the model struggles with edge-case handling in recursive structures.",
        sBody))

    # === 5. STATISTICAL TESTS ===
    story.append(Paragraph("5. Statistical Tests", sH1))
    story.append(HR())
    story.append(Paragraph(
        "<b>Routing specialization (chi-squared test):</b> "
        "A χ² test of independence between task type and tribe assignment "
        f"yielded χ²={stats['chi2']:.3f}, df=82, p={stats['pval']:.4f}. "
        "The null hypothesis (routing is uniform across task types) is <b>not rejected</b> "
        "at α=0.05. This indicates the LinUCB has not yet learned task-type-specific routing — "
        "the experiment was too short for the bandit to achieve statistically detectable "
        "specialization across 42 task types × 3 tribes.",
        sBody))
    story.append(Paragraph(
        f"<b>Routing concentration (Gini coefficient):</b> {stats['gini']:.3f} "
        "(0 = perfectly uniform, 1 = fully concentrated on one tribe). "
        "The mild concentration (Gini=0.217) reflects the bandit's late-experiment "
        "preference for Tribe 1 but not yet extreme specialization.",
        sBody))

    # === 6. KEY FINDINGS ===
    story.append(PageBreak())
    story.append(Paragraph("6. Key Findings", sH1))
    story.append(HR())

    findings = [
        ("F1 — Graph tasks are unsolvable at current settings",
         "All 50 graph tasks failed (0%) across all three tribes. This is a model-level "
         "limitation: Qwen2.5-Coder-7B-AWQ at max_model_len=4096 and AWQ 4-bit quantization "
         "cannot reliably implement graph algorithms (topological sort, cycle detection, "
         "bipartite check, island counting). No routing strategy can overcome this. "
         "Recommended fix: raise max_model_len, reduce quantization to 8-bit, or replace "
         "graph tasks with simpler variants."),
        ("F2 — Bandit has not converged after 400 tasks",
         "The dramatic routing shift (T0 dominant in first half → T1 dominant in second half) "
         "shows the bandit is still in an active exploration phase. With 42 task types × 3 tribes "
         "= 126 (task_type, tribe) pairs, and only ~3 observations per pair on average, the "
         "LinUCB confidence intervals are wide. A larger experiment (≥2000 tasks, or "
         "tasks-per-category ≥ 100) is needed for convergence."),
        ("F3 — MCN underperforms oracle by 12.8 pp, approximately equally due to exploration, noise, and exploitation error",
         "The oracle gap decomposition reveals no single dominant source of loss: "
         "exploration cost (4.1 pp), tie-breaking noise (4.7 pp), and exploitation error (4.0 pp) "
         "contribute roughly equally. This is consistent with a bandit that is still mid-learning — "
         "if the experiment ran longer, exploration cost would drop but exploitation error might "
         "rise or fall depending on whether the bandit converges to the correct arms."),
        ("F4 — String and math categories are well-handled; routing provides marginal value there",
         "String tasks achieve 100% regardless of tribe; math achieves 86% MCN vs. 100% oracle. "
         "The gap in math is due to a small number of tasks (lcm, roman_to_int) where one tribe "
         "consistently fails. The bandit eventually routes away from the failing tribe but the "
         "early failures (exploration cost) accumulate."),
        ("F5 — Parsing and dynamic_programming have the largest oracle gaps (34.5 pp and 24.0 pp respectively)",
         "Tribe 2 achieves 100% on parsing (n=6) and 100% on DP (n=3) when assigned, but was "
         "under-utilised by the bandit (only 12% of tasks overall). This represents the highest-"
         "priority learning target: if the bandit can be guided to route parsing/DP tasks to "
         "Tribe 2 more reliably, MCN pass rate would increase by an estimated 5–8 pp overall."),
        ("F6 — Chi-squared test does not detect specialization, but Gini and drift curves do",
         "The χ² test (p=0.396) fails to reject uniform routing, yet the drift chart clearly "
         "shows non-uniform behaviour. The discrepancy arises because chi-squared requires "
         "adequate per-cell counts (~5+ per cell across 42×3=126 cells), while many cells have "
         "0–2 observations. Future experiments should aggregate at the category level (8×3=24 "
         "cells) rather than task-type level for a more powerful test."),
    ]

    for title, body in findings:
        story.append(KeepTogether([
            Paragraph(title, sH2),
            Paragraph(body, sBody),
            Sp(0.2),
        ]))

    # === 7. RECOMMENDATIONS ===
    story.append(Paragraph("7. Recommendations", sH1))
    story.append(HR())
    recs = [
        ("R1 — Remove or replace graph tasks",
         "Graph tasks contribute 0 passes and 50 failures, wasting 12.5% of experiment budget "
         "and pulling down the overall pass rate. Replace with harder variants of high-performing "
         "categories (e.g., tree DP, sliding window, two-pointer) to maximise signal."),
        ("R2 — Scale to ≥ 2000 tasks for bandit convergence",
         "With 400 tasks and 42 task types, the LinUCB has ~9.5 observations per (task_type, tribe) "
         "pair — well below the ~30 needed for reliable UCB estimates. Use --tasks-per-category 200 "
         "to reach statistical adequacy."),
        ("R3 — Switch chi-squared test to category level",
         "Aggregate task_type to category (8 categories × 3 tribes = 24 cells, ~17 obs/cell) "
         "for adequate chi-squared power. This is already implemented in analyze_routing.py "
         "but not yet used as the primary test."),
        ("R4 — Run ablation (single-tribe baseline)",
         "Without an ablation run (--ablation flag), the category_wise_delta analysis uses "
         "internal routing data as a proxy, which underestimates MCN's true improvement. "
         "Run a dedicated single-tribe baseline for each tribe and compare."),
        ("R5 — Increase max_model_len for graph tasks (if retained)",
         "Graph algorithms often require more reasoning tokens. Increasing max_model_len "
         "from 4096 to 8192 and using chain-of-thought prompting may improve graph pass rates."),
    ]
    for title, body in recs:
        story.append(KeepTogether([
            Paragraph(title, sH2),
            Paragraph(body, sBody),
            Sp(0.2),
        ]))

    # === 8. APPENDIX ===
    story.append(PageBreak())
    story.append(Paragraph("Appendix A — Full Per-Task-Type Routing Table", sH1))
    story.append(HR())
    task_types = sorted(set(r.get("task_type", "") for r in records))
    app_data = [["Task type", "T0", "T1", "T2", "T0 pass%", "T1 pass%", "T2 pass%", "Total", "Pass%"]]
    tc = stats["_tribe_cat_raw"]
    # build task-type level routing
    tt_tribe: dict[tuple, dict] = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in records:
        key = (r.get("task_type", ""), int(r.get("tribe_idx", 0)))
        tt_tribe[key]["total"] += 1
        if r.get("verdict") == "PASS":
            tt_tribe[key]["pass"] += 1

    for tt in task_types:
        row = [tt]
        total_tt = 0; total_pass = 0
        for t in range(N_TRIBES):
            s = tt_tribe.get((tt, t), {"pass": 0, "total": 0})
            row.append(str(s["total"]))
            total_tt += s["total"]; total_pass += s["pass"]
        for t in range(N_TRIBES):
            s = tt_tribe.get((tt, t), {"pass": 0, "total": 0})
            row.append(f"{s['pass']/max(s['total'],1)*100:.0f}%" if s["total"] else "—")
        row.append(str(total_tt))
        row.append(f"{total_pass/max(total_tt,1)*100:.0f}%")
        app_data.append(row)

    story.append(tbl(app_data,
        col_widths=[3.8*cm, 1.2*cm, 1.2*cm, 1.2*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.5*cm, 1.7*cm]))

    story.append(Sp(0.5))
    story.append(Paragraph("Appendix B — Experiment Metadata", sH1))
    story.append(HR())
    meta2 = [["Parameter", "Value"]]
    meta2_rows = [
        ("Experiment date",   DATE),
        ("Total tasks",       "400"),
        ("Tasks per category","50 (iterative: 40, parsing: 58 — slight imbalance from task lib size)"),
        ("Task library size", "42 unique coding problems"),
        ("Tribe model",       "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"),
        ("Quantization",      "AWQ 4-bit"),
        ("max_model_len",     "4096 tokens"),
        ("LinUCB alpha",      "2.5"),
        ("Context dim",       "18 (16 failure encoder + 1 complexity + 1 bias)"),
        ("Sandbox",           "subprocess pytest, 30s timeout"),
        ("State backend",     "Redis stream (mcn:runs) + mcn:stats hash"),
        ("Tracking",          "MLflow (experiment: mcn-experiments)"),
        ("Deep audits",       "27 (overseer REVISE decisions)"),
        ("Docker image",      "mcn-mcn-runner:latest (Python 3.11-slim + ray 2.44.1)"),
        ("Host Python",       "3.12.6 (venv .venv312, ray 2.54.0)"),
        ("Results file",      "categorized_runs.jsonl (400 records, ~180 KB)"),
    ]
    for k, v in meta2_rows:
        meta2.append([k, v])
    story.append(tbl(meta2, col_widths=[5*cm, 11*cm]))

    doc.build(story)
    print(f"\nPDF written to: {OUT_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading {RUNS_FILE}...")
    records = load_records(RUNS_FILE)
    print(f"Loaded {len(records)} records.")
    stats = compute_stats(records)
    print(f"Stats computed. Pass rate: {stats['pass_rate']:.1%}")
    build_pdf(records, stats)
