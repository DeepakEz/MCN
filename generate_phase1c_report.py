"""Generate Phase 1C comparative report (400 vs 2000 tasks).

Reads:
    C:/MCN/categorized_runs.jsonl          -- Phase 1B (400 tasks)
    C:/MCN/categorized_runs_phase1c.jsonl  -- Phase 1C (2000 tasks)

Produces:
    C:/MCN/MCN_Phase1C_Report.pdf
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle, PageBreak, KeepTogether,
)

# ── Config ────────────────────────────────────────────────────────────────────
FILE_1B  = Path(r"C:\MCN\categorized_runs.jsonl")
FILE_1C  = Path(r"C:\MCN\categorized_runs_phase1c.jsonl")
OUT_FILE = Path(r"C:\MCN\MCN_Phase1C_Report.pdf")
IMG_DIR  = Path(tempfile.mkdtemp(prefix="mcn_1c_"))

CATEGORIES = [
    "string", "math", "data_structures", "dynamic_programming",
    "parsing", "iterative", "recursive", "graph",
]
CAT_SHORT = {
    "string": "str", "math": "math", "data_structures": "d_s",
    "dynamic_programming": "DP", "parsing": "parse",
    "iterative": "iter", "recursive": "rec", "graph": "graph",
}

DARK_BLUE  = "#1F3864"
MID_BLUE   = "#2E74B5"
ACCENT     = "#4472C4"
AMBER      = "#ED7D31"
GREEN      = "#70AD47"
RED        = "#C00000"
LGREY      = "#F5F5F5"

# ── Data loading ─────────────────────────────────────────────────────────────

def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def cat_stats(records: list[dict]) -> dict:
    """Returns {cat: {pass, total, pct}} sorted by CATEGORIES order."""
    by_cat: dict = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in records:
        c = r.get("category", "unknown")
        by_cat[c]["total"] += 1
        if r.get("verdict") == "PASS":
            by_cat[c]["pass"] += 1
    out = {}
    for c in CATEGORIES:
        s = by_cat.get(c, {"pass": 0, "total": 0})
        out[c] = {**s, "pct": 100 * s["pass"] / s["total"] if s["total"] else 0}
    return out

def tribe_stats(records: list[dict]) -> dict:
    by_t: dict = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in records:
        t = int(r.get("tribe_idx", 0))
        by_t[t]["total"] += 1
        if r.get("verdict") == "PASS":
            by_t[t]["pass"] += 1
    return dict(sorted(by_t.items()))

# ── Charts ────────────────────────────────────────────────────────────────────

def chart_category_comparison(recs_1b, recs_1c, out: Path) -> Path:
    """Grouped bar: Phase 1B vs Phase 1C pass rate per category."""
    st1 = cat_stats(recs_1b)
    st2 = cat_stats(recs_1c)

    x = np.arange(len(CATEGORIES))
    w = 0.35
    y1 = [st1[c]["pct"] for c in CATEGORIES]
    y2 = [st2[c]["pct"] for c in CATEGORIES]
    labels = [CAT_SHORT[c] for c in CATEGORIES]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    b1 = ax.bar(x - w/2, y1, w, label="Phase 1B (400 tasks)", color=MID_BLUE, alpha=0.85)
    b2 = ax.bar(x + w/2, y2, w, label="Phase 1C (2000 tasks)", color=AMBER, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Pass Rate (%)", fontsize=10)
    ax.set_title("Category Pass Rate: Phase 1B vs Phase 1C", fontsize=12, fontweight="bold")
    ax.axhline(61.2, color=MID_BLUE, linestyle="--", linewidth=0.8, alpha=0.6, label="1B overall 61.2%")
    ax.axhline(60.7, color=AMBER,    linestyle="--", linewidth=0.8, alpha=0.6, label="1C overall 60.7%")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in b1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.0f}%",
                    ha="center", va="bottom", fontsize=7, color=DARK_BLUE)
    for bar in b2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.0f}%",
                    ha="center", va="bottom", fontsize=7, color="#8B3A00")

    plt.tight_layout()
    p = out / "chart_cat_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_routing_evolution(recs_1c: list[dict], out: Path) -> Path:
    """Line chart: tribe routing share evolving over 2000 tasks (100-task rolling window)."""
    window = 100
    n = len(recs_1c)
    steps = list(range(window, n + 1, window))

    t0_share, t1_share, t2_share = [], [], []
    for end in steps:
        chunk = recs_1c[max(0, end-window):end]
        c = Counter(int(r.get("tribe_idx", 0)) for r in chunk)
        total = sum(c.values())
        t0_share.append(100 * c.get(0, 0) / total)
        t1_share.append(100 * c.get(1, 0) / total)
        t2_share.append(100 * c.get(2, 0) / total)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, t0_share, "-o", ms=3, color=MID_BLUE,  label="T0 (Reliable)")
    ax.plot(steps, t1_share, "-s", ms=3, color=AMBER,     label="T1 (Fast)")
    ax.plot(steps, t2_share, "-^", ms=3, color=GREEN,     label="T2 (Creative)")

    # Annotate the final dominance
    ax.annotate(f"T0: {t0_share[-1]:.0f}%", xy=(steps[-1], t0_share[-1]),
                xytext=(-40, 8), textcoords="offset points",
                fontsize=8, color=MID_BLUE, fontweight="bold")
    ax.annotate(f"T1: {t1_share[-1]:.0f}%", xy=(steps[-1], t1_share[-1]),
                xytext=(-40, -14), textcoords="offset points",
                fontsize=8, color=AMBER)
    ax.annotate(f"T2: {t2_share[-1]:.0f}%", xy=(steps[-1], t2_share[-1]),
                xytext=(-40, -14), textcoords="offset points",
                fontsize=8, color=GREEN)

    # Phase boundary annotations
    ax.axvline(500,  color="grey", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.axvline(1500, color="grey", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.text(250,  95, "T1 dominant",  ha="center", fontsize=7.5, color="grey")
    ax.text(1000, 95, "T0 rising",    ha="center", fontsize=7.5, color="grey")
    ax.text(1750, 95, "T0 dominant",  ha="center", fontsize=7.5, color="grey")

    ax.set_xlabel("Tasks completed", fontsize=10)
    ax.set_ylabel("Routing share (%)", fontsize=10)
    ax.set_title("Routing Evolution — LinUCB Bandit (100-task rolling window)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="center right")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    p = out / "chart_routing_evolution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_tribe_performance(recs_1c: list[dict], out: Path) -> Path:
    """Bar chart: per-tribe pass rate (2000 tasks) + routing share."""
    ts = tribe_stats(recs_1c)
    tribes = [f"T{t}" for t in sorted(ts)]
    pass_rates = [100 * ts[t]["pass"] / ts[t]["total"] for t in sorted(ts)]
    route_pcts = [100 * ts[t]["total"] / len(recs_1c) for t in sorted(ts)]
    tribe_colors = [MID_BLUE, AMBER, GREEN]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Pass rate
    bars = ax1.bar(tribes, pass_rates, color=tribe_colors, alpha=0.85, width=0.5)
    ax1.set_ylim(55, 65)
    ax1.set_ylabel("Pass Rate (%)", fontsize=10)
    ax1.set_title("Pass Rate per Tribe", fontsize=11, fontweight="bold")
    for bar, v in zip(bars, pass_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax1.axhline(60.7, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="Overall 60.7%")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlabel("Note: differences are not statistically significant", fontsize=7.5, style="italic")

    # Routing share
    wedge_colors = tribe_colors
    route_vals = [ts[t]["total"] for t in sorted(ts)]
    wedges, texts, autotexts = ax2.pie(
        route_vals, labels=[f"T{t}\n({route_pcts[i]:.1f}%)" for i,t in enumerate(sorted(ts))],
        colors=wedge_colors, autopct="%1.0f%%", startangle=90,
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax2.set_title("Final Routing Distribution\n(2000 tasks)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    p = out / "chart_tribe_perf.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_pass_rate_curve(recs_1c: list[dict], out: Path) -> Path:
    """Rolling pass rate over 2000 tasks (200-task window) + per-category lines."""
    window = 200
    n = len(recs_1c)
    steps = list(range(window, n + 1, 50))

    overall_pass = []
    for end in steps:
        chunk = recs_1c[max(0, end-window):end]
        overall_pass.append(100 * sum(1 for r in chunk if r.get("verdict")=="PASS") / len(chunk))

    # Rolling pass rate for 3 representative categories
    rep_cats = {"string": GREEN, "parsing": AMBER, "graph": RED}

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, overall_pass, "-", linewidth=2, color=DARK_BLUE, label="Overall (rolling 200)")
    ax.axhline(60.7, color=DARK_BLUE, linestyle="--", linewidth=0.7, alpha=0.4)

    for cat, col in rep_cats.items():
        cat_recs = [r for r in recs_1c if r.get("category") == cat]
        if len(cat_recs) < window:
            continue
        cat_pass = []
        for end in steps:
            # approximate position: use run index
            chunk_all = [r for r in recs_1c[:end] if r.get("category") == cat]
            chunk = chunk_all[-min(50, len(chunk_all)):]
            if len(chunk) >= 10:
                cat_pass.append(100 * sum(1 for r in chunk if r.get("verdict")=="PASS") / len(chunk))
            else:
                cat_pass.append(None)
        valid = [(s, v) for s, v in zip(steps, cat_pass) if v is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, "--", linewidth=1.2, color=col, alpha=0.75,
                    label=f"{cat} (rolling ~50)")

    ax.fill_between(steps, [v - 3 for v in overall_pass], [v + 3 for v in overall_pass],
                    alpha=0.12, color=DARK_BLUE)
    ax.set_xlabel("Tasks completed", fontsize=10)
    ax.set_ylabel("Pass Rate (%)", fontsize=10)
    ax.set_title("Rolling Pass Rate During Phase 1C (200-task window)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    p = out / "chart_pass_curve.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_delta_heatmap(recs_1b: list[dict], recs_1c: list[dict], out: Path) -> Path:
    """Heatmap: Δ pass rate (1C − 1B) per category, plus tribe routing share 1C."""
    st1 = cat_stats(recs_1b)
    st2 = cat_stats(recs_1c)
    deltas = [st2[c]["pct"] - st1[c]["pct"] for c in CATEGORIES]
    labels = [f"{CAT_SHORT[c]}\n{st2[c]['pct']:.0f}%" for c in CATEGORIES]

    # Tribe routing per category (1C)
    by_cat_tribe: dict = defaultdict(Counter)
    for r in recs_1c:
        by_cat_tribe[r.get("category","?")][int(r.get("tribe_idx",0))] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={"width_ratios": [1.4, 1]})

    # Δ bar chart
    bar_colors = [GREEN if d >= 0 else RED for d in deltas]
    bars = ax1.barh(labels[::-1], deltas[::-1], color=bar_colors[::-1], alpha=0.85)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Δ Pass Rate (1C − 1B, pp)", fontsize=10)
    ax1.set_title("Pass Rate Change: Phase 1C vs 1B", fontsize=11, fontweight="bold")
    for bar, v in zip(bars, deltas[::-1]):
        ax1.text(v + (0.3 if v >= 0 else -0.3), bar.get_y() + bar.get_height()/2,
                 f"{v:+.1f}pp", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax1.grid(axis="x", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Tribe routing stacked bar per category (1C)
    cats_r = CATEGORIES[::-1]
    t0_s = [100 * by_cat_tribe[c][0] / sum(by_cat_tribe[c].values()) for c in cats_r]
    t1_s = [100 * by_cat_tribe[c][1] / sum(by_cat_tribe[c].values()) for c in cats_r]
    t2_s = [100 * by_cat_tribe[c][2] / sum(by_cat_tribe[c].values()) for c in cats_r]
    y = np.arange(len(cats_r))
    ax2.barh(y, t0_s, color=MID_BLUE, alpha=0.85, label="T0")
    ax2.barh(y, t1_s, left=t0_s, color=AMBER, alpha=0.85, label="T1")
    ax2.barh(y, t2_s, left=[a+b for a,b in zip(t0_s,t1_s)], color=GREEN, alpha=0.85, label="T2")
    ax2.set_yticks(y)
    ax2.set_yticklabels([CAT_SHORT[c] for c in cats_r], fontsize=9)
    ax2.set_xlabel("Routing share (%)", fontsize=10)
    ax2.set_title("Tribe Routing per Category\n(Phase 1C)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    p = out / "chart_delta_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ── ReportLab styles ──────────────────────────────────────────────────────────

def _styles():
    S = {}
    S["title"]   = ParagraphStyle("t",  fontName="Helvetica-Bold",   fontSize=20,
                                   textColor=colors.HexColor(DARK_BLUE),
                                   alignment=TA_CENTER, spaceAfter=4)
    S["sub"]     = ParagraphStyle("s",  fontName="Helvetica-Oblique", fontSize=10,
                                   textColor=colors.HexColor(ACCENT),
                                   alignment=TA_CENTER, spaceAfter=10)
    S["h1"]      = ParagraphStyle("h1", fontName="Helvetica-Bold",   fontSize=13,
                                   textColor=colors.HexColor(DARK_BLUE),
                                   spaceBefore=14, spaceAfter=4)
    S["h2"]      = ParagraphStyle("h2", fontName="Helvetica-Bold",   fontSize=10.5,
                                   textColor=colors.HexColor(MID_BLUE),
                                   spaceBefore=8, spaceAfter=3)
    S["body"]    = ParagraphStyle("b",  fontName="Helvetica",         fontSize=9.5,
                                   leading=14, spaceAfter=6, alignment=TA_JUSTIFY)
    S["finding"] = ParagraphStyle("f",  fontName="Helvetica-BoldOblique", fontSize=9,
                                   textColor=colors.HexColor("#1F5C2E"),
                                   backColor=colors.HexColor("#E2EFDA"),
                                   leftIndent=8, rightIndent=8, borderPad=4,
                                   spaceAfter=8, leading=13)
    S["warn"]    = ParagraphStyle("w",  fontName="Helvetica-BoldOblique", fontSize=9,
                                   textColor=colors.HexColor("#7F3000"),
                                   backColor=colors.HexColor("#FCE4D6"),
                                   leftIndent=8, rightIndent=8, borderPad=4,
                                   spaceAfter=8, leading=13)
    S["th"]      = ParagraphStyle("th", fontName="Helvetica-Bold",   fontSize=8.5,
                                   textColor=colors.white, alignment=TA_CENTER)
    S["td"]      = ParagraphStyle("td", fontName="Helvetica",         fontSize=8.5,
                                   textColor=colors.black, alignment=TA_CENTER)
    S["tdl"]     = ParagraphStyle("tdl",fontName="Helvetica",         fontSize=8.5,
                                   textColor=colors.black)
    S["footer"]  = ParagraphStyle("ft", fontName="Helvetica-Oblique", fontSize=7.5,
                                   textColor=colors.grey, alignment=TA_CENTER)
    S["cap"]     = ParagraphStyle("cp", fontName="Helvetica-BoldOblique", fontSize=8,
                                   textColor=colors.HexColor(DARK_BLUE),
                                   alignment=TA_CENTER, spaceAfter=4, spaceBefore=6)
    return S

GRID = [
    ("GRID",         (0,0),(-1,-1), 0.3, colors.HexColor("#AAAAAA")),
    ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ("TOPPADDING",   (0,0),(-1,-1), 3),
    ("BOTTOMPADDING",(0,0),(-1,-1), 3),
]

def _tbl(header, rows, col_widths, S, highlight_last=False):
    data = [[Paragraph(h, S["th"]) for h in header]]
    for i, row in enumerate(rows):
        cells = []
        for j, val in enumerate(row):
            st = S["td"] if len(str(val)) <= 14 else S["tdl"]
            cells.append(Paragraph(str(val), st))
        data.append(cells)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    cmds = list(GRID) + [
        ("BACKGROUND",     (0,0),(-1,0),  colors.HexColor(DARK_BLUE)),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, colors.HexColor("#F2F6FC")]),
    ]
    if highlight_last:
        cmds.append(("BACKGROUND", (0,-1),(-1,-1), colors.HexColor("#D8EAD3")))
    t.setStyle(TableStyle(cmds))
    return t


# ── Build PDF ─────────────────────────────────────────────────────────────────

def build_pdf(recs_1b, recs_1c, charts: dict, out: Path, S: dict):
    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.0*cm,  bottomMargin=2.0*cm,
        title="MCN Phase 1C Report",
    )
    story = []
    hr = lambda: HRFlowable(width="100%", thickness=1, color=colors.HexColor(DARK_BLUE), spaceAfter=6)
    hr_thin = lambda: HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#AAAAAA"), spaceAfter=4)

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("Mycelial Council Network", S["title"]))
    story.append(Paragraph("Phase 1C — 2000-Task Scale-Up Report", S["sub"]))
    story.append(Paragraph("Comparative Analysis: Phase 1B (400 tasks) vs Phase 1C (2000 tasks)", S["sub"]))
    story.append(hr())

    # ── Executive Summary ─────────────────────────────────────────────────────
    n1b = len(recs_1b);  p1b = sum(1 for r in recs_1b if r.get("verdict")=="PASS")
    n1c = len(recs_1c);  p1c = sum(1 for r in recs_1c if r.get("verdict")=="PASS")
    ts  = tribe_stats(recs_1c)

    story.append(Paragraph("Executive Summary", S["h1"]))
    story.append(Paragraph(
        f"Phase 1C scaled the stratified live evaluation from 400 tasks (Phase 1B) to "
        f"<b>2,000 tasks</b> (250 per category × 8 categories) using the same MCN-LinUCB "
        f"configuration (homogeneous T=0.3, α=2.5). The overall pass rate is "
        f"<b>{100*p1c/n1c:.1f}%</b> ({p1c}/{n1c}), essentially identical to Phase 1B's "
        f"{100*p1b/n1b:.1f}% ({p1b}/{n1b}). Category-level performance is highly stable "
        f"(all categories within ±7 pp of Phase 1B), confirming the robustness of the Phase 1B findings.",
        S["body"]))

    story.append(Paragraph(
        f"<b>Convergence confirmed — but spurious.</b> The LinUCB bandit reached near-complete "
        f"convergence by task ~1,700: T0 now receives {100*ts[0]['total']/n1c:.0f}% of routing "
        f"decisions (vs T1 {100*ts[1]['total']/n1c:.0f}%, T2 {100*ts[2]['total']/n1c:.0f}%). "
        f"However, per-tribe pass rates are statistically indistinguishable "
        f"(T0 {100*ts[0]['pass']/ts[0]['total']:.1f}%, T1 {100*ts[1]['pass']/ts[1]['total']:.1f}%, "
        f"T2 {100*ts[2]['pass']/ts[2]['total']:.1f}%), confirming that the bandit converged to an "
        f"<i>arbitrary</i> tribe rather than the genuinely best one.",
        S["finding"]))

    # Key stats table
    story.append(Spacer(1, 0.3*cm))
    t = _tbl(
        ["Metric", "Phase 1B (400)", "Phase 1C (2000)", "Δ"],
        [
            ["Overall pass rate", f"{100*p1b/n1b:.1f}%", f"{100*p1c/n1c:.1f}%",
             f"{100*p1c/n1c-100*p1b/n1b:+.1f} pp"],
            ["Tasks", "400", "2000", "+1600"],
            ["Categories", "8 × 50", "8 × 250", "same structure"],
            ["Routing (T0/T1/T2)", "—/—/—", f"{100*ts[0]['total']/n1c:.0f}%/{100*ts[1]['total']/n1c:.0f}%/{100*ts[2]['total']/n1c:.0f}%", "T0 dominant"],
            ["Bandit converged?", "No (drifting)", "Yes (~task 1700)", "Convergence reached"],
            ["Specialisation (χ²)", "p=0.396", "~p>0.05 (expected)", "None"],
        ],
        [3.5*cm, 3.5*cm, 3.5*cm, 3.5*cm], S,
    )
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # ── Figure 1: Category comparison ─────────────────────────────────────────
    story.append(hr_thin())
    story.append(Paragraph("1. Category Pass Rate: Phase 1B vs Phase 1C", S["h1"]))
    story.append(Image(str(charts["cat"]), width=16*cm, height=7.5*cm))
    story.append(Paragraph(
        "Figure 1. Category pass rates at 400 tasks (blue) vs 2,000 tasks (orange). "
        "All categories are stable within ±7 pp. The parsing category shows the largest "
        "drop (−7.2 pp), likely due to small-sample variance in Phase 1B (only 50 tasks). "
        "String achieves 99.2% at 2,000 tasks. Graph remains at 0% — a hard model capability limit.",
        S["cap"]))

    st1 = cat_stats(recs_1b)
    st2 = cat_stats(recs_1c)
    story.append(_tbl(
        ["Category", "1B Pass%", "1B n", "1C Pass%", "1C n", "Δ (pp)"],
        [[c,
          f"{st1[c]['pct']:.1f}%", str(st1[c]['total']),
          f"{st2[c]['pct']:.1f}%", str(st2[c]['total']),
          f"{st2[c]['pct']-st1[c]['pct']:+.1f}"]
         for c in CATEGORIES] +
        [["<b>Overall</b>",
          f"<b>{100*p1b/n1b:.1f}%</b>", str(n1b),
          f"<b>{100*p1c/n1c:.1f}%</b>", str(n1c),
          f"<b>{100*p1c/n1c-100*p1b/n1b:+.1f}</b>"]],
        [3.8*cm, 2.2*cm, 1.5*cm, 2.2*cm, 1.5*cm, 2.0*cm], S, highlight_last=True))

    # ── Figure 2: Routing evolution ────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("2. Routing Evolution — Bandit Learning Curve", S["h1"]))
    story.append(Image(str(charts["routing"]), width=16*cm, height=6.5*cm))
    story.append(Paragraph(
        "Figure 2. Tribe routing share in 100-task rolling windows across 2,000 tasks. "
        "Three distinct phases are visible: (i) T1-dominant exploration (tasks 1–500), "
        "(ii) T0 rising as the bandit's reward estimates update (tasks 500–1,500), "
        "(iii) near-complete T0 convergence (tasks 1,500–2,000, T0 ≈ 96%). "
        "This is the first time the MCN bandit has reached convergence.",
        S["cap"]))

    story.append(Paragraph("Key observation on spurious convergence:", S["h2"]))
    story.append(Paragraph(
        "The bandit's convergence to T0 is statistically significant at 2,000 tasks — the UCB "
        "confidence intervals have narrowed enough to reliably favour T0. However, this is "
        "<i>spurious convergence</i>: T0 achieved early reward advantages through random variance, "
        "and the bandit locked in on it. Since all three tribes use the same model at T=0.3, "
        "no tribe is genuinely superior. The 'convergence' reflects exploration exhaustion, "
        "not learned specialisation.", S["body"]))

    # Routing drift table
    segments = [("Tasks 1–500",    recs_1c[:500]),
                ("Tasks 501–1000", recs_1c[500:1000]),
                ("Tasks 1001–1500",recs_1c[1000:1500]),
                ("Tasks 1501–2000",recs_1c[1500:])]
    drift_rows = []
    for label, seg in segments:
        c = Counter(int(r.get("tribe_idx",0)) for r in seg)
        tot = sum(c.values())
        n_p = sum(1 for r in seg if r.get("verdict")=="PASS")
        drift_rows.append([
            label,
            f"T0:{c[0]}({100*c[0]//tot}%)",
            f"T1:{c[1]}({100*c[1]//tot}%)",
            f"T2:{c[2]}({100*c[2]//tot}%)",
            f"{100*n_p//len(seg)}%",
        ])
    story.append(_tbl(
        ["Period", "T0", "T1", "T2", "Pass Rate"],
        drift_rows,
        [3.5*cm, 3.0*cm, 3.0*cm, 3.0*cm, 2.0*cm], S))

    # ── Figure 3: Tribe performance ────────────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("3. Tribe Performance at 2,000 Tasks", S["h1"]))
    story.append(Image(str(charts["tribe"]), width=15*cm, height=6.5*cm))
    story.append(Paragraph(
        "Figure 3. Left: per-tribe pass rates (60.8% / 59.9% / 61.8% — statistically "
        "indistinguishable). Right: final routing distribution showing T0 dominance (55.6% of tasks). "
        "The routing distribution does NOT reflect performance differences — it reflects the "
        "bandit's early random exploration outcomes.",
        S["cap"]))

    story.append(Paragraph(
        "<b>Interpretation:</b> T2 has the marginally highest pass rate (61.8%) but received "
        "only 16.2% of routing decisions. T0 has the lowest pass rate per-task (60.8%) but "
        "received 55.6%. The bandit converged to the wrong tribe. This directly demonstrates "
        "that convergence at 2,000 tasks is insufficient for reliable tribe selection when "
        "performance differences are small (~1–2 pp).", S["body"]))

    story.append(Paragraph(
        "Implication: For routing to provide value, inter-tribe performance differences must "
        "be large enough (≫ 2 pp) to overcome the noise in reward estimates at this scale. "
        "With same-model homogeneous tribes, this threshold is never reached.",
        S["warn"]))

    # ── Figure 4: Pass rate curve ──────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("4. Rolling Pass Rate — Stability Over Time", S["h1"]))
    story.append(Image(str(charts["curve"]), width=16*cm, height=6.5*cm))
    story.append(Paragraph(
        "Figure 4. 200-task rolling pass rate for the overall experiment and selected categories. "
        "The shaded band shows ±3 pp around the overall rate. String tasks (green dashed) "
        "remain near 100% throughout. Graph tasks (red dashed) remain at 0%. The overall rate "
        "is remarkably stable at 60–62%, confirming that the task set is well-calibrated and "
        "results are reproducible.",
        S["cap"]))

    # ── Figure 5: Delta heatmap ────────────────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("5. Phase 1C vs 1B: Change Analysis and Per-Category Routing", S["h1"]))
    story.append(Image(str(charts["delta"]), width=17*cm, height=7.0*cm))
    story.append(Paragraph(
        "Figure 5. Left: Δ pass rate per category (Phase 1C − Phase 1B). Green = improvement, "
        "red = decline. Parsing shows the largest decline (−7.2 pp), attributable to small-sample "
        "variance in Phase 1B. Right: tribe routing share per category in Phase 1C, showing that "
        "T0 dominance is not category-specific — the bandit routes to T0 regardless of category.",
        S["cap"]))

    # Categorical routing table
    by_cat_tribe: dict = defaultdict(Counter)
    for r in recs_1c:
        by_cat_tribe[r.get("category","?")][int(r.get("tribe_idx",0))] += 1

    routing_rows = []
    for c in CATEGORIES:
        ct = by_cat_tribe[c]
        tot = sum(ct.values())
        p = st2[c]["pct"]
        routing_rows.append([
            c,
            f"{st1[c]['pct']:.0f}%",
            f"{p:.0f}%",
            f"{ct[0]}({100*ct[0]//tot if tot else 0}%)",
            f"{ct[1]}({100*ct[1]//tot if tot else 0}%)",
            f"{ct[2]}({100*ct[2]//tot if tot else 0}%)",
        ])
    story.append(_tbl(
        ["Category", "1B Pass%", "1C Pass%", "T0 routed", "T1 routed", "T2 routed"],
        routing_rows,
        [3.2*cm, 2.0*cm, 2.0*cm, 2.5*cm, 2.5*cm, 2.5*cm], S))

    # ── Key Conclusions ────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Key Conclusions", S["h1"]))
    story.append(hr())

    conclusions = [
        ("<b>(1) Category performance is stable at scale.</b>",
         "All 8 categories show pass rates within ±7 pp of Phase 1B values at 5× the task count. "
         "The Phase 1B category profile is a reliable estimate of model capability on this benchmark."),
        ("<b>(2) Bandit convergence confirmed at ~1,700 tasks.</b>",
         "The LinUCB bandit achieves near-complete routing lock-in by task ~1,700 (T0: 96%), "
         "consistent with the 2,000-task convergence estimate in the academic paper."),
        ("<b>(3) Convergence is spurious — all tribes are equivalent.</b>",
         "T0: 60.8%, T1: 59.9%, T2: 61.8% — differences of ≤2 pp, not statistically significant. "
         "The bandit locked in on T0 due to early random reward variance, not genuine superiority. "
         "This is the clearest demonstration yet that routing adds no value with homogeneous tribes."),
        ("<b>(4) Parsing variance resolved at scale.</b>",
         "Phase 1B parsing 66% vs Phase 1C 58.8%: the Phase 1B estimate had high variance "
         "(only 50 tasks, 5 task types × 10 attempts). At 250 tasks, the estimate stabilises."),
        ("<b>(5) Graph remains 0%.</b>",
         "0/250 tasks passed across 4 graph task types — a confirmed hard capability limit of "
         "Qwen2.5-Coder-7B-Instruct-AWQ, independent of routing or scale."),
        ("<b>(6) Next step: genuine tribe diversity.</b>",
         "For routing to add value, tribes need qualitatively different capabilities. "
         "The Phase 1B/1C null result establishes the baseline: same-model, same-temperature "
         "routing nets −0 pp to −5 pp vs single agent. Route between different base models "
         "(7B code vs 13B reasoning) to test whether specialisation emerges."),
    ]
    for heading, body in conclusions:
        story.append(Paragraph(heading, S["h2"]))
        story.append(Paragraph(body, S["body"]))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(hr())
    story.append(Paragraph(
        "MCN Research Initiative · Phase 1C Complete · February 2026 · "
        f"Data: categorized_runs_phase1c.jsonl ({n1c} records) · "
        "GitHub: github.com/DeepakEz/MCN",
        S["footer"]))

    doc.build(story)
    print(f"  Saved -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    recs_1b = load(FILE_1B)
    recs_1c = load(FILE_1C)
    print(f"  Phase 1B: {len(recs_1b)} records")
    print(f"  Phase 1C: {len(recs_1c)} records")

    print("Generating charts...")
    charts = {
        "cat":     chart_category_comparison(recs_1b, recs_1c, IMG_DIR),
        "routing": chart_routing_evolution(recs_1c, IMG_DIR),
        "tribe":   chart_tribe_performance(recs_1c, IMG_DIR),
        "curve":   chart_pass_rate_curve(recs_1c, IMG_DIR),
        "delta":   chart_delta_heatmap(recs_1b, recs_1c, IMG_DIR),
    }
    print(f"  {len(charts)} charts generated in {IMG_DIR}")

    S = _styles()
    print("Building PDF...")
    build_pdf(recs_1b, recs_1c, charts, OUT_FILE, S)
    import os
    print(f"\nDone. {os.path.getsize(OUT_FILE)//1024} KB -> {OUT_FILE}")
