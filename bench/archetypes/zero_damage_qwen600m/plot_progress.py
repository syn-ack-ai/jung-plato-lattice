"""Generate publication-grade progress.png for Exp 256 (`Qwen3-0.6B` Zero-Damage Shape Transfer)
using `autoresearch-skill/scripts/style_presets.py`.
"""
import csv
import os
import sys
from pathlib import Path

SKILL_SCRIPTS = os.environ.get("AUTORESEARCH_SCRIPTS_DIR", "")
if SKILL_SCRIPTS and Path(SKILL_SCRIPTS).exists():
    sys.path.insert(0, SKILL_SCRIPTS)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import style_presets as sp
except ImportError:
    class _SP:
        @staticmethod
        def rcparams():
            import matplotlib as mpl
            mpl.rcParams.update({"figure.dpi": 180, "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    sp = _SP()

DIR = Path(__file__).resolve().parent
TSV = DIR / "autoresearch-results.tsv"
OUT = DIR / "progress.png"


def main():
    sp.rcparams()
    rows = []
    if TSV.exists():
        with open(TSV) as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                if row.get("iteration", "").strip() != "":
                    rows.append(row)
    if not rows:
        return

    iters = [int(r["iteration"]) for r in rows]
    scores = [float(r["score"]) for r in rows]
    statuses = [r["status"].strip().lower() for r in rows]
    labels = [r.get("short_label", f"it{i}") for i, r in zip(iters, rows)]

    best_curve = []
    cur = -1e9
    for sc, st in zip(scores, statuses):
        if st in ("keep", "baseline") and sc > cur:
            cur = sc
        best_curve.append(cur if cur > -1e8 else sc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1.3, 1.05]})
    fig.patch.set_facecolor("white")

    ax1.axhline(0.0, color="#64748b", ls="--", lw=1.2, label="Unmodified Base Qwen3-0.6B (0.0)")
    ax1.axhline(3.0, color="#16a34a", ls=":", lw=1.4, label="Target (+3.0 lift, 0 damage)")
    ax1.plot(iters, best_curve, color="#2563eb", lw=2.4, label="Best Kept Trajectory", zorder=3)

    for i, sc, st, lbl in zip(iters, scores, statuses, labels):
        c = "#16a34a" if st == "keep" else ("#f59e0b" if st == "baseline" else "#ef4444")
        mk = "o" if st in ("keep", "baseline") else "x"
        ax1.scatter([i], [sc], color=c, marker=mk, s=68, zorder=4)
        ax1.annotate(f"{lbl}\n({sc:+.2f})", (i, sc), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=7.5, color="#1e293b")

    ax1.set_title("Qwen3-0.6B Shape-Transfer Quality Lift (vs. Base)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Autoresearch Iteration")
    ax1.set_ylabel("Composite Quality Lift Score (Higher = Better)")
    ax1.set_xticks(iters)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right", fontsize=8, frameon=True)

    d_nll_gain = [-100.0 * float(r["d_nll"]) for r in rows]
    d_code_u = [100.0 * float(r["d_code_units"]) for r in rows]
    d_swe_pp = [100.0 * float(r["d_swe"]) for r in rows]
    d_proc_pp = [100.0 * float(r["d_proc"]) for r in rows]
    d_al_pts = [100.0 * float(r["d_align_peak"]) for r in rows]

    ax2.axhline(0.0, color="#64748b", ls="--", lw=1.0)
    ax2.plot(iters, d_code_u, marker="s", lw=2.0, color="#0ea5e9", label="Exec Unit Tests Δ (pp)")
    ax2.plot(iters, d_swe_pp, marker="p", lw=1.8, color="#ec4899", label="SWE Bug-Detect Δ (pp)")
    ax2.plot(iters, d_proc_pp, marker="^", lw=1.8, color="#8b5cf6", label="Proc MCQ Δ (pp)")
    ax2.plot(iters, d_nll_gain, marker="o", lw=1.8, color="#10b981", label="Fluency Gain (-100·ΔNLL)")
    ax2.plot(iters, d_al_pts, marker="d", lw=1.5, ls="--", color="#f97316", label="Blind Rival Align Δ (pts)")
    ax2.set_title("Component Deltas vs. Unmodified Base", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Autoresearch Iteration")
    ax2.set_ylabel("Delta vs. Base (pp / pts)")
    ax2.set_xticks(iters)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=7.5, frameon=True)

    plt.tight_layout()
    plt.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
