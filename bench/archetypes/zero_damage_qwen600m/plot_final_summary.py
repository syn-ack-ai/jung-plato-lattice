"""Generate publication-grade 3-panel summary figure `exp256_final_summary.png`
using `autoresearch-skill/scripts/style_presets.py`:
  Panel A: 17-Iteration Autoresearch Optimization Trajectory (Iter 0 -> 16)
  Panel B: Paper-Grade Causal Falsification (SHUFFLED vs Dual-Domain TRUE 3-Seed vs Pure-Prose TRUE 2-Seed)
  Panel C: 4-Arm Living Crystal Coding Agent Before/After + Red-Team Security Defense
"""
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np

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
CF_JSON = DIR / "causal_falsification_3seed.json"
LC_JSON = DIR / "living_crystal_coding_4arm.json"
OUT = DIR / "exp256_final_summary.png"


def main():
    sp.rcparams()
    rows = []
    with open(TSV) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("iteration", "").strip() != "":
                rows.append(row)

    cf = json.loads(CF_JSON.read_text())
    lc = json.loads(LC_JSON.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), gridspec_kw={"width_ratios": [1.25, 1.15, 1.05]})
    fig.patch.set_facecolor("white")

    # ---- Panel A: Autoresearch Trajectory ----
    ax = axes[0]
    iters = [int(r["iteration"]) for r in rows]
    scores = [float(r["score"]) for r in rows]
    guards = [r["guard_ok"].strip() == "True" for r in rows]
    passes = [r["pass"].strip() == "True" for r in rows]

    best_guarded = []
    cur = -1e9
    for sc, g in zip(scores, guards):
        if g and sc > cur:
            cur = sc
        best_guarded.append(cur if cur > -1e8 else np.nan)

    ax.axhline(0.0, color="#64748b", ls="--", lw=1.1, label="Unmodified Base (0.0)")
    ax.axhline(3.0, color="#16a34a", ls=":", lw=1.3, label="Pass Target (+3.0, 0 dmg)")
    ax.plot(iters, best_guarded, color="#2563eb", lw=2.3, label="Best Guarded Frontier", zorder=2)

    for i, sc, g, p in zip(iters, scores, guards, passes):
        if p:
            ax.scatter([i], [sc], color="#16a34a", marker="*", s=160, zorder=5, edgecolors="#052e16", linewidths=0.7)
        elif g:
            ax.scatter([i], [sc], color="#0284c7", marker="o", s=54, zorder=4)
        else:
            ax.scatter([i], [sc], color="#ef4444", marker="x", s=54, zorder=3)

    ax.annotate("Iter 0\nNaive (-2.5 al)", (0, scores[0]), textcoords="offset points", xytext=(8, -22), fontsize=7, color="#dc2626")
    ax.annotate("Iter 6\nWhole-28\n(+11.7)", (6, scores[6]), textcoords="offset points", xytext=(-8, 6), fontsize=7, color="#0369a1", ha="center")
    ax.annotate("Iter 7\n1st PASS\n(+4.3)", (7, scores[7]), textcoords="offset points", xytext=(4, -26), fontsize=7, color="#15803d", ha="center")
    ax.annotate("Iter 14\nPeak\n(+12.1)", (14, scores[14]), textcoords="offset points", xytext=(-4, 6), fontsize=7, color="#0369a1", ha="center")
    ax.annotate("Iter 15\nCHAMPION\nPASS (+10.1)", (15, scores[15]), textcoords="offset points", xytext=(-18, -32), fontsize=7.2, fontweight="bold", color="#15803d", ha="center")

    ax.set_title("A. Autoresearch Loop on Qwen3-0.6B (ce_w=0)", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Iteration (0 = Naive Exp 251 -> 16)")
    ax.set_ylabel("Composite Quality Lift Score")
    ax.set_xticks(range(0, 17, 2))
    ax.set_ylim(-9, 15.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=7.5, frameon=True)

    # ---- Panel B: Paper-Grade Causal Falsification (True Shape vs Shuffled Control) ----
    ax = axes[1]
    arm_order = ["Iter15_DualDomain_SHUFFLED", "Iter15_DualDomain_TRUE", "Iter14_PureProse_TRUE"]
    arm_labels = ["Shuffled Shape\nControl (n=2)", "Dual-Domain\nTRUE (n=3)", "Pure-Prose\nTRUE (n=2)"]
    arm_colors = ["#94a3b8", "#16a34a", "#2563eb"]

    metrics = [
        ("Composite\nScore", lambda r: r["score"]),
        ("Exec Units\nΔ (pp)", lambda r: 100.0 * r["deltas"]["d_code_units"]),
        ("Exec Pass@1\nΔ (pp)", lambda r: 100.0 * r["deltas"]["d_code_pass1"]),
        ("Fluency Gain\n(-1000·ΔNLL)", lambda r: -1000.0 * r["deltas"]["d_nll"]),
        ("Blind L14\nΔ (pts)", lambda r: 100.0 * r["deltas"]["d_align_50"]),
    ]

    x = np.arange(len(metrics))
    w = 0.26
    for ai, (arm_k, lbl, col) in enumerate(zip(arm_order, arm_labels, arm_colors)):
        sub = [r for r in cf["runs"] if r["arm"] == arm_k]
        means = [np.mean([fn(r) for r in sub]) for _, fn in metrics]
        stds = [np.std([fn(r) for r in sub]) for _, fn in metrics]
        ax.bar(x + (ai - 1) * w, means, width=w, yerr=stds, capsize=3, color=col, alpha=0.90, label=lbl, edgecolor="#1e293b", linewidth=0.5)

    ax.axhline(0.0, color="#334155", lw=1.0)
    ax.set_title("B. Causal Falsification: True Shape vs. Shuffled (ce_w=0)", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=8)
    ax.set_ylabel("Lift vs. Unmodified Base Qwen3-0.6B")
    ax.set_ylim(-5.0, 14.8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper center", ncol=3, fontsize=7.0, frameon=True)

    # ---- Panel C: 4-Arm Living Crystal Coding Agent + Red-Team ----
    ax = axes[2]
    arms = lc["arms"]
    cats = ["C1 Zero-KW\nCode (10)", "C2 2-Hop\nCode (4)", "C3 Live\nUpdate (4)", "C4 Attacks\nBlocked (16)"]
    arm_disp = [
        ("Arm1_Base_NoCrystal", "Base (No Crystal)", "#cbd5e1"),
        ("Arm2_Base_LivingCrystal", "Base + Crystal", "#38bdf8"),
        ("Arm4_Realigned_LivingCrystal", "Iter15 + Crystal", "#16a34a"),
    ]
    amap = {a["arm"]: a for a in arms}
    xc = np.arange(len(cats))
    wc = 0.25
    for bi, (ak, lbl, col) in enumerate(arm_disp):
        a = amap[ak]
        vals = [
            100.0 * a["c1_code_rate"],
            100.0 * a["c2_code_rate"],
            100.0 * a["c3_code_rate"],
            100.0 * float(a["c4_redteam"].get("attacks_blocked_rate", 0.0)),
        ]
        ax.bar(xc + (bi - 1) * wc, vals, width=wc, color=col, alpha=0.92, label=lbl, edgecolor="#1e293b", linewidth=0.5)

    ax.set_title("C. Living Crystal Coding Agent & Red-Team", fontsize=10.5, fontweight="bold")
    ax.set_xticks(xc)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("Pass / Defense Rate (%)")
    ax.set_ylim(0, 115)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True)

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
