#!/usr/bin/env python3
"""Refresh `progress.png` from `autoresearch-results.tsv` using autoresearch-skill style presets."""
import csv
import os
import sys
from pathlib import Path

SKILL_SCRIPTS = os.environ.get("AUTORESEARCH_SCRIPTS_DIR", "")
if SKILL_SCRIPTS and Path(SKILL_SCRIPTS).exists():
    sys.path.insert(0, SKILL_SCRIPTS)
    from style_presets import rcparams, OKABE_ITO  # noqa: E402
else:
    OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    def rcparams():
        import matplotlib as mpl
        mpl.rcParams.update({"figure.dpi": 180, "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
import matplotlib.pyplot as plt  # noqa: E402


def main():
    root = Path(__file__).resolve().parent
    tsv = root / "autoresearch-results.tsv"
    rows = []
    with open(tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                it = int(r["iteration"])
                val = float(r["metric_value"])
                rows.append((it, val, r["status"].strip(), r["description"].strip()))
            except Exception:
                continue
    if not rows:
        return

    rcparams()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    best_env = []
    cur_best = -1e9
    for it, val, st, _ in rows:
        if st in ("baseline", "kept") and val > cur_best:
            cur_best = val
        best_env.append(cur_best if cur_best > -1e8 else val)

    # Best-so-far envelope
    ax.plot(xs, best_env, color=OKABE_ITO[4], linewidth=2.2, label="Best-so-far envelope", zorder=2)

    # Kept / Baseline filled markers; Reverted / Guard hollow markers
    kept_x = [r[0] for r in rows if r[2] in ("baseline", "kept")]
    kept_y = [r[1] for r in rows if r[2] in ("baseline", "kept")]
    rev_x = [r[0] for r in rows if r[2] not in ("baseline", "kept")]
    rev_y = [r[1] for r in rows if r[2] not in ("baseline", "kept")]

    if kept_x:
        ax.scatter(kept_x, kept_y, color=OKABE_ITO[2], s=72, zorder=4, edgecolors="black", linewidths=0.8, label="Kept / Baseline")
    if rev_x:
        ax.scatter(rev_x, rev_y, facecolors="white", edgecolors=OKABE_ITO[5], s=68, linewidths=1.5, zorder=3, label="Reverted / Guard fail")

    ax.axhline(0.95, color=OKABE_ITO[5], linestyle="--", linewidth=1.3, label="Target (0.95)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Composite Hardened Score (0–1)")
    ax.set_title("Autoresearch: Dual-Stratum Crystal Memory & Firewall")
    ax.set_ylim(max(0.0, min(ys) - 0.08), 1.03)
    ax.set_xticks(xs)
    ax.legend(loc="lower right", fontsize=10)

    out = root / "progress.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"UPDATED_PROGRESS_PNG: {out}", flush=True)


if __name__ == "__main__":
    main()
