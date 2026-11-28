"""Generate clean, honest 3-panel summary figure (`exp257_258_rigorous_summary.png`)
for the Exp 257 (5-seed causal ladder on untouched 3rd-party benchmarks) + Exp 258
(5-family Python-Coding Platonic Lattice) + Fixed Living Crystal 5-Baseline audit.
All error bars show sample 95% CIs (`ddof=1`, `t_{df=4}`).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
C257 = json.loads((ROOT / "causal_ladder_summary.json").read_text())
C258_OOF = json.loads((ROOT / "code_lattice_blind_oof_report.json").read_text())
C258_DEV = json.loads((ROOT / "dev_code_autoresearch_results.json").read_text())
LC = json.loads((ROOT / "living_memory_rigorous_baselines.json").read_text())


def main():
    plt.rcParams.update({
        "figure.dpi": 200,
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), gridspec_kw={"width_ratios": [1.25, 1.15, 1.05]})
    fig.patch.set_facecolor("white")

    # ---- Panel A: Exp 257 5-Seed Paired Deltas vs Base (ddof=1 95% CIs) ----
    ax = axes[0]
    arms = [
        ("true_stratified_r3", "True Stratified R3", "#0072B2"),
        ("true_gut_only", "True Gut-Only (0 proc)", "#009E73"),
        ("random_lora_matched", "Random LoRA (||ΔW|| matched)", "#E69F00"),
        ("shuffle_within_domain", "Within-Dom Shuffle", "#D55E00"),
    ]
    metrics = [
        ("proc_acc_200", "Proc-200\n(pp)", 100.0),
        ("swe_full_acc", "SWE-240\n(pp)", 100.0),
        ("humaneval_pass1", "HumanEval\n(pp)", 100.0),
        ("mbpp_test_pass1", "MBPP-257\n(pp)", 100.0),
    ]
    x = np.arange(len(metrics))
    w = 0.19
    ax.axhline(0.0, color="#475569", lw=1.1, ls="--")
    for idx, (ak, lbl, col) in enumerate(arms):
        means, errs = [], []
        for mk, _, scale in metrics:
            s = C257["arms"][ak]["delta_vs_base"][mk]
            means.append(s["mean"] * scale)
            errs.append(0.5 * (s["ci95_hi"] - s["ci95_lo"]) * scale)
        off = (idx - 1.5) * w
        ax.bar(x + off, means, width=w, color=col, alpha=0.88, label=lbl, yerr=errs, capsize=3, error_kw={"lw": 1.1})
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("Δ vs. Unmodified Base (pp, 5-seed 95% CI)")
    ax.set_title("A. Exp 257: 5-Seed Causal Ladder (Exact FP32)", fontweight="bold")
    ax.legend(fontsize=7.8, loc="lower left", frameon=True)

    # ---- Panel B: Exp 258 5-Family Python-Coding Platonic Lattice ----
    ax = axes[1]
    depths = ["30%", "50%", "70%"]
    xd = np.arange(len(depths))
    series = [
        ("GPA_R3_Consensus", "3-Family GPA (G4+L3.1+Mis)", "#0072B2", "o", "-"),
        ("llama31_8b", "Llama 3.1 8B (single)", "#009E73", "s", "--"),
        ("mistral7b", "Mistral 7B (single)", "#E69F00", "^", "--"),
        ("gemma4_e4b", "Gemma 4 E4B (single)", "#D55E00", "d", "--"),
    ]
    for key, lbl, col, mk, ls in series:
        ys = [C258_OOF["by_depth"][f"depth_{d}"]["blind_oof"][key]["mean_r2_frob"] for d in (30, 50, 70)]
        ax.plot(xd, ys, color=col, marker=mk, ls=ls, lw=2.1, ms=6.5, label=lbl)
    # Add Base Qwen3-0.6B and H2 Transplanted Qwen3-0.6B!
    q_base = [C258_DEV["base"][f"code_blind_{d}_r2"] for d in (30, 50, 70)]
    h2 = [r for r in C258_DEV["runs"] if r["name"] == "H2_r3_tri_trustkl"][0]
    q_h2 = [h2["eval"][f"code_blind_{d}_r2"] for d in (30, 50, 70)]
    ax.plot(xd, q_base, color="#64748b", marker="x", ls=":", lw=2.0, ms=7, label="Base Qwen3-0.6B")
    ax.plot(xd, q_h2, color="#CC79A7", marker="*", ls="-", lw=2.3, ms=9, label="Qwen3-0.6B + Code GPA (H2, ce=0)")
    ax.set_xticks(xd)
    ax.set_xticklabels(depths)
    ax.set_ylim(0.22, 0.76)
    ax.set_ylabel("Blind OOF Procrustes R²_frob (Phi-4 & OLMo 3)")
    ax.set_title("B. Exp 258: Python-Coding Platonic Lattice", fontweight="bold")
    ax.legend(fontsize=7.3, loc="lower left", frameon=True)

    # ---- Panel C: Honest Living Crystal 5-Baseline Comparison ----
    ax = axes[2]
    m_order = [
        ("no_memory", "No Mem"),
        ("full_context", "Full Ctx\n(+235 tok)"),
        ("bm25_rag", "BM25 RAG\n(+8.5 tok)"),
        ("l8_text_rag", "L8 RAG\n(+46 tok)"),
        ("l8_zero_text_kv", "L8 0-Text\nKV (0 tok)"),
    ]
    xm = np.arange(len(m_order))
    c1_ret = [int(LC["modes"][k]["c1_ret_hit"].split("/")[0]) / 10.0 * 100.0 for k, _ in m_order]
    tot_code = [int(LC["modes"][k]["total_code"].split("/")[0]) / 18.0 * 100.0 for k, _ in m_order]
    ax.bar(xm - 0.18, c1_ret, width=0.35, color="#56B4E9", label="C1 Zero-KW Ret@2 (%)")
    ax.bar(xm + 0.18, tot_code, width=0.35, color="#0072B2", label="Total Exec Code (18 tasks, %)")
    for i, (r_v, c_v) in enumerate(zip(c1_ret, tot_code)):
        ax.text(i - 0.18, r_v + 1.8, f"{r_v:.0f}%", ha="center", fontsize=7.5)
        ax.text(i + 0.18, c_v + 1.8, f"{c_v:.0f}%", ha="center", fontsize=7.5)
    ax.set_xticks(xm)
    ax.set_xticklabels([lbl for _, lbl in m_order], fontsize=8.0)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Score (%)")
    ax.set_title("C. Fixed Living Memory Baselines (Qwen3-0.6B)", fontweight="bold")
    ax.legend(fontsize=7.8, loc="upper right", frameon=True)

    fig.suptitle(
        "Rigorous Post-Audit Re-Test (Exp 257–258): Proven vs. Null Effects Under Exact FP32 & Fair Baselines",
        fontsize=12.0, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = ROOT / "exp257_258_rigorous_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("SAVED_FIGURE:", out)


if __name__ == "__main__":
    main()
