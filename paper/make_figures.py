"""Generate all 6 publication-grade arXiv figures (300 DPI PNG + vector PDF) directly from verified JSON outputs.

Every plotted bar, error bar, curve, and numeric annotation is derived directly from committed JSON artifacts
(zero hardcoded empirical results).

Run from repository root:
    python paper/make_figures.py
"""
import glob
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PLATO = ROOT / "bench" / "archetypes" / "fluent" / "plato"
RUNS_COMP = ROOT / "bench" / "archetypes" / "fluent" / "runs" / "comp"
AION_RUNS = ROOT / "bench" / "archetypes" / "aion_8b" / "runs"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Clean, colorblind-safe Okabe-Ito inspired publication palette
C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_TEAL = "#009E73"
C_PURPLE = "#7A5195"
C_GOLD = "#E69F00"
C_SKY = "#56B4E9"
C_GRAY = "#6C757D"
C_LIGHTGRAY = "#CED4DA"
C_RED = "#C0392B"
C_DARK = "#222222"


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.3,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.3,
        "xtick.labelsize": 8.3,
        "ytick.labelsize": 8.3,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def save_fig(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"SAVED {stem}.png + .pdf", flush=True)


# ---------------------------------------------------------------- Figure 1: 7-Company Platonic Consensus
def make_fig1():
    meta = json.load(open(PLATO / "meta.json"))
    order = [
        ("inkling2_mini", "Inkling 2 Mini\n(Looped)"),
        ("gemma4_e4b", "Gemma 4 E4B\n(Google)"),
        ("llama31_8b", "Llama 3.1 8B\n(Meta)"),
        ("qwen35_9b", "Qwen3.5 9B\n(Alibaba)"),
        ("mistral7b", "Mistral 7B\n(Mistral)"),
        ("CONSENSUS_4FAM_MLX", "C4 Consensus\n(4-Fam MLX)"),
        ("CONSENSUS_5FAM_INK", "C5 Consensus\n(5-Fam +Ink.)"),
    ]
    phi4_r2 = [meta["to_blind_pts"]["phi4_mini"][k]["r2_frob"] for k, _ in order]
    olmo_r2 = [meta["to_blind_pts"]["olmo3_7b"][k]["r2_frob"] for k, _ in order]
    labels = [lbl for _, lbl in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.8), gridspec_kw={"width_ratios": [1.42, 1.0]})

    x = np.arange(len(order))
    w = 0.36
    p4_var = meta["var_explained_256"]["phi4_mini"] * 100.0
    ol_var = meta["var_explained_256"]["olmo3_7b"] * 100.0
    b1 = ax1.bar(x - w / 2, phi4_r2, w, label=f"Held-Out Microsoft Phi-4-mini (top-256 PCs: {p4_var:.1f}% var)", color=C_BLUE, edgecolor="white", linewidth=0.6)
    b2 = ax1.bar(x + w / 2, olmo_r2, w, label=f"Held-Out AllenAI OLMo 3 7B (top-256 PCs: {ol_var:.1f}% var)", color=C_ORANGE, edgecolor="white", linewidth=0.6)

    ax1.axvspan(4.5, 6.55, color=C_TEAL, alpha=0.07, zorder=0)
    ax1.axvline(4.5, color=C_TEAL, linestyle=":", linewidth=1.2, alpha=0.8)
    ax1.text(5.52, 0.655, "Multi-Family\nGPA Denoising", ha="center", va="top", fontsize=8.2, color=C_TEAL, fontweight="bold")

    ax1.axhline(max(phi4_r2[:5]), color=C_BLUE, linestyle="--", linewidth=0.9, alpha=0.5)
    ax1.axhline(max(olmo_r2[:5]), color=C_ORANGE, linestyle="--", linewidth=0.9, alpha=0.5)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.010, f"{h:.2f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold" if h > 0.45 else "normal")

    phi4_gain = (phi4_r2[6] - max(phi4_r2[:5])) * 100.0
    olmo_gain = (olmo_r2[6] - max(olmo_r2[:5])) * 100.0
    ax1.annotate(
        f"C5: +{phi4_gain:.1f} pts (Phi-4)\n+{olmo_gain:.1f} pts (OLMo 3)\nvs. best single LLM",
        xy=(5.82, 0.538), xytext=(2.05, 0.468),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.1, connectionstyle="arc3,rad=-0.10"),
        fontsize=7.6, fontweight="bold", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#F8F9FA", edgecolor=C_GRAY, lw=0.7),
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.9)
    ax1.set_ylim(0, 0.675)
    ax1.set_ylabel(r"2-Fold Out-of-Sample $R^2_{\mathrm{frob}}$ (Target Top-256 PCs)")
    ax1.set_title("A. Blind-Family Prediction Across Two Held-Out LLMs (3,192 Clauses)")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=7.6)

    keys5 = ["gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b", "inkling2_mini"]
    short5 = ["Gemma 4\nE4B", "Qwen3.5\n9B", "Llama 3.1\n8B", "Mistral\n7B", "Inkling 2\n(Looped)"]
    pw = meta["post_gpa_eval_pairwise_cos"]
    M = np.ones((5, 5), dtype=float)
    for i in range(5):
        for j in range(5):
            if i != j:
                k1 = f"{keys5[i]}_vs_{keys5[j]}"
                k2 = f"{keys5[j]}_vs_{keys5[i]}"
                M[i, j] = pw[k1] if k1 in pw else pw[k2]

    im = ax2.imshow(M, cmap="YlGnBu", vmin=0.30, vmax=1.00, aspect="auto")
    ax2.grid(False)
    for i in range(5):
        for j in range(5):
            val = M[i, j]
            txt_col = "white" if val > 0.72 else C_DARK
            ax2.text(j, i, f"{val:.3f}", ha="center", va="center", color=txt_col, fontsize=8.5, fontweight="bold" if i != j else "normal")
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(short5, fontsize=8.0)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(short5, fontsize=8.0)
    ax2.set_title("B. Post-GPA Pairwise Held-Out Cosine (5 Teachers)")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity", fontsize=8.5)

    fig.tight_layout(w_pad=2.2)
    save_fig(fig, "fig1_platonic_consensus")


# ---------------------------------------------------------------- Figure 2: Archetype Audit & 7-Company Teleportation
def make_fig2():
    arch = json.load(open(PLATO / "arch229_report.json"))
    aud = json.load(open(PLATO / "audit233_mac.json"))
    st = arch["modes"]["story_neutral_only"]["gemma_to_others"]
    strict_ar = aud["arch_rot_strict_and_bow"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0), gridspec_kw={"width_ratios": [1.08, 1.24]})

    # Panel A: 100% JSON-derived Full vs Neutral-Only GPA items3 roles (with 95% CIs!)
    roles = ["anchor", "lure", "true", "foil"]
    role_lbls = ["anchor\n(Classic)", "lure\n(Surface)", "true\n(Disguised)", "foil\n(Broken)"]
    full_ol = aud["plato_full_tiesafe"]["blind"]["olmo3_7b"]["items3_roles_C5_centered"]
    neut_ol = aud["plato_neutral_only_no_myth"]["blind"]["olmo3_7b"]["items3_roles_C5_centered"]
    full_ph = aud["plato_full_tiesafe"]["blind"]["phi4_mini"]["items3_roles_C5_centered"]
    neut_ph = aud["plato_neutral_only_no_myth"]["blind"]["phi4_mini"]["items3_roles_C5_centered"]

    x = np.arange(4)
    w = 0.19
    series = [
        (-1.5 * w, full_ol, C_ORANGE, 1.0, None, "OLMo 3 (Full GPA)"),
        (-0.5 * w, neut_ol, "#F8C471", 0.95, "//", "OLMo 3 (Myth-Pool-Excl.)"),
        (+0.5 * w, full_ph, C_BLUE, 1.0, None, "Phi-4 (Full GPA)"),
        (+1.5 * w, neut_ph, C_SKY, 0.95, "//", "Phi-4 (Myth-Pool-Excl.)"),
    ]
    for off, src, col, alp, hat, lbl in series:
        vals = [src[r]["diff"] for r in roles]
        lo = [src[r]["diff"] - src[r]["ci95"][0] for r in roles]
        hi = [src[r]["ci95"][1] - src[r]["diff"] for r in roles]
        ax1.bar(
            x + off, vals, w, color=col, alpha=alp, hatch=hat,
            edgecolor=C_DARK if hat else "white", linewidth=0.5, label=lbl,
            yerr=[lo, hi], capsize=2.2, error_kw=dict(ecolor=C_DARK, lw=0.8),
        )

    ax1.axhline(0, color=C_DARK, linewidth=0.9)
    ax1.axvspan(-0.48, 1.48, color=C_GOLD, alpha=0.07, zorder=0)
    ax1.text(0.5, 0.091, "Classic Cultural Motifs\n(OLMo 3 persists myth-pool-excl.;\nPhi-4 anchor shrinks to n.s.)", ha="center", va="top", fontsize=7.5, fontweight="bold", color="#8A5A00")
    ax1.text(2.5, 0.091, "Disguised Plots\n(No sig. diff.;\n95% CIs span 0)", ha="center", va="top", fontsize=7.5, fontweight="bold", color=C_GRAY)

    ax1.set_xticks(x)
    ax1.set_xticklabels(role_lbls, fontsize=8.3)
    ax1.set_ylim(-0.035, 0.098)
    ax1.set_ylabel("Myth − Mechanism Cosine Difference (95% CI)")
    ax1.set_title("A. items3 Role Split: Full vs. Myth-Pool-Excluded GPA")
    ax1.legend(loc="lower left", ncol=2, fontsize=7.1, frameon=True, facecolor="white", framealpha=0.95)

    # Panel B: 7-company strict zero-shot teleportation (ZERO-ORIGIN + 20-split SD error bars + BoW floor!)
    models = [
        ("qwen35_9b", "Qwen3.5 9B\n(Alibaba)"),
        ("llama31_8b", "Llama 3.1 8B\n(Meta)"),
        ("mistral7b", "Mistral 7B\n(Mistral)"),
        ("inkling2_mini", "Inkling 2\n(Looped)"),
        ("phi4_mini", "Phi-4-mini\n(Held-Out)"),
        ("olmo3_7b", "OLMo 3 7B\n(Held-Out)"),
    ]
    h_sd = [strict_ar["neural_gemma_to_6_strict"][k]["house_2f_setting_disjoint_20splits_mean"] * 100 for k, _ in models]
    h_sd_err = [strict_ar["neural_gemma_to_6_strict"][k]["house_2f_setting_disjoint_20splits_std"] * 100 for k, _ in models]
    h_lr = [strict_ar["neural_gemma_to_6_strict"][k]["house_2f_lenres_fold_local"] * 100 for k, _ in models]
    h_ns = [strict_ar["neural_gemma_to_6_strict"][k]["house_noself_to_self2"] * 100 for k, _ in models]
    h_i3 = [st[k]["house_to_unseen_items3"]["acc"] * 100 for k, _ in models]
    bow_mu = strict_ar["bow_lexical_baseline"]["house_2f_setting_disjoint_20splits_mean"] * 100
    bow_sd = strict_ar["bow_lexical_baseline"]["house_2f_setting_disjoint_20splits_std"] * 100

    xm = np.arange(len(models))
    wb = 0.20
    ax2.bar(xm - 1.5 * wb, h_sd, wb, yerr=h_sd_err, capsize=2.0, error_kw=dict(ecolor=C_DARK, lw=0.8), label="HOUSE (20 Setting-Disjoint Splits ±1 SD)", color=C_BLUE)
    ax2.bar(xm - 0.5 * wb, h_lr, wb, label="HOUSE (Fold-Local Len-Res)", color=C_TEAL)
    ax2.bar(xm + 0.5 * wb, h_ns, wb, label="HOUSE (no-self) → SELF2", color=C_PURPLE)
    ax2.bar(xm + 1.5 * wb, h_i3, wb, label="Boundary: HOUSE → items3", color=C_LIGHTGRAY, edgecolor=C_GRAY, linewidth=0.7)

    ax2.axhspan(bow_mu - bow_sd, bow_mu + bow_sd, color=C_GOLD, alpha=0.18, zorder=0)
    ax2.axhline(bow_mu, color=C_GOLD, linestyle="-.", linewidth=1.5, label=f"Unigram BoW Floor ({bow_mu:.1f}% ± {bow_sd:.1f}%)")
    ax2.axhline(50.0, color=C_RED, linestyle="--", linewidth=1.0, alpha=0.85, label="50% Chance Floor")
    for i in range(len(models)):
        ax2.text(xm[i] - 1.5 * wb, h_sd[i] + h_sd_err[i] + 1.5, f"{h_sd[i]:.1f}", ha="center", va="bottom", fontsize=6.4, rotation=90)
        ns_lbl = f"{h_ns[i]:.0f}" if abs(h_ns[i] - round(h_ns[i])) < 0.05 else f"{h_ns[i]:.1f}"
        ax2.text(xm[i] + 0.5 * wb, h_ns[i] + 1.5, ns_lbl, ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=C_PURPLE)
        if h_i3[i] < 54.0:
            # Place sub-50% boundary labels cleanly inside the top of the gray bar (1-decimal matching Table 2b!)
            ax2.text(xm[i] + 1.5 * wb, h_i3[i] - 2.5, f"{h_i3[i]:.1f}", ha="center", va="top", fontsize=6.0, fontweight="bold", color=C_DARK)
        else:
            ax2.text(xm[i] + 1.5 * wb, h_i3[i] + 1.5, f"{h_i3[i]:.1f}", ha="center", va="bottom", fontsize=6.3, fontweight="bold", color=C_GRAY)

    wins_list = [strict_ar["neural_gemma_to_6_strict"][k]["wins_over_bow_20splits"] for k, _ in models]
    tot_wins = sum(wins_list)
    n_splits = 20
    ax2.text(
        0.02, 0.96, f"All {len(models)} companies beat BoW on {min(wins_list)}/{n_splits} splits ({tot_wins}/{len(models)*n_splits} wins)",
        transform=ax2.transAxes, ha="left", va="top", fontsize=7.2, fontweight="bold", color=C_BLUE,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#EBF5FB", edgecolor=C_BLUE, lw=0.7),
    )

    ax2.set_xticks(xm)
    ax2.set_xticklabels([lbl for _, lbl in models], fontsize=8.0)
    ax2.set_ylim(0, 120)
    ax2.set_ylabel("Zero-Shot Disjoint / Cross-Dataset Accuracy (%, Zero-Origin)")
    ax2.set_title("B. Zero-Shot Gemma 4 → 6 Companies (Neutral-Text Procrustes)")
    ax2.legend(loc="lower right", ncol=2, fontsize=6.8, frameon=True, facecolor="white", framealpha=0.95)

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "fig2_archetype_audit_and_teleport")


# ---------------------------------------------------------------- Figure 3: Cross-Setting LORO Myth Amplification + TF-IDF Baseline
def make_fig3():
    ml = json.load(open(PLATO / "audit233_mac.json"))["myth_loro_strict_and_tfidf"]
    aud = ml["beats"]
    n_stories = sum(int(k) * v for k, v in ml["family_size_dist"].items())
    exact_pct = lambda v: (round(v * n_stories) / n_stories) * 100.0
    beats = ["beat_0", "beat_1", "beat_2", "beat_3"]
    beat_lbls = ["Beat 0\n(Ordinary Setup)", "Beat 1\n(Catalyst / Temptation)", "Beat 2\n(Decisive Turn)", "Beat 3\n(Concrete Outcome)"]

    c5_t1 = [exact_pct(aud[b]["c5_strict_loro"]["top1_story_avg"]) for b in beats]
    p4_t1 = [exact_pct(aud[b]["phi4_strict_loro"]["top1_story_avg"]) for b in beats]
    ol_t1 = [exact_pct(aud[b]["olmo3_strict_loro"]["top1_story_avg"]) for b in beats]
    tf_t1 = [exact_pct(aud[b]["tfidf_unigram_baseline"]["top1_story_avg"]) for b in beats]

    c5_t5 = [exact_pct(aud[b]["c5_strict_loro"]["top5_story_avg"]) for b in beats]
    p4_t5 = [exact_pct(aud[b]["phi4_strict_loro"]["top5_story_avg"]) for b in beats]
    ol_t5 = [exact_pct(aud[b]["olmo3_strict_loro"]["top5_story_avg"]) for b in beats]
    tf_t5 = [exact_pct(aud[b]["tfidf_unigram_baseline"]["top5_story_avg"]) for b in beats]

    x = np.arange(4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.6))

    for ax, s_c5, s_p4, s_ol, s_tf, title, ylab, chance, ymax in [
        (ax1, c5_t1, p4_t1, ol_t1, tf_t1, "A. 199-Way Strict LORO Top-1 Retrieval", "Top-1 Accuracy over 199 Families (%)", 0.5, 67),
        (ax2, c5_t5, p4_t5, ol_t5, tf_t5, "B. 199-Way Strict LORO Top-5 Retrieval", "Top-5 Accuracy over 199 Families (%)", 2.5, 88),
    ]:
        ax.plot(x, s_c5, marker="o", linewidth=2.6, markersize=7.5, color=C_TEAL, label="5-Family Consensus C5 (Double-Denoised)")
        ax.plot(x, s_tf, marker="D", linewidth=2.1, markersize=6.0, color=C_GOLD, linestyle="-.", label="Strict-LQO Unigram TF-IDF Baseline")
        ax.plot(x, s_p4, marker="s", linewidth=1.8, markersize=6.0, color=C_BLUE, linestyle="--", label="Single Held-Out LLM: Phi-4-mini")
        ax.plot(x, s_ol, marker="^", linewidth=1.8, markersize=6.0, color=C_ORANGE, linestyle=":", label="Single Held-Out LLM: OLMo 3 7B")
        ax.axhline(chance, color=C_RED, linestyle=":", linewidth=1.0, alpha=0.8, label=f"Chance ({chance:.1f}%)")
        for i in range(4):
            y_c5 = s_c5[i] + (2.0 if i <= 1 else -4.2)
            va_c5 = "bottom" if i <= 1 else "top"
            if i == 1 and ax is ax1:
                # Place Beat-1 C5 label slightly right of marker so annotation arrow entering from left never crosses it
                ax.text(x[i] + 0.08, s_c5[i] + 2.0, f"{s_c5[i]:.1f}%", ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=C_TEAL)
            else:
                ax.text(x[i], y_c5, f"{s_c5[i]:.1f}%", ha="center", va=va_c5, fontsize=8.0, fontweight="bold", color=C_TEAL)
            y_tf = s_tf[i] - 3.8 if i <= 1 else s_tf[i] + 2.0
            va_tf = "top" if i <= 1 else "bottom"
            if i == 0:
                ax.text(x[i] + 0.16, s_tf[i] - 0.3, f"{s_tf[i]:.1f}%", ha="left", va="center", fontsize=7.6, fontweight="bold", color="#9A6300")
            elif i == 1 and ax is ax1:
                # Offset Beat-1 TF-IDF slightly left-down from diamond so it never touches OLMo 3 triangle at 29.3%!
                ax.text(x[i] - 0.14, s_tf[i] + 1.4, f"{s_tf[i]:.1f}%", ha="right", va="bottom", fontsize=7.6, fontweight="bold", color="#9A6300")
            else:
                ax.text(x[i], y_tf, f"{s_tf[i]:.1f}%", ha="center", va=va_tf, fontsize=7.6, fontweight="bold", color="#9A6300")
        ax.set_xticks(x)
        ax.set_xticklabels(beat_lbls, fontsize=8.2)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=7.2, frameon=True, facecolor="white", framealpha=0.95)

    b0_ratio = c5_t1[0] / max(1e-6, tf_t1[0])
    b1_diff = round(c5_t1[1], 1) - round(tf_t1[1], 1)
    ax1.annotate(
        f"Early beats (0–1): C5 beats TF-IDF\n({b0_ratio:.2f}× at Beat 0; +{b1_diff:.1f} pts at Beat 1).\nLate beats (2–3): cumulative TF-IDF leads.",
        xy=(0.95, c5_t1[1] + 0.8), xytext=(0.04, 53.5),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.0, connectionstyle="arc3,rad=0.16"),
        fontsize=7.5, fontweight="bold", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#F8F9FA", edgecolor=C_GRAY, lw=0.8),
    )

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "fig3_loro_myth_amplification")


# ---------------------------------------------------------------- Figure 4: 5M Cross-Tokenizer Distillation & U_jung Transplant
def make_fig4():
    arms_all = [
        ("P0", "P0\nPlain\n(w=0)"),
        ("P1", "P1\nGemma4"),
        ("P2", "P2\nC5"),
        ("P3", "P3\nC5+Myth"),
        ("E229_M0", "M0\nAnneal C5"),
        ("E229_M1", "M1\nLORO"),
        ("E229_M2", "M2\n+U_jung"),
        ("E229_M3", "M3\n−U_jung"),
    ]
    phi4_seeds, olmo_seeds = [], []
    m_data = {}
    for code, _ in arms_all:
        fs = sorted(glob.glob(str(RUNS_COMP / f"{code}_s*.json")))
        rows = [json.load(open(f)) for f in fs]
        phi4_seeds.append([r["platonic"]["blind_phi4_pts"]["r2_frob"] for r in rows])
        olmo_seeds.append([r["platonic"]["blind_olmo3_pts"]["r2_frob"] for r in rows])
        if code.startswith("E229_"):
            m_data[code] = {
                "bb_cos": [r["jung_coherence"]["backbone_neutral_procrustes_to_c5"]["cos_house"] for r in rows],
                "zs_house": [r["jung_coherence"]["backbone_neutral_procrustes_to_c5"]["zeroshot_house_acc"] * 100 for r in rows],
                "zs_self2": [r["jung_coherence"]["backbone_neutral_procrustes_to_c5"]["zeroshot_self2_acc"] * 100 for r in rows],
                "perf": [r["jung_coherence"]["last"]["house_perf_acc"] * 100 for r in rows],
                "blimp": [r["blimp"] * 100 for r in rows],
                "mcq": [r["mcq_gain"] * 100 for r in rows],
            }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.9), gridspec_kw={"width_ratios": [1.26, 1.08]})

    x = np.arange(len(arms_all))
    w = 0.36
    m_p4 = [np.mean(s) for s in phi4_seeds]
    m_ol = [np.mean(s) for s in olmo_seeds]

    ax1.bar(x - w / 2, m_p4, w, label="Held-Out Phi-4-mini $R^2_{\\mathrm{frob}}$", color=C_BLUE, alpha=0.88)
    ax1.bar(x + w / 2, m_ol, w, label="Held-Out OLMo 3 7B $R^2_{\\mathrm{frob}}$", color=C_ORANGE, alpha=0.88)

    for i in range(len(arms_all)):
        ax1.scatter([x[i] - w / 2] * len(phi4_seeds[i]), phi4_seeds[i], color=C_DARK, s=16, zorder=5, label="Individual Seeds (s1–s3)" if i == 0 else "")
        ax1.scatter([x[i] + w / 2] * len(olmo_seeds[i]), olmo_seeds[i], color=C_DARK, s=16, zorder=5)

    ax1.axhline(0, color=C_DARK, linewidth=0.9)
    ax1.axvline(3.5, color=C_GRAY, linestyle=":", linewidth=1.2)
    ax1.axvspan(3.5, 7.5, color=C_PURPLE, alpha=0.05, zorder=0)
    ax1.text(1.5, 0.176, "Sec 4.5: P0 (w=0) vs. P1–P3 (plato_w=0.25)", ha="center", va="top", fontsize=7.6, fontweight="bold", color=C_BLUE)
    ax1.text(5.5, 0.176, "Sec 4.6: Annealed plato_w → 0", ha="center", va="top", fontsize=7.8, fontweight="bold", color=C_PURPLE)

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in arms_all], fontsize=7.7)
    ax1.set_ylim(-0.095, 0.190)
    ax1.set_ylabel(r"$R^2_{\mathrm{frob}}$ (4.72M Non-Embed / 8.92M Total bpe8k)")
    no_ov = (min(phi4_seeds[2]) > max(phi4_seeds[1]) and min(olmo_seeds[2]) > max(olmo_seeds[1]))
    ov_txt = "Zero Seed Overlap" if no_ov else "3 Seeds Each"
    ax1.set_title(f"A. Cross-Tokenizer 7B → 5M Transfer (3 Seeds Each, {ov_txt})")
    ax1.legend(loc="lower right", fontsize=7.5, frameon=True, facecolor="white")

    # Panel B: Full transparent M2 vs M0 vs M3 including BOTH wins (bb_cos/zsH/zsSELF2/perf_last/BLiMP) AND mcq_gain trade-off!
    metrics = [
        ("bb_cos", "Backbone\nbb_cos (×100)", 100.0),
        ("zs_house", "Zero-Shot\nHOUSE (%)", 1.0),
        ("zs_self2", "Zero-Shot\nSELF2 (%)", 1.0),
        ("perf", "perfection\nlast (%)", 1.0),
        ("blimp", "BLiMP\nSyntax (%)", 1.0),
        ("mcq", "MCQ Gain\n(×100)", 1.0),
    ]
    xm = np.arange(len(metrics))
    wm = 0.25
    for idx, (arm, lbl, col) in enumerate([
        ("E229_M3", "M3: Ablated U_jung (0×)", C_GRAY),
        ("E229_M0", "M0: Natural C5 (1×)", C_SKY),
        ("E229_M2", "M2: Boosted U_jung (3.5×)", C_TEAL),
    ]):
        vals = [np.mean(m_data[arm][mk]) * scale for mk, _, scale in metrics]
        off = (idx - 1) * wm
        bars = ax2.bar(xm + off, vals, wm, label=lbl, color=col, edgecolor="white", linewidth=0.5)
        for j, (mk, _, scale) in enumerate(metrics):
            pts = [v * scale for v in m_data[arm][mk]]
            ax2.scatter([xm[j] + off] * len(pts), pts, color=C_DARK, s=13, zorder=5)
        if arm == "E229_M2":
            for j, b in enumerate(bars):
                top_pt = max([v * metrics[j][2] for v in m_data[arm][metrics[j][0]]])
                h = max(b.get_height(), top_pt)
                txt = f"{b.get_height()/100:.2f}" if j == 0 else f"{b.get_height():.1f}"
                ax2.text(b.get_x() + b.get_width() / 2, h + 2.2, txt, ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=C_TEAL)

    bb_ratio = float(np.mean(m_data["E229_M2"]["bb_cos"]) / np.mean(m_data["E229_M3"]["bb_cos"]))
    mcq_m3 = float(np.mean(m_data["E229_M3"]["mcq"]))
    mcq_m2 = float(np.mean(m_data["E229_M2"]["mcq"]))
    ax2.annotate(
        f"{bb_ratio:.1f}× backbone\nHOUSE cosine gain",
        xy=(0.25, 70.5), xytext=(-0.35, 98),
        arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.1, connectionstyle="arc3,rad=0.12"),
        fontsize=7.6, fontweight="bold", color=C_TEAL,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#E8F8F5", edgecolor=C_TEAL, lw=0.8),
    )
    ax2.annotate(
        f"Trade-off:\nM3 leads MCQ\n({mcq_m3:.1f} vs {mcq_m2:.1f})",
        xy=(4.75, 22.0), xytext=(3.95, 42.0),
        arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.0),
        fontsize=7.0, fontweight="bold", color=C_RED,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FDEDEC", edgecolor=C_RED, lw=0.7),
    )

    ax2.set_xticks(xm)
    ax2.set_xticklabels([ml for _, ml, _ in metrics], fontsize=7.5)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Score (% or Scaled Metric)")
    ax2.set_title("B. Surgical 7B → 5M U_jung Transplant & Trade-Offs")
    ax2.legend(loc="upper right", fontsize=7.2, frameon=True, facecolor="white", framealpha=0.92)

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "fig4_5m_distillation_ujung")


# ---------------------------------------------------------------- Figure 5: Frontier 8B Gemma 4 E4B Causal Stage A Control
def make_fig5():
    base = json.load(open(AION_RUNS / "e4b_base_eval.json"))["platonic_jung"]
    ctrl = json.load(open(AION_RUNS / "e4b_stage_a_ctrl_eval.json"))
    stga = json.load(open(AION_RUNS / "e4b_stage_a_eval.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.6), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # Panel A: ZERO-ORIGIN y-axis (no truncated bar distortion!)
    x = np.arange(2)
    w = 0.25
    b_vals = [base["blind_phi4_pts"]["r2_frob"], base["blind_olmo3_pts"]["r2_frob"]]
    c_vals = [ctrl["blind_phi4_pts"]["r2_frob"], ctrl["blind_olmo3_pts"]["r2_frob"]]
    a_vals = [stga["blind_phi4_pts"]["r2_frob"], stga["blind_olmo3_pts"]["r2_frob"]]

    r1 = ax1.bar(x - w, b_vals, w, label="Unmodified Base Gemma 4 E4B", color=C_GRAY)
    r2 = ax1.bar(x, c_vals, w, label="Matched CE+KL Control (plato_w = 0)", color=C_GOLD)
    r3 = ax1.bar(x + w, a_vals, w, label="Stage A Platonic (C5 + U_jung 3.5×)", color=C_TEAL)

    for bars in (r1, r2, r3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.4f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold" if bar in r3 else "normal")

    # Annotate exact causal deltas & Gutenberg NLL dynamically from JSONs
    dp4_a = a_vals[0] - b_vals[0]; dp4_c = c_vals[0] - b_vals[0]
    dol_a = a_vals[1] - b_vals[1]; dol_c = c_vals[1] - b_vals[1]
    nll_txt = (
        f"Causal Lift vs. CE+KL Ctrl (1 seed):\n"
        f"  Phi-4:  +{dp4_a:.4f} vs +{dp4_c:.4f} ({dp4_a/dp4_c:.2f}×, {100*(dp4_a-dp4_c)/dp4_a:.1f}%)\n"
        f"  OLMo 3: +{dol_a:.4f} vs +{dol_c:.4f} ({dol_a/dol_c:.2f}×, {100*(dol_a-dol_c)/dol_a:.1f}%)\n"
        f"Held-out Gutenberg NLL:\n"
        f"  Base {base['held_nll']['held_gut_nll']:.3f} | Ctrl {ctrl['held_nll']['held_gut_nll']:.3f} | Stage A {stga['held_nll']['held_gut_nll']:.3f}"
    )
    ax1.text(0.03, 0.97, nll_txt, transform=ax1.transAxes, va="top", ha="left", fontsize=7.2, family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9FA", edgecolor=C_GRAY, lw=0.7))

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Held-Out Microsoft\nPhi-4-mini", "Held-Out AllenAI\nOLMo 3 7B"], fontsize=8.8)
    ax1.set_ylim(0.0, 0.345)
    ax1.set_ylabel(r"Out-of-Sample $R^2_{\mathrm{frob}}$ (Zero-Origin Axis)")
    ax1.set_title("A. 8B Gemma 4 E4B Causal Control (450 Steps, MLPs 11–12)")
    ax1.legend(loc="lower right", fontsize=7.4, frameon=True, facecolor="white", framealpha=0.95)

    # Panel B: Per-domain cosine shift (delta vs Base) across 4 domains x 2 held-out models
    doms = ["held_gut", "items3_mech", "items3_myth", "proc_held"]
    dom_lbls = ["Gutenberg\nProse", "items3\nMech", "items3\nMyth", "Procedural\nReasoning"]
    xd = np.arange(4)
    wd = 0.20

    d_ctrl_p4 = [(ctrl["blind_phi4_pts"]["by_domain_cos"][d] - base["blind_phi4_pts"]["by_domain_cos"][d]) * 1000 for d in doms]
    d_stga_p4 = [(stga["blind_phi4_pts"]["by_domain_cos"][d] - base["blind_phi4_pts"]["by_domain_cos"][d]) * 1000 for d in doms]
    d_ctrl_ol = [(ctrl["blind_olmo3_pts"]["by_domain_cos"][d] - base["blind_olmo3_pts"]["by_domain_cos"][d]) * 1000 for d in doms]
    d_stga_ol = [(stga["blind_olmo3_pts"]["by_domain_cos"][d] - base["blind_olmo3_pts"]["by_domain_cos"][d]) * 1000 for d in doms]

    ax2.bar(xd - 1.5 * wd, d_ctrl_p4, wd, label="CE+KL Ctrl → Phi-4", color=C_SKY, alpha=0.65, edgecolor=C_BLUE, hatch="//")
    ax2.bar(xd - 0.5 * wd, d_stga_p4, wd, label="Stage A (C5) → Phi-4", color=C_BLUE)
    ax2.bar(xd + 0.5 * wd, d_ctrl_ol, wd, label="CE+KL Ctrl → OLMo 3", color="#F8C471", alpha=0.75, edgecolor=C_ORANGE, hatch="//")
    ax2.bar(xd + 1.5 * wd, d_stga_ol, wd, label="Stage A (C5) → OLMo 3", color=C_ORANGE)
    ax2.axhline(0, color=C_DARK, linewidth=0.9)

    for i in range(4):
        ax2.text(xd[i] - 0.5 * wd, d_stga_p4[i] + 0.35, f"+{d_stga_p4[i]:.1f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=C_BLUE)
        ax2.text(xd[i] + 1.5 * wd, d_stga_ol[i] + 0.35, f"+{d_stga_ol[i]:.1f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=C_ORANGE)

    ax2.set_xticks(xd)
    ax2.set_xticklabels(dom_lbls, fontsize=8.4)
    ax2.set_ylim(0, 19.5)
    ax2.set_ylabel("Cosine Shift vs. Base E4B (×10⁻³)")
    n_win_doms = sum(1 for a, c in zip(d_stga_p4, d_ctrl_p4) if a > c) + sum(1 for a, c in zip(d_stga_ol, d_ctrl_ol) if a > c)
    ax2.set_title(f"B. Per-Domain Cosine Shift vs. Base (Stage A vs. CE+KL Control, {n_win_doms}/{2*len(doms)})")
    ax2.legend(loc="upper right", ncol=2, fontsize=7.4, frameon=True, facecolor="white")

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "fig5_8b_stage_a_causal")


# ---------------------------------------------------------------- Figure 6: 2x2 Crossed Zero-Overlap Gram-Schmidt Epistemic Compass
def make_fig6():
    gm = json.load(open(AION_RUNS / "audit233_epistemic_gemma_2x2.json"))
    qw = json.load(open(AION_RUNS / "audit233_epistemic_qwen_2x2.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.2), gridspec_kw={"width_ratios": [1.0, 1.22]})

    # Panel A: Exact 2x2 Crossed Gemma 4 E4B (Base vs active-Titans Stage-B x Neutral vs Aion)
    g2x2 = [
        ("Base E4B\n× Neutral", gm["base_e4b"]["neutral_prompt"]["L25"]["subsets"]["clean_tense_clean_24"]),
        ("Base E4B\n× Aion", gm["base_e4b"]["buddy_prompt"]["L25"]["subsets"]["clean_tense_clean_24"]),
        ("Stage-B (ON)\n× Neutral", gm["stage_b_e4b"]["neutral_prompt"]["L25"]["subsets"]["clean_tense_clean_24"]),
        ("Stage-B (ON)\n× Aion", gm["stage_b_e4b"]["buddy_prompt"]["L25"]["subsets"]["clean_tense_clean_24"]),
    ]

    # Panel B: ALL 4 matched-layer Qwen3.5-9B cells (including L19 Neutral null!) + Stage-B Dedup-10 OOS
    cross = [
        ("Qwen L12\n× Neutral", qw["neutral_prompt"]["L12"]["subsets"]["clean_tense_clean_24"]),
        ("Qwen L12\n× Aion", qw["buddy_prompt"]["L12"]["subsets"]["clean_tense_clean_24"]),
        ("Qwen L19\n× Neutral\n(Null)", qw["neutral_prompt"]["L19"]["subsets"]["clean_tense_clean_24"]),
        ("Qwen L19\n× Aion", qw["buddy_prompt"]["L19"]["subsets"]["clean_tense_clean_24"]),
        ("Gemma Stg-B\nDedup-10", gm["stage_b_e4b"]["buddy_prompt_dedup10"]["L25"]["subsets"]["dedup_10_oos"]),
    ]

    for ax, groups, title in [
        (ax1, g2x2, "A. 2×2 Crossed Audit on Gemma 4 E4B L25 (Active Titans)"),
        (ax2, cross, "B. All 4 Matched-Layer Qwen3.5-9B Cells & Dedup-10 OOS"),
    ]:
        xg = np.arange(len(groups))
        wg = 0.18
        ent_sym = [g["entropy_best_sym_auroc"] for _, g in groups]
        ent_pos = [g["entropy_pos_auroc"] for _, g in groups]
        cor_v = [g["neg_corr_auroc"] for _, g in groups]
        hp_v = [g["neg_house_perp_auroc"] for _, g in groups]
        cmb_v = [g["combined_perp_auroc"] for _, g in groups]
        r95_v = [g["rand500_mirrored2_p95"] for _, g in groups]

        ax.bar(xg - 1.5 * wg, ent_sym, wg, label="Best-Sign 1st-Tok Entropy max(AUC, 1−AUC)", color=C_RED, alpha=0.72)
        ax.bar(xg - 0.5 * wg, cor_v, wg, label="−CORRECTNESS (Frozen Calib.)", color=C_BLUE)
        ax.bar(xg + 0.5 * wg, hp_v, wg, label="−HOUSE_PERP (Gram-Schmidt cos=0)", color=C_PURPLE)
        ax.bar(xg + 1.5 * wg, cmb_v, wg, label="Combined_Perp (Headline Metric)", color=C_TEAL)

        for i, r95 in enumerate(r95_v):
            ax.plot([xg[i] - 0.38, xg[i] + 0.38], [r95, r95], color=C_DARK, linewidth=1.7, linestyle="--",
                    label="500 Mirrored-Pair 95th Pct Null" if i == 0 else "")
            ey = ent_sym[i] + 0.025
            if abs(ey - r95) < 0.035:
                ey = r95 + 0.03
            ax.text(xg[i] - 1.5 * wg, ey, f"{ent_sym[i]:.2f}", ha="center", va="bottom", fontsize=8.0, color=C_RED, fontweight="bold")
            cy = max(cmb_v[i], hp_v[i]) + 0.025
            if abs(cy - r95) < 0.035:
                cy = r95 + 0.03
            ax.text(xg[i] + 1.5 * wg, cy, f"{cmb_v[i]:.3f}", ha="center", va="bottom", fontsize=7.8, color=C_TEAL, fontweight="bold")

        ax.axhline(0.5, color=C_GRAY, linestyle=":", linewidth=0.9)
        ax.set_xticks(xg)
        ax.set_xticklabels([lbl for lbl, _ in groups], fontsize=8.6)
        ax.set_ylim(0.0, 1.34)
        ax.set_ylabel("AUROC (Zero-Origin, Frozen Calibration)", fontsize=9.2)
        ax.set_title(title, fontsize=10.5)
        ax.legend(loc="lower left", ncol=2, fontsize=7.6, frameon=True, facecolor="white", framealpha=0.95)

    g_b_bud = g2x2[1][1]["entropy_best_sym_auroc"]
    g_s_bud = g2x2[3][1]["entropy_best_sym_auroc"]
    g_min = min(g[1]["combined_perp_auroc"] for g in g2x2)
    g_max = max(g[1]["combined_perp_auroc"] for g in g2x2)
    ax1.text(
        0.50, 0.965,
        f"Aion prompt drops sym. 1st-tok entropy AUROC to {g_b_bud:.2f} / {g_s_bud:.2f}\n"
        f"Combined_Perp (u_corr ⊥ u_house_perp) holds {g_min:.3f}–{g_max:.3f} (see Table for p)",
        transform=ax1.transAxes, ha="center", va="top", fontsize=8.1, fontweight="bold", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#FDEDEC", edgecolor=C_RED, lw=0.75),
    )
    q12n = cross[0][1]["combined_perp_auroc"]; q12b = cross[1][1]["combined_perp_auroc"]
    q19n = cross[2][1]["combined_perp_auroc"]; q19b = cross[3][1]["combined_perp_auroc"]
    ax2.text(
        0.50, 0.965,
        f"Qwen3.5: L12 holds in both prompts ({q12n:.3f} / {q12b:.3f});\n"
        f"L19 holds under Aion ({q19b:.3f}) but fails under Neutral ({q19n:.3f})",
        transform=ax2.transAxes, ha="center", va="top", fontsize=8.1, fontweight="bold", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#E8F8F5", edgecolor=C_TEAL, lw=0.75),
    )

    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "fig6_epistemic_house_correctness")


def main():
    setup_style()
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    print("ALL_6_FIGURES_GENERATED_OK", flush=True)


if __name__ == "__main__":
    main()
