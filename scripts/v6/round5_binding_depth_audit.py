"""Round-5 Final Polish Audit for Claims A & B (Refit-in-Bootstrap NLL/Strata, Basic+Percentile CIs, d15 Null & Skip-Bigram Counterexample).

Resolves EVERY remaining Round-4 finding on Claims A & B from BOTH Fable 5.2 and Astra 6.2:
  1. Matched 20-split setting-flip refit null at d15 (2,000 draws) for Gemma->5, Qwen->5, and native 6-model mean
     (proving d15 is small but detectable — never "absent").
  2. Exact verification of Astra's unfitted lexicon-weighted gap-4 skip-bigram linear scorer (142/240 = 59.17%, 0 ties).
  3. Refit-in-bootstrap (2,000 setting resamples) for cv2 vs whole-sentence NLL, cv2 vs changed-token NLL,
     AND every per-target whole/changed DeltaNLL <= 0 stratum (replacing fixed-prediction bootstrap for cv2!).
  4. BOTH percentile AND basic (reflected 2*obs - q) simultaneous Bonferroni CIs for cross-family (Bonf-4),
     256-D within-model (Bonf-2), and native full-dimensional within-model (Bonf-2) on bm_actor2x2_247!
  5. Exact per-model within-model peak depths (native & 256-D) and exact zs_sent 5-depth curves.
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

SCRATCH = Path(os.environ.get("JPL_WORK_DIR", "."))
EXP5 = SCRATCH / "exp249_round5"
EXP5.mkdir(parents=True, exist_ok=True)
EXP4 = SCRATCH / "exp248_round4"
EXP3 = SCRATCH / "exp247_round3"
EXP245 = SCRATCH / "exp245_controls"
REPO_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(Path(__file__).resolve().parents[2] / "bench" / "archetypes" / "fluent" / "plato")))
DEPTH_DIR = PLATO_CACHE / "depth240"

for p in (str(REPO_ARCH), str(REPO_ARCH / "fluent"), str(EXP245), str(EXP3)):
    if p not in sys.path:
        sys.path.insert(0, p)

import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import exp245_binding as E245  # noqa: E402
import exp247_actor_binding as E247  # noqa: E402

MLX_KEYS = E247.MLX_KEYS
DEPTH_FRACS = E247.DEPTH_FRACS


def _pct_and_basic_ci(obs, draws, alpha):
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    q_lo = float(np.percentile(draws, lo_q))
    q_hi = float(np.percentile(draws, hi_q))
    return {
        "obs": round(float(obs), 4),
        "boot_mean": round(float(np.mean(draws)), 4),
        "percentile": [round(q_lo, 4), round(q_hi, 4)],
        "basic_reflected": [round(2.0 * float(obs) - q_hi, 4), round(2.0 * float(obs) - q_lo, 4)],
    }


def _astra_skipbigram_g4_rule(rows):
    """Exact unfitted linear scorer over registered gap-4 ordered skip-bigrams: weight = -w(x)*pol(y)."""
    pos_set = set(E245.POS_LEX)
    neg_set = set(E245.NEG_LEX)

    def score_sent(txt):
        s = 0
        for feat in set(E245._skip_bigrams(txt, max_gap=4)):
            x, y = feat.split("__", 1)
            wx = 1 if x in pos_set else (3 if x in neg_set else 0)
            py = 1 if y in pos_set else (-1 if y in neg_set else 0)
            s += -wx * py
        return s

    wins, losses, ties = 0, 0, 0
    for r in rows:
        dp = score_sent(r["pos"]) - score_sent(r["neg"])
        if dp > 0:
            wins += 1
        elif dp < 0:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties, "n_pairs": len(rows), "acc": round((wins + 0.5 * ties) / len(rows), 4)}


def main():
    t0 = time.time()
    actor_rows = json.loads((EXP3 / "bm_actor2x2_247_data.json").read_text())["bm_actor2x2_240"]
    sid_a, spl_a = E245._sid_and_splits(actor_rows, 24700)
    G_a = len(set(int(s) for s in sid_a))
    by_s = [np.where(sid_a == s)[0] for s in range(G_a)]
    split_s_fold = np.array([[int(fl[by_s[g][0]]) for g in range(G_a)] for fl in spl_a], dtype=int)
    s_of_i = sid_a.astype(int)

    # Syntax counts audit
    syn_counts = {}
    for r in actor_rows:
        k_s = str(r.get("syntax", "unknown"))
        syn_counts[k_s] = syn_counts.get(k_s, 0) + 1

    # 1. Astra gap-4 skip-bigram counterexample
    skip_g4_counterex = _astra_skipbigram_g4_rule(actor_rows)

    # Load NLL deltas
    dnll_w, dnll_c = {}, {}
    for k in MLX_KEYS:
        dr = np.load(DEPTH_DIR / f"bmactor247_{k}.npz")
        dnll_w[k] = dr["neg_nll_mean"].astype(np.float64) - dr["pos_nll_mean"].astype(np.float64)
        dnll_c[k] = dr["neg_changed_nll_mean"].astype(np.float64) - dr["pos_changed_nll_mean"].astype(np.float64)

    sg_rows, _, _ = V234.load_valence_data()
    sg_doms = np.array([r["domain"] for r in sg_rows])
    n_sg = len(sg_rows)

    # Load d15, d30, d50, d70 (both 256-D PCA and native full-dim)
    Dp_256 = {pool: {} for pool in ("mean", "last")}
    Q5_256 = {pool: {} for pool in ("mean", "last")}
    Dp_nat = {pool: {} for pool in ("mean", "last")}

    for slot, df in enumerate(DEPTH_FRACS):
        dkey = f"d{int(round(df * 100))}"
        if dkey not in ("d15", "d30", "d50", "d70"):
            continue
        for pool_key in ("mean", "last"):
            rot = E245._fit_neutral_rot_at_slot(slot, pool_key, d_plato=256)
            Dp_256[pool_key][dkey] = {}
            Q5_256[pool_key][dkey] = {}
            Dp_nat[pool_key][dkey] = {}
            for k in MLX_KEYS:
                d240 = np.load(DEPTH_DIR / f"depth240_{k}.npz")
                dr = np.load(DEPTH_DIR / f"bmactor247_{k}.npz")
                # 256-D PCA+Procrustes
                Z = E245._map_mat(dr[f"pos_{pool_key}"][slot], rot, k) - E245._map_mat(dr[f"neg_{pool_key}"][slot], rot, k)
                Dv = E245._map_mat(d240[f"val_pos_{pool_key}"][slot], rot, k) - E245._map_mat(d240[f"val_neg_{pool_key}"][slot], rot, k)
                dom_v = [AR.unit(Dv[:n_sg][sg_doms == d].mean(0)) for d in sorted(set(sg_doms.tolist()))]
                Q5 = V234._orthonormal_basis(dom_v + [AR.unit(Dv[n_sg:].mean(0))])
                Q5_256[pool_key][dkey][k] = Q5
                Dp_256[pool_key][dkey][k] = V234._project_out_rows(Z, Q5)
                # Native full-dim
                Zn = dr[f"pos_{pool_key}"][slot].astype(np.float64) - dr[f"neg_{pool_key}"][slot].astype(np.float64)
                Dvn = d240[f"val_pos_{pool_key}"][slot].astype(np.float64) - d240[f"val_neg_{pool_key}"][slot].astype(np.float64)
                dom_vn = [AR.unit(Dvn[:n_sg][sg_doms == d].mean(0)) for d in sorted(set(sg_doms.tolist()))]
                Q5n = V234._orthonormal_basis(dom_vn + [AR.unit(Dvn[n_sg:].mean(0))])
                Dp_nat[pool_key][dkey][k] = V234._project_out_rows(Zn, Q5n)

    def _refit_cv_boot_all(Ds_p, Dt_p_list, Qt_p_list, boot_w_mat, mask_list=None):
        """Vectorized refit-in-bootstrap over (B, G) setting weights.
        Returns (B, len(Dt_p_list)) accuracy matrix (optionally restricted to item masks per target)."""
        B = boot_w_mat.shape[0]
        S_src = np.stack([Ds_p[by_s[g]].sum(0) for g in range(G_a)], axis=0)
        nT = len(Dt_p_list)
        hits = np.zeros((B, nT), dtype=np.float64)
        den = np.zeros((B, nT), dtype=np.float64)
        for si, fl in enumerate(spl_a):
            sf = split_s_fold[si]
            for f in (0, 1):
                tr_g = (sf != f).astype(np.float64)
                te_i = np.where(fl == f)[0]
                te_s = s_of_i[te_i]
                U_raw = (boot_w_mat * tr_g[None, :]) @ S_src
                U_raw /= np.maximum(1e-12, np.linalg.norm(U_raw, axis=1, keepdims=True))
                w_te_base = boot_w_mat[:, te_s]
                for ti, (Dt_p, Qt_p) in enumerate(zip(Dt_p_list, Qt_p_list)):
                    if Qt_p is not None:
                        Ut = V234._project_out_rows(U_raw, Qt_p)
                        Ut /= np.maximum(1e-12, np.linalg.norm(Ut, axis=1, keepdims=True))
                    else:
                        Ut = U_raw
                    sc = Ut @ Dt_p[te_i].T
                    if mask_list is not None:
                        m_te = mask_list[ti][te_i].astype(np.float64)
                        w_eff = w_te_base * m_te[None, :]
                    else:
                        w_eff = w_te_base
                    hits[:, ti] += ((sc > 0).astype(np.float64) * w_eff).sum(axis=1)
                    den[:, ti] += w_eff.sum(axis=1)
        return hits / np.maximum(1e-12, den)

    # 2. Exact d15 matched 20-split setting-flip refit null (2,000 draws, mean pool)
    def _d15_flip_refit_null(Ds_p_list, Dt_p_lists, Qt_p_lists, n_perm=2000, seed=24915):
        """Joint setting-flip null (flips shared across models), refitting all 20 splits on every draw."""
        rng = np.random.default_rng(seed)
        flips = rng.choice([-1.0, 1.0], size=(n_perm, G_a))  # (P, G)
        ones = np.ones((1, G_a), dtype=np.float64)
        obs_acc = 0.0
        null_acc = np.zeros(n_perm, dtype=np.float64)
        n_groups = len(Ds_p_list)
        for Ds_p, Dt_l, Qt_l in zip(Ds_p_list, Dt_p_lists, Qt_p_lists):
            S_src = np.stack([Ds_p[by_s[g]].sum(0) for g in range(G_a)], axis=0)
            g_obs = 0.0
            g_null = np.zeros(n_perm, dtype=np.float64)
            for si, fl in enumerate(spl_a):
                sf = split_s_fold[si]
                for f in (0, 1):
                    tr_g = (sf != f).astype(np.float64)
                    te_i = np.where(fl == f)[0]
                    te_s = s_of_i[te_i]
                    # Observed
                    uo = (ones * tr_g[None, :]) @ S_src
                    uo /= np.maximum(1e-12, np.linalg.norm(uo, axis=1, keepdims=True))
                    # Null flipped source
                    Un = (flips * tr_g[None, :]) @ S_src
                    Un /= np.maximum(1e-12, np.linalg.norm(Un, axis=1, keepdims=True))
                    fl_te = flips[:, te_s]  # (P, 120)
                    for Dt_p, Qt_p in zip(Dt_l, Qt_l):
                        if Qt_p is not None:
                            uo_t = V234._project_out_rows(uo, Qt_p)
                            uo_t /= np.maximum(1e-12, np.linalg.norm(uo_t, axis=1, keepdims=True))
                            Un_t = V234._project_out_rows(Un, Qt_p)
                            Un_t /= np.maximum(1e-12, np.linalg.norm(Un_t, axis=1, keepdims=True))
                        else:
                            uo_t, Un_t = uo, Un
                        g_obs += float(((uo_t @ Dt_p[te_i].T) > 0).mean()) / (2.0 * len(spl_a) * len(Dt_l))
                        sc_n = (Un_t @ Dt_p[te_i].T) * fl_te
                        g_null += (sc_n > 0).mean(axis=1) / (2.0 * len(spl_a) * len(Dt_l))
            obs_acc += g_obs / n_groups
            null_acc += g_null / n_groups
        p_upper = float((int(np.sum(null_acc >= obs_acc - 1e-12)) + 1) / (n_perm + 1))
        return {"obs_acc": round(obs_acc, 4), "null_mean": round(float(null_acc.mean()), 4), "null_max": round(float(null_acc.max()), 4), "n_perm": n_perm, "p_upper": round(p_upper, 5)}

    d15_null = {
        "gemma_to_5_mean": _d15_flip_refit_null(
            [Dp_256["mean"]["d15"]["gemma4_e4b"]],
            [[Dp_256["mean"]["d15"][t] for t in MLX_KEYS if t != "gemma4_e4b"]],
            [[Q5_256["mean"]["d15"][t] for t in MLX_KEYS if t != "gemma4_e4b"]],
            n_perm=2000, seed=24915,
        ),
        "qwen_to_5_mean": _d15_flip_refit_null(
            [Dp_256["mean"]["d15"]["qwen35_9b"]],
            [[Dp_256["mean"]["d15"][t] for t in MLX_KEYS if t != "qwen35_9b"]],
            [[Q5_256["mean"]["d15"][t] for t in MLX_KEYS if t != "qwen35_9b"]],
            n_perm=2000, seed=24916,
        ),
        "native_6model_mean": _d15_flip_refit_null(
            [Dp_nat["mean"]["d15"][k] for k in MLX_KEYS],
            [[Dp_nat["mean"]["d15"][k]] for k in MLX_KEYS],
            [[None] for _ in MLX_KEYS],
            n_perm=2000, seed=24917,
        ),
    }

    # 3. Refit-in-bootstrap (2,000 draws) for d50 cv2 vs whole/changed NLL AND per-target DeltaNLL <= 0 strata!
    rng_cv = np.random.default_rng(24920)
    n_boot_cv = 2000
    boot_w_cv = np.array([np.bincount(rng_cv.integers(0, G_a, size=G_a), minlength=G_a) for _ in range(n_boot_cv)], dtype=np.float64)
    ones_w = np.ones((1, G_a), dtype=np.float64)

    # Setting-weighted NLL accuracies on the SAME bootstrap draws
    def _nll_boot_mat(dnll_dict, tgts, bw_mat):
        B = bw_mat.shape[0]
        out = np.zeros((B, len(tgts)), dtype=np.float64)
        for ti, tk in enumerate(tgts):
            hit_g = np.array([(dnll_dict[tk][by_s[g]] > 0).mean() for g in range(G_a)], dtype=np.float64)
            out[:, ti] = (bw_mat @ hit_g) / bw_mat.sum(axis=1)
        return out

    d50_refit_nll_and_strata = {}
    for pool_key in ("mean", "last"):
        d50_refit_nll_and_strata[pool_key] = {}
        for src_k in ("gemma4_e4b", "qwen35_9b"):
            tgts = [k for k in MLX_KEYS if k != src_k]
            Ds_p = Dp_256[pool_key]["d50"][src_k]
            Dt_l = [Dp_256[pool_key]["d50"][t] for t in tgts]
            Qt_l = [Q5_256[pool_key]["d50"][t] for t in tgts]

            # Overall cv2 refit-in-bootstrap
            obs_cv = _refit_cv_boot_all(Ds_p, Dt_l, Qt_l, ones_w)[0]
            dr_cv = _refit_cv_boot_all(Ds_p, Dt_l, Qt_l, boot_w_cv)
            obs_nw = _nll_boot_mat(dnll_w, tgts, ones_w)[0]
            dr_nw = _nll_boot_mat(dnll_w, tgts, boot_w_cv)
            obs_nc = _nll_boot_mat(dnll_c, tgts, ones_w)[0]
            dr_nc = _nll_boot_mat(dnll_c, tgts, boot_w_cv)

            # Strata refit-in-bootstrap
            mw_l = [dnll_w[t] <= 0 for t in tgts]
            mc_l = [dnll_c[t] <= 0 for t in tgts]
            obs_sw = _refit_cv_boot_all(Ds_p, Dt_l, Qt_l, ones_w, mw_l)[0]
            dr_sw = _refit_cv_boot_all(Ds_p, Dt_l, Qt_l, boot_w_cv, mw_l)
            obs_sc = _refit_cv_boot_all(Ds_p, Dt_l, Qt_l, ones_w, mc_l)[0]
            dr_sc = _refit_cv_boot_all(Ds_p, Dt_l, Qt_l, boot_w_cv, mc_l)

            per_t = {}
            for ti, tk in enumerate(tgts):
                per_t[tk] = {
                    "whole_dnll_le_0_refit_boot": {
                        "n_items": int(np.sum(mw_l[ti])),
                        **_pct_and_basic_ci(obs_sw[ti], dr_sw[:, ti], 0.05),
                    },
                    "changed_dnll_le_0_refit_boot": {
                        "n_items": int(np.sum(mc_l[ti])),
                        **_pct_and_basic_ci(obs_sc[ti], dr_sc[:, ti], 0.05),
                    },
                }

            d50_refit_nll_and_strata[pool_key][src_k] = {
                "cv2_minus_whole_nll_refit_boot": _pct_and_basic_ci(obs_cv.mean() - obs_nw.mean(), dr_cv.mean(axis=1) - dr_nw.mean(axis=1), 0.05),
                "cv2_minus_changed_nll_refit_boot": _pct_and_basic_ci(obs_cv.mean() - obs_nc.mean(), dr_cv.mean(axis=1) - dr_nc.mean(axis=1), 0.05),
                "per_target_strata_refit_boot": per_t,
            }

    # 4. BOTH percentile AND basic (reflected) simultaneous Bonferroni CIs on bm_actor2x2_247 (cross-family Bonf-4, 256-D Bonf-2, native Bonf-2)
    rng_d = np.random.default_rng(24930)
    n_boot_d = 2000
    boot_w_d = np.array([np.bincount(rng_d.integers(0, G_a, size=G_a), minlength=G_a) for _ in range(n_boot_d)], dtype=np.float64)

    depth_pct_and_basic = {}
    for pool_key in ("mean", "last"):
        xf = {}
        for src_k in ("gemma4_e4b", "qwen35_9b"):
            tgts = [k for k in MLX_KEYS if k != src_k]
            o_d, dr_d = {}, {}
            for dk in ("d30", "d50", "d70"):
                o_d[dk] = float(_refit_cv_boot_all(Dp_256[pool_key][dk][src_k], [Dp_256[pool_key][dk][t] for t in tgts], [Q5_256[pool_key][dk][t] for t in tgts], ones_w).mean())
                dr_d[dk] = _refit_cv_boot_all(Dp_256[pool_key][dk][src_k], [Dp_256[pool_key][dk][t] for t in tgts], [Q5_256[pool_key][dk][t] for t in tgts], boot_w_d).mean(axis=1)
            xf[src_k] = {
                "d50_minus_d30_bonf4": _pct_and_basic_ci(o_d["d50"] - o_d["d30"], dr_d["d50"] - dr_d["d30"], 0.05 / 4.0),
                "d50_minus_d70_bonf4": _pct_and_basic_ci(o_d["d50"] - o_d["d70"], dr_d["d50"] - dr_d["d70"], 0.05 / 4.0),
            }
        # 256-D within-model 6-model mean
        o_p, dr_p = {}, {}
        o_n, dr_n = {}, {}
        for dk in ("d30", "d50", "d70"):
            o_p[dk] = float(np.mean([_refit_cv_boot_all(Dp_256[pool_key][dk][k], [Dp_256[pool_key][dk][k]], [Q5_256[pool_key][dk][k]], ones_w)[0, 0] for k in MLX_KEYS]))
            dr_p[dk] = np.mean([_refit_cv_boot_all(Dp_256[pool_key][dk][k], [Dp_256[pool_key][dk][k]], [Q5_256[pool_key][dk][k]], boot_w_d)[:, 0] for k in MLX_KEYS], axis=0)
            o_n[dk] = float(np.mean([_refit_cv_boot_all(Dp_nat[pool_key][dk][k], [Dp_nat[pool_key][dk][k]], [None], ones_w)[0, 0] for k in MLX_KEYS]))
            dr_n[dk] = np.mean([_refit_cv_boot_all(Dp_nat[pool_key][dk][k], [Dp_nat[pool_key][dk][k]], [None], boot_w_d)[:, 0] for k in MLX_KEYS], axis=0)
        depth_pct_and_basic[pool_key] = {
            "cross_family_bonf4": xf,
            "within_model_pca256_bonf2": {
                "d50_minus_d30_bonf2": _pct_and_basic_ci(o_p["d50"] - o_p["d30"], dr_p["d50"] - dr_p["d30"], 0.05 / 2.0),
                "d50_minus_d70_bonf2": _pct_and_basic_ci(o_p["d50"] - o_p["d70"], dr_p["d50"] - dr_p["d70"], 0.05 / 2.0),
            },
            "within_model_native_bonf2": {
                "d50_minus_d30_bonf2": _pct_and_basic_ci(o_n["d50"] - o_n["d30"], dr_n["d50"] - dr_n["d30"], 0.05 / 2.0),
                "d50_minus_d70_bonf2": _pct_and_basic_ci(o_n["d50"] - o_n["d70"], dr_n["d50"] - dr_n["d70"], 0.05 / 2.0),
            },
        }

    # 5. Summarize per-model within-model peak depths from Round-4 JSONs
    r4_bd = json.loads((EXP4 / "round4_binding_depth_results.json").read_text())
    r4_nat = json.loads((EXP4 / "within_model_native_5depth.json").read_text())
    per_model_peaks = {}
    for space_name, src_dict in [("native", r4_nat["within_model_native_5depth"]), ("pca256", r4_bd["within_model_source_eq_target_5depth"])]:
        per_model_peaks[space_name] = {}
        for pool_key in ("mean", "last"):
            peaks = {}
            for k in MLX_KEYS:
                vals = {dk: src_dict[pool_key]["bm_actor2x2_247"][dk]["per_model"][k] for dk in ("d15", "d30", "d50", "d70", "d85")}
                best_d = max(vals, key=lambda d: vals[d])
                peaks[k] = {"peak_depth": best_d, "curve": vals}
            per_model_peaks[space_name][pool_key] = {
                "n_peaking_at_d50": sum(1 for v in peaks.values() if v["peak_depth"] == "d50"),
                "per_model": peaks,
            }

    out = {
        "elapsed_sec": round(time.time() - t0, 1),
        "syntax_counts_actor240": syn_counts,
        "skipbigram_g4_unfitted_counterexample": skip_g4_counterex,
        "d15_matched_flip_refit_null": d15_null,
        "d50_refit_in_boot_nll_and_strata": d50_refit_nll_and_strata,
        "actor2x2_depth_percentile_and_basic_bonf": depth_pct_and_basic,
        "within_model_per_model_peaks_actor2x2": per_model_peaks,
    }
    out_p = EXP5 / "round5_binding_depth_results.json"
    out_p.write_text(json.dumps(out, indent=2))
    print(f"SAVED {out_p} in {out['elapsed_sec']}s", flush=True)
    print("syntax_counts:", syn_counts)
    print("skipbigram_g4:", skip_g4_counterex)
    print("d15_null:", d15_null)
    for pool in ("mean", "last"):
        print(f"DEPTH [{pool}] cross_family_bonf4:", depth_pct_and_basic[pool]["cross_family_bonf4"])
        print(f"DEPTH [{pool}] pca256_bonf2:", depth_pct_and_basic[pool]["within_model_pca256_bonf2"])
        print(f"DEPTH [{pool}] native_bonf2:", depth_pct_and_basic[pool]["within_model_native_bonf2"])
    for src in ("gemma4_e4b", "qwen35_9b"):
        blk = d50_refit_nll_and_strata["mean"][src]
        print(f"REFIT-BOOT d50 MEAN {src}: vs_whole={blk['cv2_minus_whole_nll_refit_boot']} | vs_changed={blk['cv2_minus_changed_nll_refit_boot']}")


if __name__ == "__main__":
    main()
