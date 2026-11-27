"""Round-4 Final Polish Audit for Claims A & B (Pure CPU on Verified Cached Activations).

Resolves EVERY remaining Round-3 finding on Claims A & B from BOTH Fable 5.2 and Astra 6.2:
  1. Fixes rounding-before-comparison bug: compares EXACT integer counts and full-precision floats
     in all setting-flip and random-direction nulls (rounding ONLY at serialization!).
  2. Uses consistent 20-split-averaged accuracy everywhere (benign, adverse, whole-NLL <= 0,
     changed-token-NLL <= 0) WITH setting-clustered 95% bootstrap CIs on every NLL stratum!
  3. Computes exact 3-term first-clause lexical parity rule (frozen POS_LEX -> 238/240; +incorruptible -> 240/240)
     AND conditional actor-signed sentiment control (`±zs_sent`) across all 5 depths!
  4. Computes paired setting-clustered bootstrap CIs comparing `cv2`, `xdom_bmneut`, and `xdom_bm120`
     directly against BOTH whole-sentence NLL AND disjoint changed-tokens-only NLL (plus OLS regression
     of probe margin on changed-token DeltaNLL with setting-clustered CI on intercept & partial R^2)!
  5. Computes WITHIN-MODEL (`source == target`) 5-depth curves (`d15..d85`, BOTH `mean` and `last`,
     `cv2 perp VAL5`) across all 6 MLX labs for `bm_actor2x2_247`, `bm120`, `bm_rel242`, `bm_neut245`,
     `bm_polconf245`, and setting-disjoint `HOUSE` — separating representation depth from Procrustes alignment!
  6. Computes simultaneous 4-contrast Bonferroni CIs across BOTH sources for `bm_actor2x2_247` AND saves
     BOTH the 13 retained myth IDs AND all 69 retained mechanical IDs for `items3`!
"""
import json
import os
import re
import sys
import time
from pathlib import Path
import numpy as np

SCRATCH = Path(os.environ.get("JPL_WORK_DIR", "."))
EXP4 = SCRATCH / "exp248_round4"
EXP4.mkdir(parents=True, exist_ok=True)
EXP3 = SCRATCH / "exp247_round3"
EXP245 = SCRATCH / "exp245_controls"
EXP242 = SCRATCH / "exp240_242"
REPO_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(Path(__file__).resolve().parents[2] / "bench" / "archetypes" / "fluent" / "plato")))
DEPTH_DIR = PLATO_CACHE / "depth240"

for p in (str(REPO_ARCH), str(REPO_ARCH / "fluent"), str(EXP245), str(EXP3)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import bow_mirror238 as BM238  # noqa: E402
import exp245_binding as E245  # noqa: E402
import exp247_actor_binding as E247  # noqa: E402
import depth_audit247 as DA247  # noqa: E402

MLX_KEYS = E247.MLX_KEYS
DEPTH_FRACS = E247.DEPTH_FRACS


def _holm(p_map):
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for r, (k, pv) in enumerate(items):
        run = max(run, min(1.0, (m - r) * pv))
        out[k] = round(run, 5)
    return out


def _unrounded_setting_perm(scores, sid_idx, n_perm=2000, seed=2480):
    """Exact INTEGER-COUNT uncentered setting-flip Monte Carlo null (fixes rounding bug!)."""
    uniq = sorted(set(int(s) for s in sid_idx))
    G = len(uniq)
    M = np.stack([scores[sid_idx == s] for s in uniq], axis=0)  # (G, K)
    K = M.shape[1]
    obs_acc_cnt = int(np.sum(M > 0))
    obs_all_cnt = int(np.sum(np.all(M > 0, axis=1)))
    obs_none_cnt = int(np.sum(np.all(M <= 0, axis=1)))

    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, G, 1))
    F = flips * M[None, :, :]
    null_acc_cnt = np.sum(F > 0, axis=(1, 2))
    null_all_cnt = np.sum(np.all(F > 0, axis=2), axis=1)
    p_acc = float((int(np.sum(null_acc_cnt >= obs_acc_cnt)) + 1) / (n_perm + 1))
    p_all = float((int(np.sum(null_all_cnt >= obs_all_cnt)) + 1) / (n_perm + 1))
    return {
        "acc": round(obs_acc_cnt / (G * K), 4),
        "all_in_setting_right": round(obs_all_cnt / G, 4),
        "counts_all_split_none": [obs_all_cnt, G - obs_all_cnt - obs_none_cnt, obs_none_cnt],
        "p_mc_uncentered_acc": round(p_acc, 5),
        "p_mc_uncentered_all": round(p_all, 5),
    }


def _unrounded_rand_spec(Dt_p, Qt_p, obs_acc_full_prec, n_rand=2000, seed=2481):
    """Exact full-precision 2,000-random-direction two-sided specificity null!"""
    D = Dt_p.shape[1]
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n_rand, D))
    if Qt_p is not None and Qt_p.shape[1] > 0:
        R = V234._project_out_rows(R, Qt_p)
    R /= np.maximum(1e-12, np.linalg.norm(R, axis=1, keepdims=True))
    accs = (Dt_p @ R.T > 0).mean(axis=0)
    abs_dev = np.abs(accs - 0.5)
    obs_dev = abs(float(obs_acc_full_prec) - 0.5)
    p_two = float((int(np.sum(abs_dev >= obs_dev - 1e-12)) + 1) / (n_rand + 1))
    return {
        "n_rand": n_rand,
        "rand_p95_two_sided": round(float(0.5 + np.percentile(abs_dev, 95)), 4),
        "p_vs_rand_two_sided": round(p_two, 5),
        "_p_fp": p_two,
    }


def _three_term_first_clause_rule(rows, extra_pos=()):
    """Exact Astra-6.2 3-term first-clause lexical parity rule stopping by word 9."""
    pos_set = set(E245.POS_LEX) | set(extra_pos)
    neg_set = set(E245.NEG_LEX)

    def score_sent(txt):
        ws = E245._words(txt)
        act_sign = 1 if (len(ws) > 1 and ws[1] in pos_set) else -1
        for i in range(2, len(ws) - 2):
            p1 = 1 if ws[i] in pos_set else (-1 if ws[i] in neg_set else 0)
            if p1 != 0 and ws[i + 1] == "the":
                p2 = 1 if ws[i + 2] in pos_set else (-1 if ws[i + 2] in neg_set else 0)
                if p2 != 0:
                    return act_sign * p1 * p2
        return 0

    wins = 0
    by_s = {}
    for r in rows:
        ok = score_sent(r["pos"]) > score_sent(r["neg"])
        wins += int(ok)
        by_s.setdefault(str(r["setting_id"]), []).append(ok)
    all4 = sum(1 for v in by_s.values() if all(v))
    return {"pairs_correct": wins, "n_pairs": len(rows), "acc": round(wins / len(rows), 4), "settings_all4": all4, "n_settings": len(by_s)}


def _setting_boot_ci_mask(item_acc_vec, mask, sid_idx, n_boot=2000, seed=2482):
    """Setting-clustered 95% bootstrap CI on mean(item_acc_vec[mask])."""
    uniq = sorted(set(int(s) for s in sid_idx))
    G = len(uniq)
    g_ix = [np.where((sid_idx == s) & mask)[0] for s in uniq]
    obs = float(np.mean(item_acc_vec[mask])) if np.any(mask) else None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ch = rng.integers(0, G, size=G)
        ix = np.concatenate([g_ix[c] for c in ch if len(g_ix[c]) > 0])
        if len(ix) > 0:
            vals.append(float(np.mean(item_acc_vec[ix])))
    arr = np.asarray(vals)
    return {
        "n_items": int(np.sum(mask)),
        "n_settings": int(sum(1 for ix in g_ix if len(ix) > 0)),
        "acc": round(obs, 4),
        "ci95": [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)],
    }


def _paired_boot_vs_nll(probe_correct_by_t, nll_correct_by_t, sid_idx, n_boot=5000, seed=2483):
    """Joint setting-clustered bootstrap of 5-target mean (probe_acc - nll_acc) matching Astra 6.2."""
    uniq = sorted(set(int(s) for s in sid_idx))
    G = len(uniq)
    tgts = list(probe_correct_by_t.keys())
    # Per setting diff averaged over 5 targets
    g_diffs = np.zeros(G, dtype=np.float64)
    for gi, s in enumerate(uniq):
        ix = (sid_idx == s)
        g_diffs[gi] = float(np.mean([probe_correct_by_t[tk][ix].mean() - nll_correct_by_t[tk][ix].mean() for tk in tgts]))
    obs = float(g_diffs.mean())
    rng = np.random.default_rng(seed)
    draws = g_diffs[rng.integers(0, G, size=(n_boot, G))].mean(axis=1)
    return {
        "diff": round(obs, 4),
        "ci95": [round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)],
    }


def main():
    t0 = time.time()
    actor_rows = json.loads((EXP3 / "bm_actor2x2_247_data.json").read_text())["bm_actor2x2_240"]
    bm120_rows = BM238.load_mirror_data()
    bmrel_rows = json.loads((EXP242 / "bm_rel242_data.json").read_text())["bm_rel_120"]
    bmneut_rows = json.loads((EXP245 / "bm_neut245_data.json").read_text())["bm_neut_120"]
    bmpolc_rows = json.loads((EXP245 / "bm_polconf245_data.json").read_text())["bm_polconf_120"]

    benches = {
        "bm_actor2x2_247": actor_rows,
        "bm120": bm120_rows,
        "bm_rel242": bmrel_rows,
        "bm_neut245": bmneut_rows,
        "bm_polconf245": bmpolc_rows,
    }
    ben_m = np.array([r["actor_type"] == "benign" for r in actor_rows], dtype=bool)
    adv_m = ~ben_m

    sid_map, split_map = {}, {}
    for bn, rows in benches.items():
        sb = {"bm120": 23800, "bm_rel242": 24200, "bm_neut245": 24500, "bm_polconf245": 24600, "bm_actor2x2_247": 24700}[bn]
        sid_map[bn], split_map[bn] = E245._sid_and_splits(rows, sb)

    # 1. Exact 3-term lexical rule check
    lex3_frozen = _three_term_first_clause_rule(actor_rows)
    lex3_plus1 = _three_term_first_clause_rule(actor_rows, extra_pos=("incorruptible",))

    # 2. Fixed unrounded NLL baselines (BOTH 6-model mean AND 5-non-Gemma / 5-non-Qwen means!)
    dnll_w, dnll_c = {}, {}
    nll_fixed = {}
    for k in MLX_KEYS:
        dr = np.load(DEPTH_DIR / f"bmactor247_{k}.npz")
        dw = dr["neg_nll_mean"].astype(np.float64) - dr["pos_nll_mean"].astype(np.float64)
        dc = dr["neg_changed_nll_mean"].astype(np.float64) - dr["pos_changed_nll_mean"].astype(np.float64)
        dnll_w[k], dnll_c[k] = dw, dc
        nll_fixed[k] = {
            "whole_sent": _unrounded_setting_perm(dw, sid_map["bm_actor2x2_247"], n_perm=2000, seed=2480),
            "changed_tokens_only": _unrounded_setting_perm(dc, sid_map["bm_actor2x2_247"], n_perm=2000, seed=2481),
        }
    nll_summary = {
        "six_model_whole_mean": round(float(np.mean([float(np.mean(dnll_w[k] > 0)) for k in MLX_KEYS])), 4),
        "six_model_changed_mean": round(float(np.mean([float(np.mean(dnll_c[k] > 0)) for k in MLX_KEYS])), 4),
        "five_non_gemma_whole_mean": round(float(np.mean([float(np.mean(dnll_w[k] > 0)) for k in MLX_KEYS if k != "gemma4_e4b"])), 4),
        "five_non_gemma_changed_mean": round(float(np.mean([float(np.mean(dnll_c[k] > 0)) for k in MLX_KEYS if k != "gemma4_e4b"])), 4),
        "five_non_qwen_whole_mean": round(float(np.mean([float(np.mean(dnll_w[k] > 0)) for k in MLX_KEYS if k != "qwen35_9b"])), 4),
        "five_non_qwen_changed_mean": round(float(np.mean([float(np.mean(dnll_c[k] > 0)) for k in MLX_KEYS if k != "qwen35_9b"])), 4),
        "per_model": nll_fixed,
    }

    # 3. Evaluative metadata + HOUSE setting-disjoint splits
    pairs_arch = AR.load_arch_pairs()
    house_idx = np.array([i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS])
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])
    hsets = [str(pairs_arch[i].get("setting", pairs_arch[i].get("id", i))) for i in house_idx]
    uniq_hs = sorted(set(hsets))
    hset_idx = np.array([{s: i for i, s in enumerate(uniq_hs)}[s] for s in hsets], dtype=int)
    house_splits = []
    for s_i in range(20):
        rng = np.random.default_rng(24010 + s_i)
        f0 = set(int(x) for x in rng.permutation(len(uniq_hs))[: len(uniq_hs) // 2])
        house_splits.append(np.array([0 if int(s) in f0 else 1 for s in hset_idx], dtype=int))

    sg_rows, ov_rows, _ = V234.load_valence_data()
    sg_doms = np.array([r["domain"] for r in sg_rows])
    n_sg = len(sg_rows)

    # 4. Within-model (source == target) 5-depth curves + conditional sentiment 5-depth curve + d50 full unrounded audit!
    within_model_curves = {pool: {bn: {} for bn in list(benches.keys()) + ["house_disjoint"]} for pool in ("mean", "last")}
    cond_sent_5depth = {pool: {src: {} for src in ("gemma4_e4b", "qwen35_9b")} for pool in ("mean", "last")}
    d50_actor_clean = {}

    # Also store d30, d50, d70 matrices for 4-contrast simultaneous Bonferroni across BOTH sources!
    Z_act_d = {pool: {} for pool in ("mean", "last")}
    Q5_act_d = {pool: {} for pool in ("mean", "last")}

    for slot, df in enumerate(DEPTH_FRACS):
        dkey = f"d{int(round(df * 100))}"
        print(f"Round-4 audit depth {dkey} ...", flush=True)
        for pool_key in ("mean", "last"):
            rot = E245._fit_neutral_rot_at_slot(slot, pool_key, d_plato=256)
            Z = {bn: {} for bn in benches}
            Dh_k, Q5_k, us_k = {}, {}, {}
            for k in MLX_KEYS:
                d240 = np.load(DEPTH_DIR / f"depth240_{k}.npz")
                Z["bm120"][k] = E245._map_mat(d240[f"bm120_pos_{pool_key}"][slot], rot, k) - E245._map_mat(d240[f"bm120_neg_{pool_key}"][slot], rot, k)
                for bn, pref in [("bm_rel242", "bmrel242"), ("bm_neut245", "bmneut245"), ("bm_polconf245", "bmpolconf245"), ("bm_actor2x2_247", "bmactor247")]:
                    dr = np.load(DEPTH_DIR / f"{pref}_{k}.npz")
                    Z[bn][k] = E245._map_mat(dr[f"pos_{pool_key}"][slot], rot, k) - E245._map_mat(dr[f"neg_{pool_key}"][slot], rot, k)
                Dh_k[k] = E245._map_mat(d240[f"house_pos_{pool_key}"][slot][house_idx], rot, k) - E245._map_mat(d240[f"house_neg_{pool_key}"][slot][house_idx], rot, k)
                Dv = E245._map_mat(d240[f"val_pos_{pool_key}"][slot], rot, k) - E245._map_mat(d240[f"val_neg_{pool_key}"][slot], rot, k)
                us_k[k] = AR.unit(Dv[:n_sg].mean(0))
                dom_vecs = [AR.unit(Dv[:n_sg][sg_doms == d].mean(0)) for d in sorted(set(sg_doms.tolist()))]
                Q5_k[k] = V234._orthonormal_basis(dom_vecs + [AR.unit(Dv[n_sg:].mean(0))])

            if dkey in ("d30", "d50", "d70"):
                Z_act_d[pool_key][dkey] = Z["bm_actor2x2_247"]
                Q5_act_d[pool_key][dkey] = Q5_k

            # Within-model (source == target, 20-split setting-disjoint CV perp VAL5) across all 6 MLX models!
            for bn in benches:
                m_accs = {}
                for k in MLX_KEYS:
                    Dp = V234._project_out_rows(Z[bn][k], Q5_k[k])
                    sp_a = []
                    for fl in split_map[bn]:
                        oof = np.zeros(len(Dp))
                        for f in (0, 1):
                            tr, te = (fl != f), (fl == f)
                            ut = AR.unit(Dp[tr].mean(0))
                            oof[te] = Dp[te] @ ut
                        sp_a.append(float(np.mean(oof > 0)))
                    m_accs[k] = round(float(np.mean(sp_a)), 4)
                within_model_curves[pool_key][bn][dkey] = {
                    "mean_6models": round(float(np.mean(list(m_accs.values()))), 4),
                    "per_model": m_accs,
                }
            # Within-model HOUSE setting-disjoint perp VAL5
            h_accs = {}
            for k in MLX_KEYS:
                Dp = V234._project_out_rows(Dh_k[k], Q5_k[k])
                sp_a = []
                for fl in house_splits:
                    oof = np.zeros(len(Dp))
                    for f in (0, 1):
                        tr, te = (fl != f), (fl == f)
                        ut = AR.house_dir(Dp[tr], house_cats[tr])
                        oof[te] = Dp[te] @ ut
                    sp_a.append(float(np.mean(oof > 0)))
                h_accs[k] = round(float(np.mean(sp_a)), 4)
            within_model_curves[pool_key]["house_disjoint"][dkey] = {
                "mean_6models": round(float(np.mean(list(h_accs.values()))), 4),
                "per_model": h_accs,
            }

            # Conditional actor-signed sentiment control on bm_actor2x2_247 across all 5 depths
            for src_k in ("gemma4_e4b", "qwen35_9b"):
                tgts = [k for k in MLX_KEYS if k != src_k]
                cs_list = []
                for tk in tgts:
                    sc = Z["bm_actor2x2_247"][tk] @ us_k[src_k]
                    # Keep +sign on benign, reverse -sign on adverse
                    cond_ok = np.where(ben_m, sc > 0, sc < 0)
                    cs_list.append(float(np.mean(cond_ok)))
                cond_sent_5depth[pool_key][src_k][dkey] = round(float(np.mean(cs_list)), 4)

            # Full unrounded, split-averaged d50 audit on bm_actor2x2_247!
            if dkey == "d50":
                d50_actor_clean[pool_key] = {}
                sid_a = sid_map["bm_actor2x2_247"]
                spl_a = split_map["bm_actor2x2_247"]
                for src_k in ("gemma4_e4b", "qwen35_9b"):
                    tgts = [k for k in MLX_KEYS if k != src_k]
                    Qs5 = Q5_k[src_k]
                    Ds5 = V234._project_out_rows(Z["bm_actor2x2_247"][src_k], Qs5)
                    u_bm120 = AR.unit(Z["bm120"][src_k].mean(0))
                    u_neut = AR.unit(Z["bm_neut245"][src_k].mean(0))

                    tgt_d = {}
                    cv_hit_by_t, xd_bm_hit_by_t, xd_nt_hit_by_t = {}, {}, {}
                    nll_w_hit_by_t, nll_c_hit_by_t = {}, {}

                    for tk in tgts:
                        Qt5 = Q5_k[tk]
                        Dt = Z["bm_actor2x2_247"][tk]
                        Dt5 = V234._project_out_rows(Dt, Qt5)
                        nll_w_hit_by_t[tk] = (dnll_w[tk] > 0).astype(np.float64)
                        nll_c_hit_by_t[tk] = (dnll_c[tk] > 0).astype(np.float64)

                        # 20-split-averaged per-item accuracy vector (100% consistent with headline!)
                        item_hits = np.zeros(len(actor_rows), dtype=np.float64)
                        item_margins = np.zeros(len(actor_rows), dtype=np.float64)
                        sp_all4 = []
                        for fl in spl_a:
                            oof = np.zeros(len(actor_rows))
                            for f in (0, 1):
                                tr, te = (fl != f), (fl == f)
                                ut = V234._project_out(AR.unit(Ds5[tr].mean(0)), Qt5)
                                oof[te] = Dt5[te] @ ut
                            item_hits += (oof > 0).astype(np.float64) / len(spl_a)
                            item_margins += oof / len(spl_a)
                            # Exact integer all-4 per split
                            sp_all4.append(sum(1 for s in range(60) if np.all(oof[sid_a == s] > 0)) / 60.0)
                        cv_acc_fp = float(item_hits.mean())
                        cv_hit_by_t[tk] = item_hits

                        # Zero-shot xdom_bm120 & xdom_bmneut (full-precision unrounded nulls!)
                        sc_xb = Dt5 @ V234._project_out(V234._project_out(u_bm120, Qs5), Qt5)
                        sc_xn = Dt5 @ V234._project_out(V234._project_out(u_neut, Qs5), Qt5)
                        xd_bm_hit_by_t[tk] = (sc_xb > 0).astype(np.float64)
                        xd_nt_hit_by_t[tk] = (sc_xn > 0).astype(np.float64)

                        # OLS regression of split-mean probe margin on changed-token DeltaNLL (setting-clustered intercept CI!)
                        ols_chg = BM238._setting_clustered_ols_intercept_ci(item_margins, dnll_c[tk], sid_a, seed=2485)

                        tgt_d[tk] = {
                            "zs_sent_pos": round(float(np.mean((Dt @ us_k[src_k]) > 0)), 4),
                            "zs_sent_cond_oracle": round(float(np.mean(np.where(ben_m, (Dt @ us_k[src_k]) > 0, (Dt @ us_k[src_k]) < 0))), 4),
                            "cv2_split_avg": {
                                "acc": round(cv_acc_fp, 4),
                                "benign_acc": round(float(item_hits[ben_m].mean()), 4),
                                "adverse_acc": round(float(item_hits[adv_m].mean()), 4),
                                "all4_right": round(float(np.mean(sp_all4)), 4),
                                "spec_rand2000": _unrounded_rand_spec(Dt5, Qt5, cv_acc_fp, n_rand=2000, seed=2481),
                                "whole_dnll_le_0_ci": _setting_boot_ci_mask(item_hits, dnll_w[tk] <= 0, sid_a, seed=2482),
                                "changed_dnll_le_0_ci": _setting_boot_ci_mask(item_hits, dnll_c[tk] <= 0, sid_a, seed=2483),
                                "ols_margin_vs_changed_dnll": ols_chg,
                            },
                            "xdom_bm120_val5": {
                                **_unrounded_setting_perm(sc_xb, sid_a, n_perm=2000, seed=2484),
                                "benign_acc": round(float(np.mean(sc_xb[ben_m] > 0)), 4),
                                "adverse_acc": round(float(np.mean(sc_xb[adv_m] > 0)), 4),
                                "spec_rand2000": _unrounded_rand_spec(Dt5, Qt5, float(np.mean(sc_xb > 0)), n_rand=2000, seed=2485),
                            },
                            "xdom_bmneut_val5": {
                                **_unrounded_setting_perm(sc_xn, sid_a, n_perm=2000, seed=2486),
                                "benign_acc": round(float(np.mean(sc_xn[ben_m] > 0)), 4),
                                "adverse_acc": round(float(np.mean(sc_xn[adv_m] > 0)), 4),
                                "spec_rand2000": _unrounded_rand_spec(Dt5, Qt5, float(np.mean(sc_xn > 0)), n_rand=2000, seed=2487),
                            },
                        }

                    holm_cv = _holm({t: tgt_d[t]["cv2_split_avg"]["spec_rand2000"]["_p_fp"] for t in tgts})
                    holm_x120 = _holm({t: tgt_d[t]["xdom_bm120_val5"]["spec_rand2000"]["_p_fp"] for t in tgts})
                    holm_xnt = _holm({t: tgt_d[t]["xdom_bmneut_val5"]["spec_rand2000"]["_p_fp"] for t in tgts})
                    for t in tgts:
                        tgt_d[t]["cv2_split_avg"]["spec_rand2000"].pop("_p_fp", None)
                        tgt_d[t]["xdom_bm120_val5"]["spec_rand2000"].pop("_p_fp", None)
                        tgt_d[t]["xdom_bmneut_val5"]["spec_rand2000"].pop("_p_fp", None)

                    d50_actor_clean[pool_key][src_k] = {
                        "paired_vs_whole_nll": {
                            "cv2": _paired_boot_vs_nll(cv_hit_by_t, nll_w_hit_by_t, sid_a, seed=2488),
                            "xdom_bm120": _paired_boot_vs_nll(xd_bm_hit_by_t, nll_w_hit_by_t, sid_a, seed=2488),
                            "xdom_bmneut": _paired_boot_vs_nll(xd_nt_hit_by_t, nll_w_hit_by_t, sid_a, seed=2488),
                        },
                        "paired_vs_changed_nll": {
                            "cv2": _paired_boot_vs_nll(cv_hit_by_t, nll_c_hit_by_t, sid_a, seed=2489),
                            "xdom_bm120": _paired_boot_vs_nll(xd_bm_hit_by_t, nll_c_hit_by_t, sid_a, seed=2489),
                            "xdom_bmneut": _paired_boot_vs_nll(xd_nt_hit_by_t, nll_c_hit_by_t, sid_a, seed=2489),
                        },
                        "holm_rand2000_cv2": holm_cv,
                        "holm_rand2000_xdom_bm120": holm_x120,
                        "holm_rand2000_xdom_bmneut": holm_xnt,
                        "targets": tgt_d,
                    }

    # 4b. Simultaneous 4-contrast Bonferroni CIs (cross-family, both sources) + Within-model 2-contrast Bonf CIs on bm_actor2x2_247!
    actor_depth_bonf = {}
    sid_a = sid_map["bm_actor2x2_247"]
    spl_a = split_map["bm_actor2x2_247"]
    uniq_s = sorted(set(int(s) for s in sid_a))
    G_a = len(uniq_s)
    # Precompute setting-level masks & sums for exact, fast vectorized refit-in-bootstrap!
    by_s = [np.where(sid_a == s)[0] for s in uniq_s]
    # Check each setting belongs cleanly to a single fold in every split
    split_s_fold = np.zeros((len(spl_a), G_a), dtype=int)
    for si, fl in enumerate(spl_a):
        for g in range(G_a):
            vals = set(int(x) for x in fl[by_s[g]])
            assert len(vals) == 1
            split_s_fold[si, g] = next(iter(vals))

    def _refit_cv_boot_matrix(Ds_p, Dt_p_list, Qt_p_list, boot_w_mat):
        """Exact refit-in-bootstrap over (B, G) setting multiplicity weights; returns (B,) accuracies."""
        B = boot_w_mat.shape[0]
        S_src = np.stack([Ds_p[by_s[g]].sum(0) for g in range(G_a)], axis=0)  # (G, D)
        # Item-to-setting index map (240,)
        s_of_i = np.array([int(x) for x in sid_a], dtype=int)
        tot_hits = np.zeros(B, dtype=np.float64)
        for si, fl in enumerate(spl_a):
            sf = split_s_fold[si]
            for f in (0, 1):
                tr_g = (sf != f).astype(np.float64)
                te_i = np.where(fl == f)[0]
                te_s = s_of_i[te_i]  # (120,)
                w_tr = boot_w_mat * tr_g[None, :]  # (B, G)
                U_raw = w_tr @ S_src  # (B, D)
                U_raw /= np.maximum(1e-12, np.linalg.norm(U_raw, axis=1, keepdims=True))
                w_te_i = boot_w_mat[:, te_s]  # (B, 120)
                for Dt_p, Qt_p in zip(Dt_p_list, Qt_p_list):
                    Ut = V234._project_out_rows(U_raw, Qt_p)
                    Ut /= np.maximum(1e-12, np.linalg.norm(Ut, axis=1, keepdims=True))
                    scores = Ut @ Dt_p[te_i].T  # (B, 120)
                    tot_hits += ((scores > 0).astype(np.float64) * w_te_i).sum(axis=1)
        tot_w_items = boot_w_mat.sum(axis=1) * 4.0  # 4 items per setting
        return tot_hits / (tot_w_items * len(Dt_p_list) * len(spl_a))

    for pool_key in ("mean", "last"):
        Dp_d = {dk: {k: V234._project_out_rows(Z_act_d[pool_key][dk][k], Q5_act_d[pool_key][dk][k]) for k in MLX_KEYS} for dk in ("d30", "d50", "d70")}
        rng_b = np.random.default_rng(24891)
        n_boot = 1000
        boot_w = np.array([np.bincount(rng_b.integers(0, G_a, size=G_a), minlength=G_a) for _ in range(n_boot)], dtype=np.float64)
        ones_w = np.ones((1, G_a), dtype=np.float64)

        xfam_out = {}
        for src_k in ("gemma4_e4b", "qwen35_9b"):
            tgts = [k for k in MLX_KEYS if k != src_k]
            obs, draws = {}, {}
            for dk in ("d30", "d50", "d70"):
                Ds_p = Dp_d[dk][src_k]
                Dt_l = [Dp_d[dk][t] for t in tgts]
                Qt_l = [Q5_act_d[pool_key][dk][t] for t in tgts]
                obs[dk] = float(_refit_cv_boot_matrix(Ds_p, Dt_l, Qt_l, ones_w)[0])
                draws[dk] = _refit_cv_boot_matrix(Ds_p, Dt_l, Qt_l, boot_w)
            d50_30 = draws["d50"] - draws["d30"]
            d50_70 = draws["d50"] - draws["d70"]
            xfam_out[src_k] = {
                "d30": round(obs["d30"], 4), "d50": round(obs["d50"], 4), "d70": round(obs["d70"], 4),
                "d50_minus_d30": {
                    "obs": round(obs["d50"] - obs["d30"], 4),
                    "ci95_bonf2": [round(float(np.percentile(d50_30, 1.25)), 4), round(float(np.percentile(d50_30, 98.75)), 4)],
                    "ci95_bonf4_both_sources": [round(float(np.percentile(d50_30, 0.625)), 4), round(float(np.percentile(d50_30, 99.375)), 4)],
                },
                "d50_minus_d70": {
                    "obs": round(obs["d50"] - obs["d70"], 4),
                    "ci95_bonf2": [round(float(np.percentile(d50_70, 1.25)), 4), round(float(np.percentile(d50_70, 98.75)), 4)],
                    "ci95_bonf4_both_sources": [round(float(np.percentile(d50_70, 0.625)), 4), round(float(np.percentile(d50_70, 99.375)), 4)],
                },
            }

        wm_obs, wm_draws = {}, {}
        for dk in ("d30", "d50", "d70"):
            wm_obs[dk] = float(np.mean([_refit_cv_boot_matrix(Dp_d[dk][k], [Dp_d[dk][k]], [Q5_act_d[pool_key][dk][k]], ones_w)[0] for k in MLX_KEYS]))
            wm_draws[dk] = np.mean([_refit_cv_boot_matrix(Dp_d[dk][k], [Dp_d[dk][k]], [Q5_act_d[pool_key][dk][k]], boot_w) for k in MLX_KEYS], axis=0)
        w50_30 = wm_draws["d50"] - wm_draws["d30"]
        w50_70 = wm_draws["d50"] - wm_draws["d70"]
        actor_depth_bonf[pool_key] = {
            "cross_family_both_sources": xfam_out,
            "within_model_6model_mean": {
                "d30": round(wm_obs["d30"], 4), "d50": round(wm_obs["d50"], 4), "d70": round(wm_obs["d70"], 4),
                "d50_minus_d30": {
                    "obs": round(wm_obs["d50"] - wm_obs["d30"], 4),
                    "ci95": [round(float(np.percentile(w50_30, 2.5)), 4), round(float(np.percentile(w50_30, 97.5)), 4)],
                    "ci95_bonf2": [round(float(np.percentile(w50_30, 1.25)), 4), round(float(np.percentile(w50_30, 98.75)), 4)],
                },
                "d50_minus_d70": {
                    "obs": round(wm_obs["d50"] - wm_obs["d70"], 4),
                    "ci95": [round(float(np.percentile(w50_70, 2.5)), 4), round(float(np.percentile(w50_70, 97.5)), 4)],
                    "ci95_bonf2": [round(float(np.percentile(w50_70, 1.25)), 4), round(float(np.percentile(w50_70, 98.75)), 4)],
                },
            },
        }

    # 5. Save BOTH 13 retained myth IDs AND 69 retained mechanical IDs for items3 + verify Exp 244 question-overlap AUROC!
    ev_rows = [json.loads(l) for l in open(PLATO_CACHE / "eval_passages.jsonl")]
    ret_myth = [r["id"] for r in ev_rows if r["domain"] == "items3_myth" and r["id"].endswith("_anchor") and not DA247._strict_has_proper_name(r["text"])]
    ret_mech = [r["id"] for r in ev_rows if r["domain"] == "items3_mech" and r["id"].endswith("_anchor") and not DA247._strict_has_proper_name(r["text"])]

    epi_pairs = json.loads((EXP245 / "epi_matched245_data.json").read_text())["pairs"]
    def _q_overlap(q, ctx):
        qw = set(re.findall(r"[a-z]+", q.lower()))
        cw = set(re.findall(r"[a-z]+", ctx.lower()))
        return len(qw & cw) / max(1, len(qw))
    pos_q = np.array([_q_overlap(r["question"], r["known_context"]) for r in epi_pairs], dtype=np.float64)
    neg_q = np.array([_q_overlap(r["question"], r["unknown_context"]) for r in epi_pairs], dtype=np.float64)
    epi_qov_auroc = float(np.mean(pos_q[:, None] > neg_q[None, :]) + 0.5 * np.mean(pos_q[:, None] == neg_q[None, :]))

    report = {
        "elapsed_sec": round(time.time() - t0, 1),
        "three_term_first_clause_lexical_rule": {"frozen_POS_LEX": lex3_frozen, "plus_incorruptible": lex3_plus1},
        "nll_baselines_unrounded": nll_summary,
        "conditional_oracle_sentiment_5depth": cond_sent_5depth,
        "within_model_source_eq_target_5depth": within_model_curves,
        "actor2x2_depth_bonf_refit_boot": actor_depth_bonf,
        "d50_actor2x2_unrounded_split_avg_audit": d50_actor_clean,
        "items3_strict_retained_ids": {"n_myth": len(ret_myth), "myth_ids": ret_myth, "n_mech": len(ret_mech), "mech_ids": ret_mech},
        "exp244_question_word_overlap_auroc": round(epi_qov_auroc, 6),
    }
    out_p = EXP4 / "round4_binding_depth_results.json"
    out_p.write_text(json.dumps(report, indent=2))
    print(f"SAVED {out_p} in {report['elapsed_sec']}s", flush=True)
    print("3-term rule:", report["three_term_first_clause_lexical_rule"])
    print("NLL 6-model vs 5-target:", nll_summary["six_model_whole_mean"], nll_summary["six_model_changed_mean"], "|", nll_summary["five_non_gemma_whole_mean"], nll_summary["five_non_gemma_changed_mean"])
    print("NLL unrounded p_all:", {k: nll_fixed[k]["whole_sent"]["p_mc_uncentered_all"] for k in MLX_KEYS})
    for pool in ("mean", "last"):
        print(f"WITHIN-MODEL 6-model mean [{pool}]:")
        for bn in ("bm_actor2x2_247", "bm120", "bm_rel242", "bm_neut245", "bm_polconf245", "house_disjoint"):
            cs = [within_model_curves[pool][bn][dk]["mean_6models"] for dk in ("d15", "d30", "d50", "d70", "d85")]
            print(f"  {bn:16s}: {cs}")
    for src in ("gemma4_e4b", "qwen35_9b"):
        blk = d50_actor_clean["mean"][src]
        print(f"d50 MEAN {src} vs whole NLL:", blk["paired_vs_whole_nll"], "| vs changed NLL:", blk["paired_vs_changed_nll"])
        print(f"  Holm rand2000: cv2={blk['holm_rand2000_cv2']} | xd120={blk['holm_rand2000_xdom_bm120']} | xdNt={blk['holm_rand2000_xdom_bmneut']}")


if __name__ == "__main__":
    main()
