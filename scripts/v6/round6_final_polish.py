"""Round-6 Final Sign-Off Polish Script (Closes Every Remaining Round-5 Medium & Low Finding from Fable 5.2 and Astra 6.2).

Computes and serializes:
  1. Claim A:
     - Bonferroni-20 (alpha=0.05/20, percentile & basic reflected) alongside nominal 95% refit-in-bootstrap
       intervals for all 20 d50.mean DeltaNLL <= 0 strata (2,000 draws).
     - Text-derived frozen-lexical-switch sentiment baseline (+zs_sent if word 2 in POS_LEX, else -zs_sent)
       at d50.mean and d50.last (reproducing 238/240 agreement with gold actor_type).
     - Full-precision d50.mean Qwen->5 adverse accuracy and bm_neut245->actor setting-level all-4 Holm-5.
  2. Claim B:
     - Cross-pool Bonferroni-8 (alpha=0.05/8, percentile & basic reflected) across all 8 cross-family depth
       contrasts (mean & last x Gemma & Qwen x d50-d30 & d50-d70).
  3. Claim C:
     - BOTH percentile and basic reflected 95% CIs on the joint 40-Jung-setting + target-ATU-type bootstrap
       (2,000 draws) on both 480 and 476 records.
     - Shared-draw 20,000-resample stratified bootstrap (identical draws for 9-bin 10-way and 9-way argmax so
       the identical macro-AUROC statistic receives the exact same unadjusted p-value), plus joint 16-cell Holm
       combining main 10-bin and 9-bin/9-way all-3-lexical max_p comparisons.
     - True 9-class refit (removing the 39 Western 10_Stupid_Ogre_Devil training records -> 84 train / 135 test
       on 480; 85 train / 120 test on 480_excl15) with shared 20,000-draw all-3-lexical Holm-8 and chance Holm-8.
     - Complete 476-record structural suite (fixed-corpus Holm-7, 200-base shuffle, 5-pair diagonal, type-preserving
       PR, and exact 8!=40,320 geometry).
     - Exact original ATU construction range boundaries alongside observed sample min/max and exact 9-bin
       test-majority oracle denominators (135 on 480; 120 on 480_excl15).
"""
import itertools
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np

SCRATCH = Path(os.environ.get("JPL_WORK_DIR", "."))
EXP6 = SCRATCH / "exp250_round6"
EXP6.mkdir(parents=True, exist_ok=True)
EXP5 = SCRATCH / "exp249_round5"
EXP4 = SCRATCH / "exp248_round4"
EXP3 = SCRATCH / "exp247_round3"
EXP245 = SCRATCH / "exp245_controls"
FOLK = SCRATCH / "folklore_test"
REPO_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(Path(__file__).resolve().parents[2] / "bench" / "archetypes" / "fluent" / "plato")))
DEPTH_DIR = PLATO_CACHE / "depth240"

for p in (str(REPO_ARCH), str(REPO_ARCH / "fluent"), str(FOLK), str(EXP245), str(EXP3), str(EXP4), str(EXP5)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import named_arch235 as NA235  # noqa: E402
import nonjung_roles237 as NJ237  # noqa: E402
import real_myth_basis239 as R239  # noqa: E402
import exp245_binding as E245  # noqa: E402
import exp247_actor_binding as E247  # noqa: E402
import aft_audit245 as A245  # noqa: E402
import round4_folklore_audit as R4F  # noqa: E402

MLX_KEYS = E247.MLX_KEYS
DEPTH_FRACS = E247.DEPTH_FRACS
FRAMING_SENS_AFT_IDXS = {277, 356, 383, 1390}

ORIGINAL_ATU_CONSTRUCTION_RANGES = {
    "01_Animal_Trickster": [1, 299],
    "02_Supernatural_Adversary": [300, 399],
    "03_Supernatural_Spouse": [400, 459],
    "05_Supernatural_Helpers": [500, 559],
    "06_Magic_Objects": [560, 649],
    "07_Supernatural_Power": [650, 749],
    "08_Religious_Tales": [750, 849],
    "09_Realistic_Wisdom": [850, 999],
    "10_Stupid_Ogre_Devil": [1000, 1199],
    "11_Jokes_Tricksters": [1200, 1999],
}


def _holm(p_map):
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for r, (k, pv) in enumerate(items):
        run = max(run, min(1.0, (m - r) * pv))
        out[k] = round(run, 5)
    return out


def _pct_and_basic(obs, draws, alpha, digits=4):
    q_lo = float(np.percentile(draws, 100.0 * (alpha / 2.0)))
    q_hi = float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "obs": round(float(obs), digits),
        "percentile": [round(q_lo, digits), round(q_hi, digits)],
        "basic_reflected": [round(2.0 * float(obs) - q_hi, digits), round(2.0 * float(obs) - q_lo, digits)],
    }


def main():
    t0 = time.time()

    # =========================================================================
    # PART 1: CLAIM A & B POLISH
    # =========================================================================
    actor_rows = json.loads((EXP3 / "bm_actor2x2_247_data.json").read_text())["bm_actor2x2_240"]
    sid_a, spl_a = E245._sid_and_splits(actor_rows, 24700)
    G_a = len(set(int(s) for s in sid_a))
    by_s = [np.where(sid_a == s)[0] for s in range(G_a)]
    split_s_fold = np.array([[int(fl[by_s[g][0]]) for g in range(G_a)] for fl in spl_a], dtype=int)
    s_of_i = sid_a.astype(int)
    ben_m = np.array([r["actor_type"] == "benign" for r in actor_rows], dtype=bool)

    # Text-derived frozen-lexical switch (word 2 in POS_LEX -> +1, else -1)
    pos_lex = set(E245.POS_LEX)
    text_ben_m = np.array([(len(E245._words(r["pos"])) > 1 and E245._words(r["pos"])[1] in pos_lex) for r in actor_rows], dtype=bool)
    text_switch_agree = int(np.sum(text_ben_m == ben_m))

    dnll_w, dnll_c = {}, {}
    for k in MLX_KEYS:
        dr = np.load(DEPTH_DIR / f"bmactor247_{k}.npz")
        dnll_w[k] = dr["neg_nll_mean"].astype(np.float64) - dr["pos_nll_mean"].astype(np.float64)
        dnll_c[k] = dr["neg_changed_nll_mean"].astype(np.float64) - dr["pos_changed_nll_mean"].astype(np.float64)

    sg_rows, _, _ = V234.load_valence_data()
    sg_doms = np.array([r["domain"] for r in sg_rows])
    n_sg = len(sg_rows)

    Dp_256 = {pool: {} for pool in ("mean", "last")}
    Q5_256 = {pool: {} for pool in ("mean", "last")}
    Z_256_d50 = {pool: {} for pool in ("mean", "last")}
    us_d50 = {pool: {} for pool in ("mean", "last")}

    for slot, df in enumerate(DEPTH_FRACS):
        dkey = f"d{int(round(df * 100))}"
        if dkey not in ("d30", "d50", "d70"):
            continue
        for pool_key in ("mean", "last"):
            rot = E245._fit_neutral_rot_at_slot(slot, pool_key, d_plato=256)
            Dp_256[pool_key][dkey] = {}
            Q5_256[pool_key][dkey] = {}
            for k in MLX_KEYS:
                d240 = np.load(DEPTH_DIR / f"depth240_{k}.npz")
                dr = np.load(DEPTH_DIR / f"bmactor247_{k}.npz")
                Z = E245._map_mat(dr[f"pos_{pool_key}"][slot], rot, k) - E245._map_mat(dr[f"neg_{pool_key}"][slot], rot, k)
                Dv = E245._map_mat(d240[f"val_pos_{pool_key}"][slot], rot, k) - E245._map_mat(d240[f"val_neg_{pool_key}"][slot], rot, k)
                dom_v = [AR.unit(Dv[:n_sg][sg_doms == d].mean(0)) for d in sorted(set(sg_doms.tolist()))]
                Q5 = V234._orthonormal_basis(dom_v + [AR.unit(Dv[n_sg:].mean(0))])
                Q5_256[pool_key][dkey][k] = Q5
                Dp_256[pool_key][dkey][k] = V234._project_out_rows(Z, Q5)
                if dkey == "d50":
                    Z_256_d50[pool_key][k] = Z
                    us_d50[pool_key][k] = AR.unit(Dv[:n_sg].mean(0))

    # 1a. Text-derived sentiment baseline at d50.mean and d50.last
    text_switch_sent = {"switch_agreement_with_gold": f"{text_switch_agree}/{len(actor_rows)}"}
    for pool_key in ("mean", "last"):
        text_switch_sent[pool_key] = {}
        for src_k in ("gemma4_e4b", "qwen35_9b"):
            tgts = [k for k in MLX_KEYS if k != src_k]
            accs_txt, accs_gold = {}, {}
            for tk in tgts:
                sc = Z_256_d50[pool_key][tk] @ us_d50[pool_key][src_k]
                accs_txt[tk] = round(float(np.mean(np.where(text_ben_m, sc > 0, sc < 0))), 4)
                accs_gold[tk] = round(float(np.mean(np.where(ben_m, sc > 0, sc < 0))), 4)
            text_switch_sent[pool_key][src_k] = {
                "text_switch_5tgt_mean": round(float(np.mean([float(np.mean(np.where(text_ben_m, (Z_256_d50[pool_key][tk] @ us_d50[pool_key][src_k]) > 0, (Z_256_d50[pool_key][tk] @ us_d50[pool_key][src_k]) < 0))) for tk in tgts])), 4),
                "gold_oracle_5tgt_mean": round(float(np.mean([float(np.mean(np.where(ben_m, (Z_256_d50[pool_key][tk] @ us_d50[pool_key][src_k]) > 0, (Z_256_d50[pool_key][tk] @ us_d50[pool_key][src_k]) < 0))) for tk in tgts])), 4),
                "per_target_text_switch": accs_txt,
            }

    # 1b. Exact full-precision d50.mean Qwen->5 adverse & bm_neut->actor setting all-4 Holm-5 from Round-4 JSON
    r4_bd = json.loads((EXP4 / "round4_binding_depth_results.json").read_text())
    xneut_all4_holm = {}
    for src_k in ("gemma4_e4b", "qwen35_9b"):
        tgts_d = r4_bd["d50_actor2x2_unrounded_split_avg_audit"]["mean"][src_k]["targets"]
        xneut_all4_holm[src_k] = _holm({tk: tgts_d[tk]["xdom_bmneut_val5"]["p_mc_uncentered_all"] for tk in tgts_d})

    def _refit_cv_boot_all(Ds_p, Dt_p_list, Qt_p_list, boot_w_mat, mask_list=None):
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
                    Ut = V234._project_out_rows(U_raw, Qt_p)
                    Ut /= np.maximum(1e-12, np.linalg.norm(Ut, axis=1, keepdims=True))
                    sc = Ut @ Dt_p[te_i].T
                    w_eff = w_te_base * mask_list[ti][te_i][None, :].astype(np.float64) if mask_list is not None else w_te_base
                    hits[:, ti] += ((sc > 0).astype(np.float64) * w_eff).sum(axis=1)
                    den[:, ti] += w_eff.sum(axis=1)
        return hits / np.maximum(1e-12, den)

    # 1c. Exact Bonf-20 (alpha=0.05/20) alongside nominal 95% on all 20 d50.mean DeltaNLL <= 0 strata (same seed 24920!)
    rng_cv = np.random.default_rng(24920)
    boot_w_cv = np.array([np.bincount(rng_cv.integers(0, G_a, size=G_a), minlength=G_a) for _ in range(2000)], dtype=np.float64)
    ones_w = np.ones((1, G_a), dtype=np.float64)

    strata_bonf20 = {}
    cnt_w_nom = cnt_w_b20 = cnt_c_nom = cnt_c_b20 = 0
    for src_k in ("gemma4_e4b", "qwen35_9b"):
        tgts = [k for k in MLX_KEYS if k != src_k]
        Ds_p = Dp_256["mean"]["d50"][src_k]
        Dt_l = [Dp_256["mean"]["d50"][t] for t in tgts]
        Qt_l = [Q5_256["mean"]["d50"][t] for t in tgts]
        mw_l = [dnll_w[t] <= 0 for t in tgts]
        mc_l = [dnll_c[t] <= 0 for t in tgts]
        obs_sw = _refit_cv_boot_all(Ds_p, Dt_l, Qt_l, ones_w, mw_l)[0]
        dr_sw = _refit_cv_boot_all(Ds_p, Dt_l, Qt_l, boot_w_cv, mw_l)
        obs_sc = _refit_cv_boot_all(Ds_p, Dt_l, Qt_l, ones_w, mc_l)[0]
        dr_sc = _refit_cv_boot_all(Ds_p, Dt_l, Qt_l, boot_w_cv, mc_l)
        # Also exact full-precision adverse mean for Qwen->5 / Gemma->5
        obs_adv = _refit_cv_boot_all(Ds_p, Dt_l, Qt_l, ones_w, [~ben_m] * 5)[0]
        per_t = {}
        for ti, tk in enumerate(tgts):
            wn = _pct_and_basic(obs_sw[ti], dr_sw[:, ti], 0.05)
            wb = _pct_and_basic(obs_sw[ti], dr_sw[:, ti], 0.05 / 20.0)
            cn = _pct_and_basic(obs_sc[ti], dr_sc[:, ti], 0.05)
            cb = _pct_and_basic(obs_sc[ti], dr_sc[:, ti], 0.05 / 20.0)
            if wn["percentile"][0] > 0.5 and wn["basic_reflected"][0] > 0.5:
                cnt_w_nom += 1
            if wb["percentile"][0] > 0.5 and wb["basic_reflected"][0] > 0.5:
                cnt_w_b20 += 1
            if cn["percentile"][0] > 0.5 and cn["basic_reflected"][0] > 0.5:
                cnt_c_nom += 1
            if cb["percentile"][0] > 0.5 and cb["basic_reflected"][0] > 0.5:
                cnt_c_b20 += 1
            per_t[tk] = {"whole_nom95": wn, "whole_bonf20": wb, "changed_nom95": cn, "changed_bonf20": cb}
        strata_bonf20[src_k] = {"full_prec_adverse_5tgt_mean": round(float(obs_adv.mean()), 4), "targets": per_t}
    strata_bonf20["pass_counts_both_pct_and_basic_gt_050"] = {
        "whole_nom95": f"{cnt_w_nom}/10", "whole_bonf20": f"{cnt_w_b20}/10",
        "changed_nom95": f"{cnt_c_nom}/10", "changed_bonf20": f"{cnt_c_b20}/10",
    }

    # 1d. Cross-pool Bonf-8 (alpha=0.05/8) across all 8 cross-family depth contrasts (same seed 24930!)
    rng_d = np.random.default_rng(24930)
    boot_w_d = np.array([np.bincount(rng_d.integers(0, G_a, size=G_a), minlength=G_a) for _ in range(2000)], dtype=np.float64)
    depth_bonf8 = {}
    for pool_key in ("mean", "last"):
        depth_bonf8[pool_key] = {}
        for src_k in ("gemma4_e4b", "qwen35_9b"):
            tgts = [k for k in MLX_KEYS if k != src_k]
            o_d, dr_d = {}, {}
            for dk in ("d30", "d50", "d70"):
                o_d[dk] = float(_refit_cv_boot_all(Dp_256[pool_key][dk][src_k], [Dp_256[pool_key][dk][t] for t in tgts], [Q5_256[pool_key][dk][t] for t in tgts], ones_w).mean())
                dr_d[dk] = _refit_cv_boot_all(Dp_256[pool_key][dk][src_k], [Dp_256[pool_key][dk][t] for t in tgts], [Q5_256[pool_key][dk][t] for t in tgts], boot_w_d).mean(axis=1)
            depth_bonf8[pool_key][src_k] = {
                "d50_minus_d30_bonf8_across_pools": _pct_and_basic(o_d["d50"] - o_d["d30"], dr_d["d50"] - dr_d["d30"], 0.05 / 8.0),
                "d50_minus_d70_bonf8_across_pools": _pct_and_basic(o_d["d50"] - o_d["d70"], dr_d["d50"] - dr_d["d70"], 0.05 / 8.0),
            }

    # =========================================================================
    # PART 2: CLAIM C POLISH
    # =========================================================================
    aft_rows = json.loads((FOLK / "aft_real_myths_500.json").read_text())
    m4 = json.loads((EXP4 / "aft_clean248_manifest.json").read_text())
    jung_rows = NA235.load_named_arch_data()
    fam_rows = json.loads((FOLK / "family_roles239_data.json").read_text())
    jobs_rows = NJ237.load_nonjung_data()

    div_names = list(R239.ATU_DIVS)
    div2i = {d: i for i, d in enumerate(div_names)}
    y_a = np.array([div2i[r["division"]] for r in aft_rows], dtype=int)
    atu_a = np.array([str(r["atu_id"]) for r in aft_rows])
    aidx_a = np.array([int(r["aft_idx"]) for r in aft_rows], dtype=int)
    keep480 = np.array([bool(m["included"]) for m in m4], dtype=bool)
    keep476 = keep480 & (~np.isin(aidx_a, list(FRAMING_SENS_AFT_IDXS)))
    prov_a = np.array([m["adjudicated_provenance"] for m in m4])
    deb_a = np.array([R4F._is_debatable_nonwest(r["provenance"]) for r in aft_rows], dtype=bool)

    c5_keys, all_keys = list(PD.CONSENSUS_KEYS), list(PD.ALL_KEYS)
    orig_plato = AR.PLATO
    AR.PLATO = PLATO_CACHE
    try:
        rot_m = AR.fit_neutral_or_full_rotations(pool="story", neutral_only=True, d_plato=256)
    finally:
        AR.PLATO = orig_plato
    pairs_arch = AR.load_arch_pairs()
    house_idx = [i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS]
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])

    Zj, Zn, Zf, Za, Q3, uh_all, us_all, uo_all = {}, {}, {}, {}, {}, {}, {}, {}
    for k in all_keys:
        Zj[k] = NA235._map_stories(AR._load_npz_dict(PLATO_CACHE / f"arch235_{k}.npz")["mean"], rot_m, k, frame="to_c5")
        Zn[k] = NA235._map_stories(AR._load_npz_dict(PLATO_CACHE / f"nj237_{k}.npz")["mean"], rot_m, k, frame="to_c5")
        Zf[k] = NA235._map_stories(AR._load_npz_dict(PLATO_CACHE / f"fam239_{k}.npz")["mean"], rot_m, k, frame="to_c5")
        Za[k] = NA235._map_stories(AR._load_npz_dict(PLATO_CACHE / f"aft239_{k}.npz")["mean"], rot_m, k, frame="to_c5")
        a_d = AR._load_npz_dict(PLATO_CACHE / f"arch229_{k}.npz")
        v_d = AR._load_npz_dict(PLATO_CACHE / f"val234_{k}.npz")
        _, _, Dh = AR.map_diffs(a_d["done_mean"][house_idx], a_d["neg_mean"][house_idx], rot_m, k, frame="to_c5")
        _, _, Dv = AR.map_diffs(v_d["pos_mean"], v_d["neg_mean"], rot_m, k, frame="to_c5")
        groups = v_d["groups"]
        uh_all[k] = AR.house_dir(Dh, house_cats)
        us_all[k] = AR.unit(Dv[groups == "sentiment_gen"].mean(0))
        uo_all[k] = AR.unit(Dv[groups == "outcome_valence"].mean(0))
        Q3[k] = V234._orthonormal_basis([us_all[k], uo_all[k], uh_all[k]])

    Q3_c5 = V234._orthonormal_basis([
        AR.unit(np.mean([us_all[k] for k in c5_keys], 0)),
        AR.unit(np.mean([uo_all[k] for k in c5_keys], 0)),
        AR.unit(np.mean([uh_all[k] for k in c5_keys], 0)),
    ])
    Zj_c5 = V234._project_out_rows(np.mean([Zj[k] for k in c5_keys], 0), Q3_c5)
    Zn_c5 = V234._project_out_rows(np.mean([Zn[k] for k in c5_keys], 0), Q3_c5)
    Zf_c5 = V234._project_out_rows(np.mean([Zf[k] for k in c5_keys], 0), Q3_c5)
    Za_c5_all = V234._project_out_rows(np.mean([Za[k] for k in c5_keys], 0), Q3_c5)

    jung_names = list(NA235.ARCHETYPES)
    y_j = np.array([jung_names.index(r["archetype"]) for r in jung_rows], dtype=int)
    sids_j = np.array([int(r["setting_id"]) for r in jung_rows], dtype=int)
    y_n = np.array([NJ237.ROLES.index(r["role"]) for r in jobs_rows], dtype=int)
    sids_n = np.array([int(r["setting_id"]) for r in jobs_rows], dtype=int)
    y_f = np.array([R239.FAM_ROLES.index(r["role"]) for r in fam_rows], dtype=int)
    sids_f = np.array([int(r["setting_id"]) for r in fam_rows], dtype=int)
    uniq_sj = sorted(set(int(s) for s in sids_j))
    sj_ix = [np.where(sids_j == s)[0] for s in uniq_sj]

    # 2a. Joint bootstrap with BOTH percentile AND basic reflected 95% CIs on 480 and 476
    def joint_boot_pct_basic(keep_mask, seed):
        Za_cl, y_cl, atu_cl = Za_c5_all[keep_mask], y_a[keep_mask], atu_a[keep_mask]
        by_dt = {c: [np.array(v, dtype=int) for v in sorted({str(t): np.where((y_cl == c) & (atu_cl == t))[0].tolist() for t in set(atu_cl[y_cl == c].tolist())}.values(), key=lambda x: x[0])] for c in range(10)}
        Qj_o = R239._centroid_basis_k(Zj_c5, y_j, 8, 7)
        Qp_o = A245._pca_basis_k(Zj_c5, 7)
        Qs_o = A245._setting_centered_pca_basis_k(Zj_c5, sids_j, 7)
        vj_o = R239._eval_basis_capture(Qj_o, Za_cl, y_cl, 10)["v_dir"]
        vp_o = R239._eval_basis_capture(Qp_o, Za_cl, y_cl, 10)["v_dir"]
        vs_o = R239._eval_basis_capture(Qs_o, Za_cl, y_cl, 10)["v_dir"]
        rng = np.random.default_rng(seed)
        d_raw, d_sc = [], []
        for _ in range(2000):
            ch_s = rng.integers(0, len(uniq_sj), size=len(uniq_sj))
            ix_j = np.concatenate([sj_ix[c] for c in ch_s])
            sids_b = np.concatenate([np.full(len(sj_ix[c]), b_i, dtype=int) for b_i, c in enumerate(ch_s)])
            Zb, yb = Zj_c5[ix_j], y_j[ix_j]
            Qj_b = R239._centroid_basis_k(Zb, yb, 8, 7)
            Qp_b = A245._pca_basis_k(Zb, 7)
            Qs_b = A245._setting_centered_pca_basis_k(Zb, sids_b, 7)
            ix_t = np.concatenate([by_dt[c][g] for c in range(10) for g in rng.integers(0, len(by_dt[c]), size=len(by_dt[c]))])
            Zt_b, yt_b = Za_cl[ix_t], y_cl[ix_t]
            U_b = NA235._ovr_dirs(Zt_b - Zt_b.mean(0, keepdims=True), yt_b, 10)
            vj_b = float(np.mean(np.sum((U_b @ Qj_b) ** 2, axis=1)))
            d_raw.append(float(np.mean(np.sum((U_b @ Qp_b) ** 2, axis=1))) - vj_b)
            d_sc.append(float(np.mean(np.sum((U_b @ Qs_b) ** 2, axis=1))) - vj_b)
        d_raw, d_sc = np.asarray(d_raw), np.asarray(d_sc)
        return {
            "n_records": int(keep_mask.sum()),
            "PCA7_unlabeled_minus_Q_jung7_joint": _pct_and_basic(vp_o - vj_o, d_raw, 0.05, digits=5),
            "PCA7_setcent_minus_Q_jung7_joint": _pct_and_basic(vs_o - vj_o, d_sc, 0.05, digits=5),
        }

    joint_pct_basic = {"n480": joint_boot_pct_basic(keep480, 24950), "n476": joint_boot_pct_basic(keep476, 24951)}

    # 2b. Complete 476-record structural suite (fixed Holm-7, shuffle, diagonal, PR, exact 8! geometry)
    Za_476, y_476, atu_476, prov_476 = Za_c5_all[keep476], y_a[keep476], atu_a[keep476], prov_a[keep476]
    bases = {
        "Q_jung7_labeled": R239._centroid_basis_k(Zj_c5, y_j, 8, 7),
        "PCA7_jung_unlabeled": A245._pca_basis_k(Zj_c5, 7),
        "PCA7_jung_setcent": A245._setting_centered_pca_basis_k(Zj_c5, sids_j, 7),
        "Q_fam7_labeled": R239._centroid_basis_k(Zf_c5, y_f, 8, 7),
        "PCA7_fam_unlabeled": A245._pca_basis_k(Zf_c5, 7),
        "Q_jobs7_labeled": R239._centroid_basis_k(Zn_c5, y_n, 8, 7),
        "PCA7_jobs_unlabeled": A245._pca_basis_k(Zn_c5, 7),
    }
    base_476 = {bn: round(R239._eval_basis_capture(Q, Za_476, y_476, 10)["v_dir"], 5) for bn, Q in bases.items()}
    cpairs = [
        ("Q_jung7_minus_PCA7_jung_unlabeled", "Q_jung7_labeled", "PCA7_jung_unlabeled", 24851),
        ("Q_jung7_minus_PCA7_jung_setcent", "Q_jung7_labeled", "PCA7_jung_setcent", 24852),
        ("Q_jung7_minus_Q_fam7", "Q_jung7_labeled", "Q_fam7_labeled", 24853),
        ("Q_jung7_minus_Q_jobs7", "Q_jung7_labeled", "Q_jobs7_labeled", 24854),
        ("Q_fam7_minus_Q_jobs7", "Q_fam7_labeled", "Q_jobs7_labeled", 24855),
        ("PCA7_jung_minus_PCA7_jobs", "PCA7_jung_unlabeled", "PCA7_jobs_unlabeled", 24856),
        ("PCA7_fam_minus_PCA7_jobs", "PCA7_fam_unlabeled", "PCA7_jobs_unlabeled", 24857),
    ]
    c476 = {cn: A245._type_clustered_boot_diff(bases[ka], bases[kb], Za_476, y_476, atu_476, 10, n_boot=2000, seed=sd)["v_dir"] for cn, ka, kb, sd in cpairs}
    c476_holm7 = _holm({k: v["p_two_sided"] for k, v in c476.items()})
    U_476 = NA235._ovr_dirs(Za_476 - Za_476.mean(0, keepdims=True), y_476, 10)
    U_j = NA235._ovr_dirs(Zj_c5 - Zj_c5.mean(0, keepdims=True), y_j, 8)
    cos_ja476 = U_j @ U_476.T
    j2i = {a: i for i, a in enumerate(jung_names)}
    vp = [(j2i[ja], div2i[da]) for ja, da in [("Trickster", "01_Animal_Trickster"), ("Trickster", "11_Jokes_Tricksters"), ("Hero", "02_Supernatural_Adversary"), ("Wise_Old_Man", "05_Supernatural_Helpers"), ("Shadow", "10_Stupid_Ogre_Devil")]]
    d476_obs = float(np.mean([cos_ja476[ji, di] for ji, di in vp]))
    rng_d476 = np.random.default_rng(24865)
    d476_null = [float(np.mean([cos_ja476[pj[ji], pd_[di]] for ji, di in vp])) for pj, pd_ in ((rng_d476.permutation(8), rng_d476.permutation(10)) for _ in range(5000))]
    diag_476 = {"mean_cos": round(d476_obs, 4), "p_perm_upper": round(float((np.sum(np.asarray(d476_null) >= d476_obs - 1e-12) + 1) / 5001), 4)}
    pr_476 = A245._pr_type_preserving_null(Za_476, y_476, atu_476, 10, n_perm=4999, seed=24866)
    # Exact 8! geometry on 476
    w476 = (prov_476 == "western")
    nw476 = (prov_476 == "non_western")
    w476_disj = w476 & np.array([t not in set(atu_476[nw476].tolist()) for t in atu_476], dtype=bool)
    ok_xc = [c for c in range(10) if np.sum(y_476[w476_disj] == c) >= 3 and np.sum(y_476[nw476] == c) >= 3]
    def gmat(Zs, ys):
        Zc = Zs - Zs.mean(0, keepdims=True)
        M = np.stack([AR.unit(Zc[ys == c].mean(0)) for c in ok_xc], 0)
        return M @ M.T
    Gw, Gnw = gmat(Za_476[w476_disj], y_476[w476_disj]), gmat(Za_476[nw476], y_476[nw476])
    iu = np.triu_indices(len(ok_xc), 1)
    gw_v = (Gw[iu] - Gw[iu].mean()) / np.linalg.norm(Gw[iu] - Gw[iu].mean())
    obs_r476 = float(np.corrcoef(Gw[iu], Gnw[iu])[0, 1])
    ge476 = 0
    for pm in itertools.permutations(range(len(ok_xc))):
        v = Gnw[np.ix_(pm, pm)][iu]
        v = v - v.mean()
        if float(np.dot(gw_v, v / max(1e-12, float(np.linalg.norm(v))))) >= obs_r476 - 1e-12:
            ge476 += 1
    geom_476 = {"n_divs": len(ok_xc), "obs_r": round(obs_r476, 4), "exact_ge_count": ge476, "total_perms": 40320, "p_exact_one_sided": round(ge476 / 40320, 5)}

    # 2c. Shared-draw 20,000-resample bootstrap on 480 (10-bin, 9-bin eval of 10-class fits, joint 16-cell Holm, AND true 9-class refit!)
    def shared_20k_transfer(keep_mask, exclude_debatable=False, true_9class_refit=False, n_boot=20000, seed=24960):
        y_cl, atu_cl, prov_cl, deb_cl = y_a[keep_mask], atu_a[keep_mask], prov_a[keep_mask], deb_a[keep_mask]
        texts_cl = [aft_rows[i]["text"] for i in np.where(keep_mask)[0]]
        Za_cl = Za_c5_all[keep_mask]
        w_m = (prov_cl == "western")
        nw_m = (prov_cl == "non_western") & (~deb_cl if exclude_debatable else np.ones(len(prov_cl), dtype=bool))
        nw_types = set(atu_cl[nw_m].tolist())
        w_disj = w_m & np.array([t not in nw_types for t in atu_cl], dtype=bool)
        cls10 = sorted(set(y_cl[nw_m].tolist()))
        tpd = {c: len(set(atu_cl[nw_m][y_cl[nw_m] == c].tolist())) for c in cls10}
        cls9 = [c for c in cls10 if tpd[c] >= 2]

        if true_9class_refit:
            # Remove omitted singleton class from BOTH Western training AND Non-Western test!
            w_use = w_disj & np.isin(y_cl, cls9)
            nw_use = nw_m & np.isin(y_cl, cls9)
            # Remap cls9 -> 0..8 cleanly
            c2idx = {c: i for i, c in enumerate(cls9)}
            y_tr = np.array([c2idx[int(c)] for c in y_cl[w_use]], dtype=int)
            y_te = np.array([c2idx[int(c)] for c in y_cl[nw_use]], dtype=int)
            atu_te = atu_cl[nw_use]
            n_cls = len(cls9)
            eval_c = list(range(n_cls))
        else:
            w_use, nw_use = w_disj, nw_m
            y_tr, y_te, atu_te = y_cl[w_use], y_cl[nw_use], atu_cl[nw_use]
            n_cls = 10
            eval_c = cls10

        lex_sc = R239._fit_eval_lexical_fold([texts_cl[i] for i in np.where(w_use)[0]], y_tr, [texts_cl[i] for i in np.where(nw_use)[0]], n_cls)

        # Precompute shared 20,000 bootstrap index draws on clist!
        def make_shared_draws(clist, sd):
            by_dt = {}
            for c in clist:
                bt = {}
                for i in np.where(y_te == c)[0]:
                    bt.setdefault(str(atu_te[i]), []).append(i)
                by_dt[c] = [np.array(v, dtype=int) for v in sorted(bt.values(), key=lambda x: x[0])]
            rng = np.random.default_rng(sd)
            return [np.concatenate([by_dt[c][g] for c in clist for g in rng.integers(0, len(by_dt[c]), size=len(by_dt[c]))]) for _ in range(n_boot)]

        draws_9 = make_shared_draws(eval_c if true_9class_refit else cls9, seed)
        draws_10 = None if true_9class_refit else make_shared_draws(cls10, seed + 1)

        def score_draws(sc, clist, draws_list, restrict_argmax=False):
            b_arr = np.empty(len(draws_list))
            a_arr = np.empty(len(draws_list))
            pr = np.array([clist[j] for j in np.argmax(sc[:, clist], axis=1)]) if restrict_argmax else np.argmax(sc, axis=1)
            for bi, ix in enumerate(draws_list):
                yb = y_te[ix]
                b_arr[bi] = A245._balanced_acc(yb, pr[ix], clist)
                a_arr[bi] = A245._macro_auroc(yb, sc[ix], clist)
            return b_arr, a_arr

        def diff_p_two(arr_n, arr_l):
            d = arr_n - arr_l
            n_d = len(d)
            return min(1.0, 2.0 * min(float((int(np.sum(d <= 0)) + 1) / (n_d + 1)), float((int(np.sum(d >= 0)) + 1) / (n_d + 1))))

        # Precompute lexical draw scores ONCE (shared across all 4 neural models!)
        lex_d10 = {} if true_9class_refit else {ln: score_draws(lex_sc[ln], cls10, draws_10, False) for ln in lex_sc}
        lex_d9_10 = {} if true_9class_refit else {ln: score_draws(lex_sc[ln], cls9, draws_9, False) for ln in lex_sc}
        lex_d9_9 = {ln: score_draws(lex_sc[ln], eval_c if true_9class_refit else cls9, draws_9, not true_9class_refit) for ln in lex_sc}
        if not true_9class_refit:
            for ln in lex_sc:
                assert np.allclose(lex_d9_10[ln][1], lex_d9_9[ln][1]), "Shared draws must give identical lexical AUROC!"

        def perm_chance_9(sc_n, clist, restrict_argmax=False, n_perm=2000, sd=24970):
            m = np.isin(y_te, clist)
            ix_m = np.where(m)[0]
            ym, atm = y_te[m], atu_te[m]
            bt = {}
            for pos_i, t in enumerate(atm):
                bt.setdefault(str(t), []).append(pos_i)
            tg = [np.array(v, dtype=int) for v in sorted(bt.values(), key=lambda x: x[0])]
            tl = np.array([Counter(ym[ix].tolist()).most_common(1)[0][0] for ix in tg], dtype=int)
            sc_m = sc_n[ix_m]
            pr = np.array([clist[j] for j in np.argmax(sc_m[:, clist], axis=1)]) if restrict_argmax else np.argmax(sc_m, axis=1)
            obs_b, obs_a = A245._balanced_acc(ym, pr, clist), A245._macro_auroc(ym, sc_m, clist)
            rng = np.random.default_rng(sd)
            gb = ga = 0
            for _ in range(n_perm):
                pl = rng.permutation(tl)
                yp = np.zeros_like(ym)
                for gi, ix in enumerate(tg):
                    yp[ix] = pl[gi]
                if A245._balanced_acc(yp, pr, clist) >= obs_b - 1e-12: gb += 1
                if A245._macro_auroc(yp, sc_m, clist) >= obs_a - 1e-12: ga += 1
            return (gb + 1) / (n_perm + 1), (ga + 1) / (n_perm + 1)

        out_models = {}
        p10, p9_10w, p9_9w, ch9 = {}, {}, {}, {}
        for hk in ("phi4_mini", "olmo3_7b", "gemma4_e4b", "qwen35_9b"):
            Zt_cl = V234._project_out_rows(Za[hk][keep_mask], Q3[hk])
            src_Z = Za_cl if hk in ("phi4_mini", "olmo3_7b") else V234._project_out_rows(np.mean([Za[k][keep_mask] for k in c5_keys if k != hk], 0), Q3_c5)
            Zs_tr = src_Z[w_use] - src_Z[w_use].mean(0, keepdims=True)
            Zt_te = Zt_cl[nw_use] - Zt_cl[w_use].mean(0, keepdims=True)
            U = np.stack([V234._project_out(u, Q3[hk]) for u in NA235._ovr_dirs(Zs_tr, y_tr, n_cls)], 0)
            sc_n = Zt_te @ U.T

            if true_9class_refit:
                pr9 = np.argmax(sc_n, axis=1)
                out_models[hk] = {
                    "top1": round(float(np.mean(pr9 == y_te)), 4),
                    "balanced_acc": round(A245._balanced_acc(y_te, pr9, eval_c), 4),
                    "macro_auroc": round(A245._macro_auroc(y_te, sc_n, eval_c), 4),
                }
                nb, na = score_draws(sc_n, eval_c, draws_9, False)
                p9_9w[f"{hk}_bal"] = max(diff_p_two(nb, lex_d9_9[ln][0]) for ln in lex_sc)
                p9_9w[f"{hk}_auc"] = max(diff_p_two(na, lex_d9_9[ln][1]) for ln in lex_sc)
                cb, ca = perm_chance_9(sc_n, eval_c, False, sd=24970)
                ch9[f"{hk}_bal"], ch9[f"{hk}_auc"] = cb, ca
            else:
                nb10, na10 = score_draws(sc_n, cls10, draws_10, False)
                nb9_10, na9_10 = score_draws(sc_n, cls9, draws_9, False)
                nb9_9, na9_9 = score_draws(sc_n, cls9, draws_9, True)
                assert np.allclose(na9_10, na9_9), "Shared draws must give exact identical neural AUROC!"
                p10[f"{hk}_bal"] = max(diff_p_two(nb10, lex_d10[ln][0]) for ln in lex_sc)
                p10[f"{hk}_auc"] = max(diff_p_two(na10, lex_d10[ln][1]) for ln in lex_sc)
                p9_10w[f"{hk}_bal"] = max(diff_p_two(nb9_10, lex_d9_10[ln][0]) for ln in lex_sc)
                p9_10w[f"{hk}_auc"] = max(diff_p_two(na9_10, lex_d9_10[ln][1]) for ln in lex_sc)
                p9_9w[f"{hk}_bal"] = max(diff_p_two(nb9_9, lex_d9_9[ln][0]) for ln in lex_sc)
                p9_9w[f"{hk}_auc"] = max(diff_p_two(na9_9, lex_d9_9[ln][1]) for ln in lex_sc)
                assert abs(p9_10w[f"{hk}_auc"] - p9_9w[f"{hk}_auc"]) < 1e-12
                out_models[hk] = {"raw_maxp_9bin_auc_shared": round(p9_9w[f"{hk}_auc"], 5), "raw_maxp_9bin_10w_bal": round(p9_10w[f"{hk}_bal"], 5), "raw_maxp_9bin_9w_bal": round(p9_9w[f"{hk}_bal"], 5)}

        if true_9class_refit:
            te_maj_c, te_maj_n = Counter(y_te.tolist()).most_common(1)[0]
            return {
                "n_train": int(w_use.sum()), "n_test": int(nw_use.sum()),
                "test_majority_oracle": {"hits": f"{te_maj_n}/{len(y_te)}", "rate": round(te_maj_n / len(y_te), 4)},
                "lexical_baselines": {ln: {"top1": round(float(np.mean(np.argmax(sc, 1) == y_te)), 4), "balanced_acc": round(A245._balanced_acc(y_te, np.argmax(sc, 1), eval_c), 4), "macro_auroc": round(A245._macro_auroc(y_te, sc, eval_c), 4)} for ln, sc in lex_sc.items()},
                "neural": out_models,
                "holm8_all3_20k": _holm(p9_9w),
                "holm8_chance": _holm(ch9),
            }
        else:
            m9 = np.isin(y_te, cls9)
            te9_maj_c, te9_maj_n = Counter(y_te[m9].tolist()).most_common(1)[0]
            joint16 = {f"10bin_{k}": v for k, v in p10.items()}
            joint16.update({f"9bin9w_{k}": v for k, v in p9_9w.items()})
            return {
                "n_train": int(w_use.sum()), "n_test_10bin": int(nw_use.sum()), "n_test_9bin": int(np.sum(m9)),
                "test_majority_oracle_9bin": {"hits": f"{te9_maj_n}/{int(np.sum(m9))}", "rate": round(te9_maj_n / int(np.sum(m9)), 4)},
                "n_boot_shared": n_boot,
                "per_model_raw_maxp": out_models,
                "holm8_10bin_20k": _holm(p10),
                "holm8_9bin_10way_20k_shared": _holm(p9_10w),
                "holm8_9bin_9way_20k_shared": _holm(p9_9w),
                "holm16_joint_10bin_and_9bin9way_20k": _holm(joint16),
            }

    print("Running 20,000-draw shared-seed folklore transfer & true 9-class refits ...", flush=True)
    tr480_20k = shared_20k_transfer(keep480, exclude_debatable=False, true_9class_refit=False, n_boot=20000, seed=24960)
    tr480_ex15_20k = shared_20k_transfer(keep480, exclude_debatable=True, true_9class_refit=False, n_boot=20000, seed=24961)
    tr480_true9 = shared_20k_transfer(keep480, exclude_debatable=False, true_9class_refit=True, n_boot=20000, seed=24962)
    tr480_ex15_true9 = shared_20k_transfer(keep480, exclude_debatable=True, true_9class_refit=True, n_boot=20000, seed=24963)

    # Observed vs construction ranges
    r5_f = json.loads((EXP5 / "round5_folklore_results.json").read_text())
    atu_ranges_both = {
        d: {
            "construction_atu_range": ORIGINAL_ATU_CONSTRUCTION_RANGES[d],
            "observed_480_min_max": r5_f["atu_derived_10bin_mapping"][d]["atu_numeric_min_max"],
        }
        for d in div_names
    }

    out = {
        "elapsed_sec": round(time.time() - t0, 1),
        "claim_a_text_switch_sentiment_d50": text_switch_sent,
        "claim_a_xdom_bmneut_all4_holm5": xneut_all4_holm,
        "claim_a_d50_mean_strata_nom95_and_bonf20": strata_bonf20,
        "claim_b_cross_family_bonf8_across_both_pools": depth_bonf8,
        "claim_c_atu_construction_and_observed_ranges": atu_ranges_both,
        "claim_c_joint_boot_pct_and_basic": joint_pct_basic,
        "claim_c_476_complete_structural_suite": {
            "bases_v_dir": base_476, "contrasts_holm7": c476_holm7,
            "five_pair_diagonal": diag_476, "pr_type_preserving": pr_476, "geometry_exact_8fact": geom_476,
        },
        "claim_c_shared_20k_eval_of_10class_480": tr480_20k,
        "claim_c_shared_20k_eval_of_10class_480_excl15": tr480_ex15_20k,
        "claim_c_true_9class_refit_480": tr480_true9,
        "claim_c_true_9class_refit_480_excl15": tr480_ex15_true9,
    }
    out_p = EXP6 / "round6_final_polish.json"
    out_p.write_text(json.dumps(out, indent=2))
    print(f"SAVED {out_p} in {out['elapsed_sec']}s", flush=True)
    print("TEXT-SWITCH SENT:", text_switch_sent)
    print("XNEUT ALL4 HOLM5:", xneut_all4_holm)
    print("STRATA PASS COUNTS:", strata_bonf20["pass_counts_both_pct_and_basic_gt_050"])
    print("DEPTH BONF-8 MEAN:", depth_bonf8["mean"])
    print("JOINT PCT+BASIC:", joint_pct_basic)
    print("476 STRUCTURAL:", out["claim_c_476_complete_structural_suite"])
    print("480 20K 9bin10w:", tr480_20k["holm8_9bin_10way_20k_shared"])
    print("480 20K 9bin9w :", tr480_20k["holm8_9bin_9way_20k_shared"])
    print("480 20K JOINT16:", tr480_20k["holm16_joint_10bin_and_9bin9way_20k"])
    print("480 TRUE 9-CLASS REFIT:", tr480_true9)
    print("480_EX15 TRUE 9-CLASS REFIT:", tr480_ex15_true9)


if __name__ == "__main__":
    main()
