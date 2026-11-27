"""Round-7b Claim C Final Seal Script (100% Matched Division-Stratified Bootstrap + Complete 476 & 480 Serialization).

Computes and serializes:
  1. Hard assertion that all 480 retained records satisfy lo <= atu_num <= hi against study-defined ATU bin boundaries.
  2. Fixed-source Holm-7 contrasts under the EXACT division-stratified type-clustered bootstrap (`A245._type_clustered_boot_diff`,
     seeds 24851..24857, 2,000 draws, zero discarded draws) reporting BOTH percentile-tail AND basic-reflected intervals on 480 AND 476.
  3. 200-base within-setting shuffle null on 480 & 476 (Holm-3 across Jung/Fam/Jobs) + 10,000-draw Jung shuffle on BOTH 480 & 476.
  4. Shared-draw 20,000-resample stratified bootstrap on 476_main & 476_excl15: 10-bin, 9-bin 10-way, 9-bin 9-way,
     joint 16-cell Holm (10-bin + 9-bin/9-way), AND true 9-class refits (with point estimates, lexical baselines, and chance Holm-8).
  5. 2,000-draw type-level chance Holm-8 for 9-bin 10-way argmax across ALL FOUR splits (480, 480_excl15, 476, 476_excl15).
"""
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get("JPL_WORK_DIR", "."))
EXP6 = SCRATCH / "exp250_round6"
EXP4 = SCRATCH / "exp248_round4"
EXP245 = SCRATCH / "exp245_controls"
FOLK = Path(os.environ.get("JPL_FOLK_DIR", "./folklore_local_data"))
REPO_ARCH = SCRIPT_DIR.parents[1] / "bench" / "archetypes"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(REPO_ARCH / "fluent" / "plato")))

for p in (str(REPO_ARCH), str(REPO_ARCH / "fluent"), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import named_arch235 as NA235  # noqa: E402
import nonjung_roles237 as NJ237  # noqa: E402
import real_myth_basis239 as R239  # noqa: E402
import aft_audit245 as A245  # noqa: E402
import round4_folklore_audit as R4F  # noqa: E402
import round6_final_polish as R6  # noqa: E402

FRAMING_SENS_AFT_IDXS = {277, 356, 383, 1390}


def _holm(p_map):
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for r, (k, pv) in enumerate(items):
        run = max(run, min(1.0, (m - r) * pv))
        out[k] = round(run, 5)
    return out


def _fixed_source_div_stratified_pct_and_basic(Qa, Qb, Z, y, atu, n_cls=10, n_boot=2000, seed=24851):
    """Exact division-stratified whole-ATU-type bootstrap matching `A245._type_clustered_boot_diff`
    bit-for-bit, returning BOTH percentile-tail AND basic-reflected CIs & two-sided p-values."""
    va = R239._eval_basis_capture(Qa, Z, y, n_cls)["v_dir"]
    vb = R239._eval_basis_capture(Qb, Z, y, n_cls)["v_dir"]
    obs = va - vb

    div_type_groups = []
    for c in range(n_cls):
        ix_c = np.where(y == c)[0]
        by_t = {}
        for i in ix_c:
            by_t.setdefault(str(atu[i]), []).append(i)
        div_type_groups.append([np.array(v, dtype=int) for v in by_t.values()])

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx_parts = []
        for groups in div_type_groups:
            ch = rng.integers(0, len(groups), size=len(groups))
            idx_parts.extend([groups[g] for g in ch])
        idx = np.concatenate(idx_parts)
        Zb, yb = Z[idx], y[idx]
        Ub = NA235._ovr_dirs(Zb - Zb.mean(0, keepdims=True), yb, n_cls)
        diffs[b] = float(np.mean(np.sum((Ub @ Qa) ** 2, axis=1)) - np.mean(np.sum((Ub @ Qb) ** 2, axis=1)))

    p_pct = min(1.0, 2.0 * min(float((int(np.sum(diffs <= 0)) + 1) / (n_boot + 1)), float((int(np.sum(diffs >= 0)) + 1) / (n_boot + 1))))
    basic_arr = 2.0 * obs - diffs
    p_basic = min(1.0, 2.0 * min(float((int(np.sum(basic_arr <= 0)) + 1) / (n_boot + 1)), float((int(np.sum(basic_arr >= 0)) + 1) / (n_boot + 1))))
    return {
        "n_draws": n_boot,
        "obs_diff": round(obs, 5),
        "ci95_percentile": [round(float(np.percentile(diffs, 2.5)), 5), round(float(np.percentile(diffs, 97.5)), 5)],
        "ci95_basic": [round(float(np.percentile(basic_arr, 2.5)), 5), round(float(np.percentile(basic_arr, 97.5)), 5)],
        "p_two_pct": round(p_pct, 5),
        "p_two_basic": round(p_basic, 5),
        "_p_pct_fp": p_pct,
        "_p_basic_fp": p_basic,
    }


def main():
    t0 = time.time()
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

    # 1. Hard-assert every retained 480 row falls inside study-defined ATU bin boundaries!
    for i in np.where(keep480)[0]:
        d = aft_rows[i]["division"]
        lo, hi = R6.ORIGINAL_ATU_CONSTRUCTION_RANGES[d]
        num = int(re.match(r"(\d+)", str(aft_rows[i]["atu_id"])).group(1))
        assert lo <= num <= hi, f"Row {i} aft_idx={aidx_a[i]} ATU {num} outside [{lo}, {hi}] for {d}"
    range_assert_ok = True

    # Load frozen C5 frame
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

    bases = {
        "Q_jung7_labeled": R239._centroid_basis_k(Zj_c5, y_j, 8, 7),
        "PCA7_jung_unlabeled": A245._pca_basis_k(Zj_c5, 7),
        "PCA7_jung_setcent": A245._setting_centered_pca_basis_k(Zj_c5, sids_j, 7),
        "Q_fam7_labeled": R239._centroid_basis_k(Zf_c5, y_f, 8, 7),
        "PCA7_fam_unlabeled": A245._pca_basis_k(Zf_c5, 7),
        "Q_jobs7_labeled": R239._centroid_basis_k(Zn_c5, y_n, 8, 7),
        "PCA7_jobs_unlabeled": A245._pca_basis_k(Zn_c5, 7),
    }
    cpairs = [
        ("Q_jung7_minus_PCA7_jung_unlabeled", "Q_jung7_labeled", "PCA7_jung_unlabeled", 24851),
        ("Q_jung7_minus_PCA7_jung_setcent", "Q_jung7_labeled", "PCA7_jung_setcent", 24852),
        ("Q_jung7_minus_Q_fam7", "Q_jung7_labeled", "Q_fam7_labeled", 24853),
        ("Q_jung7_minus_Q_jobs7", "Q_jung7_labeled", "Q_jobs7_labeled", 24854),
        ("Q_fam7_minus_Q_jobs7", "Q_fam7_labeled", "Q_jobs7_labeled", 24855),
        ("PCA7_jung_minus_PCA7_jobs", "PCA7_jung_unlabeled", "PCA7_jobs_unlabeled", 24856),
        ("PCA7_fam_minus_PCA7_jobs", "PCA7_fam_unlabeled", "PCA7_jobs_unlabeled", 24857),
    ]

    # 2. Exact division-stratified fixed-source Holm-7 under BOTH percentile-tail AND basic-reflected on 480 & 476!
    fixed_both = {}
    for label, km in (("n480", keep480), ("n476", keep476)):
        det = {cn: _fixed_source_div_stratified_pct_and_basic(bases[ka], bases[kb], Za_c5_all[km], y_a[km], atu_a[km], 10, 2000, sd) for cn, ka, kb, sd in cpairs}
        hp = _holm({k: v.pop("_p_pct_fp") for k, v in det.items()})
        hb = _holm({k: v.pop("_p_basic_fp") for k, v in det.items()})
        fixed_both[label] = {"holm7_percentile": hp, "holm7_basic": hb, "contrasts": det}

    # 3. 200-base shuffle null on 480 & 476 (Holm-3 across Jung/Fam/Jobs) + 10,000-draw Jung shuffle on BOTH 480 & 476!
    def make_shuf_qs(Z_s, y_s, sids_s, seed_s, m=200):
        r_ = np.random.default_rng(seed_s)
        qs = []
        for _ in range(m):
            yp = y_s.copy()
            for s in set(int(x) for x in sids_s):
                ix = np.where(sids_s == s)[0]
                yp[ix] = r_.permutation(yp[ix])
            qs.append(R239._centroid_basis_k(Z_s, yp, 8, 7))
        return qs

    qs_j200 = make_shuf_qs(Zj_c5, y_j, sids_j, 24861, 200)
    qs_f200 = make_shuf_qs(Zf_c5, y_f, sids_f, 24862, 200)
    qs_n200 = make_shuf_qs(Zn_c5, y_n, sids_n, 24863, 200)

    def eval_shuf_set(km):
        Zc = Za_c5_all[km]
        U_c = NA235._ovr_dirs(Zc - Zc.mean(0, keepdims=True), y_a[km], 10)
        res, pmap = {}, {}
        for name, Q_o, qs in (("jung", bases["Q_jung7_labeled"], qs_j200), ("fam", bases["Q_fam7_labeled"], qs_f200), ("jobs", bases["Q_jobs7_labeled"], qs_n200)):
            vo = float(np.mean(np.sum((U_c @ Q_o) ** 2, axis=1)))
            vn = np.array([float(np.mean(np.sum((U_c @ q) ** 2, axis=1))) for q in qs])
            pv = float((int(np.sum(vn >= vo - 1e-12)) + 1) / (len(qs) + 1))
            res[name] = {"v_dir": round(vo, 5), "null_200_mean": round(float(vn.mean()), 5), "p_nominal": round(pv, 5)}
            pmap[name] = pv
        return {"per_corpus": res, "holm3": _holm(pmap)}

    shuf_480 = eval_shuf_set(keep480)
    shuf_476 = eval_shuf_set(keep476)

    U_480 = NA235._ovr_dirs(Za_c5_all[keep480] - Za_c5_all[keep480].mean(0, keepdims=True), y_a[keep480], 10)
    U_476 = NA235._ovr_dirs(Za_c5_all[keep476] - Za_c5_all[keep476].mean(0, keepdims=True), y_a[keep476], 10)
    vo_j480 = float(np.mean(np.sum((U_480 @ bases["Q_jung7_labeled"]) ** 2, axis=1)))
    vo_j476 = float(np.mean(np.sum((U_476 @ bases["Q_jung7_labeled"]) ** 2, axis=1)))
    rng_10k = np.random.default_rng(24980)
    ge_480 = ge_476 = 0
    s_groups_j = [np.where(sids_j == s)[0] for s in sorted(set(int(x) for x in sids_j))]
    for _ in range(10000):
        yp = y_j.copy()
        for ix in s_groups_j:
            yp[ix] = rng_10k.permutation(yp[ix])
        q_b = R239._centroid_basis_k(Zj_c5, yp, 8, 7)
        if float(np.mean(np.sum((U_480 @ q_b) ** 2, axis=1))) >= vo_j480 - 1e-12:
            ge_480 += 1
        if float(np.mean(np.sum((U_476 @ q_b) ** 2, axis=1))) >= vo_j476 - 1e-12:
            ge_476 += 1
    jung_shuf_10k = {
        "n480": {"n_draws": 10000, "ge_count": ge_480, "p_upper": round((ge_480 + 1) / 10001, 5), "holm3_with_200base_fam_jobs": round(min(1.0, 3.0 * (ge_480 + 1) / 10001), 5)},
        "n476": {"n_draws": 10000, "ge_count": ge_476, "p_upper": round((ge_476 + 1) / 10001, 5), "holm3_with_200base_fam_jobs": round(min(1.0, 3.0 * (ge_476 + 1) / 10001), 5)},
    }

    # 4. Shared 20,000-draw bootstrap on 476_main & 476_excl15 (incl. joint Holm-16 + true 9-class point/lex/chance!)
    def shared_20k_eval(keep_mask, exclude_debatable=False, true_9class_refit=False, n_boot=20000, seed=24964):
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
            w_use, nw_use = w_disj & np.isin(y_cl, cls9), nw_m & np.isin(y_cl, cls9)
            c2i = {c: i for i, c in enumerate(cls9)}
            y_tr = np.array([c2i[int(c)] for c in y_cl[w_use]], dtype=int)
            y_te = np.array([c2i[int(c)] for c in y_cl[nw_use]], dtype=int)
            atu_te = atu_cl[nw_use]
            n_cls, eval_c = len(cls9), list(range(len(cls9)))
        else:
            w_use, nw_use = w_disj, nw_m
            y_tr, y_te, atu_te = y_cl[w_use], y_cl[nw_use], atu_cl[nw_use]
            n_cls, eval_c = 10, cls10

        lex_sc = R239._fit_eval_lexical_fold([texts_cl[i] for i in np.where(w_use)[0]], y_tr, [texts_cl[i] for i in np.where(nw_use)[0]], n_cls)

        def make_draws(clist, sd):
            by_dt = {c: [np.array(v, dtype=int) for v in sorted({str(t): np.where((y_te == c) & (atu_te == t))[0].tolist() for t in set(atu_te[y_te == c].tolist())}.values(), key=lambda x: x[0])] for c in clist}
            rng = np.random.default_rng(sd)
            return [np.concatenate([by_dt[c][g] for c in clist for g in rng.integers(0, len(by_dt[c]), size=len(by_dt[c]))]) for _ in range(n_boot)]

        draws_9 = make_draws(eval_c if true_9class_refit else cls9, seed)
        draws_10 = None if true_9class_refit else make_draws(cls10, seed + 1)

        def score_draws(sc, clist, draws_list, restrict_argmax=False):
            b_arr, a_arr = np.empty(len(draws_list)), np.empty(len(draws_list))
            pr = np.array([clist[j] for j in np.argmax(sc[:, clist], axis=1)]) if restrict_argmax else np.argmax(sc, axis=1)
            for bi, ix in enumerate(draws_list):
                yb = y_te[ix]
                b_arr[bi] = A245._balanced_acc(yb, pr[ix], clist)
                a_arr[bi] = A245._macro_auroc(yb, sc[ix], clist)
            return b_arr, a_arr

        def diff_p_two(an, al):
            d = an - al
            nd = len(d)
            return min(1.0, 2.0 * min(float((int(np.sum(d <= 0)) + 1) / (nd + 1)), float((int(np.sum(d >= 0)) + 1) / (nd + 1))))

        lex_d10 = {} if true_9class_refit else {ln: score_draws(lex_sc[ln], cls10, draws_10, False) for ln in lex_sc}
        lex_d9_10 = {} if true_9class_refit else {ln: score_draws(lex_sc[ln], cls9, draws_9, False) for ln in lex_sc}
        lex_d9_9 = {ln: score_draws(lex_sc[ln], eval_c if true_9class_refit else cls9, draws_9, not true_9class_refit) for ln in lex_sc}

        p10, p9_10w, p9_9w, pts_t9, ch_t9 = {}, {}, {}, {}, {}
        lex_pts_t9 = {}
        if true_9class_refit:
            for ln, lsc in lex_sc.items():
                pr_l = np.argmax(lsc, axis=1)
                lex_pts_t9[ln] = {
                    "top1": round(float(np.mean(pr_l == y_te)), 4),
                    "bal_acc": round(float(A245._balanced_acc(y_te, pr_l, eval_c)), 4),
                    "macro_auroc": round(float(A245._macro_auroc(y_te, lsc, eval_c)), 4),
                }
            # Precompute type groups for 2,000-draw chance test
            bt = {}
            for pos_i, t in enumerate(atu_te):
                bt.setdefault(str(t), []).append(pos_i)
            tg = [np.array(v, dtype=int) for v in sorted(bt.values(), key=lambda x: x[0])]
            tl = np.array([Counter(y_te[ix].tolist()).most_common(1)[0][0] for ix in tg], dtype=int)

        for hk in ("phi4_mini", "olmo3_7b", "gemma4_e4b", "qwen35_9b"):
            Zt_cl = V234._project_out_rows(Za[hk][keep_mask], Q3[hk])
            src_Z = Za_cl if hk in ("phi4_mini", "olmo3_7b") else V234._project_out_rows(np.mean([Za[k][keep_mask] for k in c5_keys if k != hk], 0), Q3_c5)
            Zs_tr = src_Z[w_use] - src_Z[w_use].mean(0, keepdims=True)
            Zt_te = Zt_cl[nw_use] - Zt_cl[w_use].mean(0, keepdims=True)
            U = np.stack([V234._project_out(u, Q3[hk]) for u in NA235._ovr_dirs(Zs_tr, y_tr, n_cls)], 0)
            sc_n = Zt_te @ U.T
            if true_9class_refit:
                pr_n = np.argmax(sc_n, axis=1)
                obs_b = float(A245._balanced_acc(y_te, pr_n, eval_c))
                obs_a = float(A245._macro_auroc(y_te, sc_n, eval_c))
                pts_t9[hk] = {"top1": round(float(np.mean(pr_n == y_te)), 4), "bal_acc": round(obs_b, 4), "macro_auroc": round(obs_a, 4)}
                nb, na = score_draws(sc_n, eval_c, draws_9, False)
                p9_9w[f"{hk}_bal"] = max(diff_p_two(nb, lex_d9_9[ln][0]) for ln in lex_sc)
                p9_9w[f"{hk}_auc"] = max(diff_p_two(na, lex_d9_9[ln][1]) for ln in lex_sc)
                rng_c = np.random.default_rng(24970)
                gb = ga = 0
                for _ in range(2000):
                    pl = rng_c.permutation(tl)
                    yp = np.zeros_like(y_te)
                    for gi, ix in enumerate(tg):
                        yp[ix] = pl[gi]
                    if A245._balanced_acc(yp, pr_n, eval_c) >= obs_b - 1e-12: gb += 1
                    if A245._macro_auroc(yp, sc_n, eval_c) >= obs_a - 1e-12: ga += 1
                ch_t9[f"{hk}_bal"], ch_t9[f"{hk}_auc"] = (gb + 1) / 2001, (ga + 1) / 2001
            else:
                nb10, na10 = score_draws(sc_n, cls10, draws_10, False)
                nb9_10, na9_10 = score_draws(sc_n, cls9, draws_9, False)
                nb9_9, na9_9 = score_draws(sc_n, cls9, draws_9, True)
                p10[f"{hk}_bal"] = max(diff_p_two(nb10, lex_d10[ln][0]) for ln in lex_sc)
                p10[f"{hk}_auc"] = max(diff_p_two(na10, lex_d10[ln][1]) for ln in lex_sc)
                p9_10w[f"{hk}_bal"] = max(diff_p_two(nb9_10, lex_d9_10[ln][0]) for ln in lex_sc)
                p9_10w[f"{hk}_auc"] = max(diff_p_two(na9_10, lex_d9_10[ln][1]) for ln in lex_sc)
                p9_9w[f"{hk}_bal"] = max(diff_p_two(nb9_9, lex_d9_9[ln][0]) for ln in lex_sc)
                p9_9w[f"{hk}_auc"] = max(diff_p_two(na9_9, lex_d9_9[ln][1]) for ln in lex_sc)

        if true_9class_refit:
            return {
                "n_train": int(w_use.sum()), "n_test": int(nw_use.sum()),
                "lexical_baselines": lex_pts_t9,
                "neural_point_estimates": pts_t9,
                "holm8_all3_20k": _holm(p9_9w),
                "holm8_chance_2k": _holm(ch_t9),
            }
        joint16 = {**{f"10bin_{k}": v for k, v in p10.items()}, **{f"9bin9w_{k}": v for k, v in p9_9w.items()}}
        return {
            "n_train": int(w_use.sum()), "n_test_10bin": int(nw_use.sum()),
            "holm8_10bin_20k": _holm(p10),
            "holm8_9bin_10way_20k_shared": _holm(p9_10w),
            "holm8_9bin_9way_20k_shared": _holm(p9_9w),
            "holm16_joint_10bin_plus_9bin9w_20k": _holm(joint16),
        }

    # 5. 2,000-draw type-level chance Holm-8 for 9-bin 10-way argmax across ALL FOUR splits (480, 480_excl15, 476, 476_excl15)
    def chance_9bin_10way(keep_mask, exclude_debatable=False):
        y_cl, atu_cl, prov_cl, deb_cl = y_a[keep_mask], atu_a[keep_mask], prov_a[keep_mask], deb_a[keep_mask]
        Za_cl = Za_c5_all[keep_mask]
        w_m = (prov_cl == "western")
        nw_m = (prov_cl == "non_western") & (~deb_cl if exclude_debatable else np.ones(len(prov_cl), dtype=bool))
        nw_types = set(atu_cl[nw_m].tolist())
        w_disj = w_m & np.array([t not in nw_types for t in atu_cl], dtype=bool)
        y_tr, y_te, atu_te = y_cl[w_disj], y_cl[nw_m], atu_cl[nw_m]
        cls10 = sorted(set(y_te.tolist()))
        cls9 = [c for c in cls10 if len(set(atu_te[y_te == c].tolist())) >= 2]
        m = np.isin(y_te, cls9)
        ym, atm = y_te[m], atu_te[m]
        bt = {}
        for pos_i, t in enumerate(atm):
            bt.setdefault(str(t), []).append(pos_i)
        tg = [np.array(v, dtype=int) for v in sorted(bt.values(), key=lambda x: x[0])]
        tl = np.array([Counter(ym[ix].tolist()).most_common(1)[0][0] for ix in tg], dtype=int)
        ch = {}
        for hk in ("phi4_mini", "olmo3_7b", "gemma4_e4b", "qwen35_9b"):
            Zt_cl = V234._project_out_rows(Za[hk][keep_mask], Q3[hk])
            src_Z = Za_cl if hk in ("phi4_mini", "olmo3_7b") else V234._project_out_rows(np.mean([Za[k][keep_mask] for k in c5_keys if k != hk], 0), Q3_c5)
            Zs_tr = src_Z[w_disj] - src_Z[w_disj].mean(0, keepdims=True)
            Zt_te = Zt_cl[nw_m] - Zt_cl[w_disj].mean(0, keepdims=True)
            U = np.stack([V234._project_out(u, Q3[hk]) for u in NA235._ovr_dirs(Zs_tr, y_tr, 10)], 0)
            sc_m = (Zt_te @ U.T)[m]
            pr = np.argmax(sc_m, axis=1)
            obs_b, obs_a = A245._balanced_acc(ym, pr, cls9), A245._macro_auroc(ym, sc_m, cls9)
            rng = np.random.default_rng(24876)
            gb = ga = 0
            for _ in range(2000):
                pl = rng.permutation(tl)
                yp = np.zeros_like(ym)
                for gi, ix in enumerate(tg):
                    yp[ix] = pl[gi]
                if A245._balanced_acc(yp, pr, cls9) >= obs_b - 1e-12: gb += 1
                if A245._macro_auroc(yp, sc_m, cls9) >= obs_a - 1e-12: ga += 1
            ch[f"{hk}_bal"], ch[f"{hk}_auc"] = (gb + 1) / 2001, (ga + 1) / 2001
        return _holm(ch)

    print("Running 20,000-draw shared-seed 476 transfer & 10-way chance checks ...", flush=True)
    tr476_20k = shared_20k_eval(keep476, exclude_debatable=False, true_9class_refit=False, n_boot=20000, seed=24964)
    tr476_ex15_20k = shared_20k_eval(keep476, exclude_debatable=True, true_9class_refit=False, n_boot=20000, seed=24965)
    tr476_true9 = shared_20k_eval(keep476, exclude_debatable=False, true_9class_refit=True, n_boot=20000, seed=24966)
    tr476_ex15_true9 = shared_20k_eval(keep476, exclude_debatable=True, true_9class_refit=True, n_boot=20000, seed=24967)
    ch_all = {
        "n480": chance_9bin_10way(keep480, False),
        "n480_excl15": chance_9bin_10way(keep480, True),
        "n476": chance_9bin_10way(keep476, False),
        "n476_excl15": chance_9bin_10way(keep476, True),
    }

    out = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_sec": round(time.time() - t0, 1),
        "atu_range_assertion_480_passed": range_assert_ok,
        "fixed_source_div_stratified_holm7_pct_and_basic": fixed_both,
        "shuffle_200base_with_holm3": {"n480": shuf_480, "n476": shuf_476},
        "jung_shuffle_10k": jung_shuf_10k,
        "chance_holm8_9bin_10way": ch_all,
        "transfer_476_20k_shared": tr476_20k,
        "transfer_476_excl15_20k_shared": tr476_ex15_20k,
        "transfer_476_true9_20k": tr476_true9,
        "transfer_476_excl15_true9_20k": tr476_ex15_true9,
    }
    out_p = REPO_ARCH / "fluent" / "plato" / "v6_round7_claim_c_seal.json"
    out_p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"SAVED {out_p} in {out['elapsed_sec']}s", flush=True)
    print("DIV-STRAT FIXED HOLM7 480 pct | basic:", fixed_both["n480"]["holm7_percentile"], "|", fixed_both["n480"]["holm7_basic"])
    print("DIV-STRAT FIXED HOLM7 476 pct | basic:", fixed_both["n476"]["holm7_percentile"], "|", fixed_both["n476"]["holm7_basic"])
    print("JUNG 10K:", jung_shuf_10k)
    print("476 JOINT16:", {k: v for k, v in tr476_20k["holm16_joint_10bin_plus_9bin9w_20k"].items() if v < 0.1})
    print("476_EX15 JOINT16:", {k: v for k, v in tr476_ex15_20k["holm16_joint_10bin_plus_9bin9w_20k"].items() if v < 0.1})
    print("CHANCE 10WAY MAX:", {k: max(v.values()) for k, v in ch_all.items()})


if __name__ == "__main__":
    main()
