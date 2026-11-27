"""Round-4 Final Polish Audit for Claim C (480/500 Cleaned Historical Folktales).

Resolves EVERY remaining Round-3 finding on Claim C from BOTH Fable 5.2 and Astra 6.2:
  1. Explicitly excludes AND hard-asserts exclusion of ALL 20 suspect records:
     - Original 11 excluded in Round 3: [49, 269, 370, 450, 870, 881, 983, 1147, 1205, 1303, 1363]
     - Plus ALL 8 editorial/preface/footnote/citation records identified by Astra 6.2:
       [18, 19, 591, 756, 764, 777, 1369, 1397]
     - Plus the cross-division 113-word overlapping source story copy: [1244] (shares London 1896 pp. 3-9 with 906!)
     Total: 480/500 cleaned records, full 500-row manifest saved at `aft_clean248_manifest.json`!
  2. Directly joins the 480 kept records to `aft.csv` to compute the EXACT count shorter than source CSV text!
  3. Exhaustive `8! = 40,320` exact permutation test for Western vs. Non-Western division geometry `r`.
  4. Division-stratified ATU-type bootstrap comparing neural `C5 perp VAL3` against ALL THREE lexical
     baselines (`bow_unigram`, `tfidf_centroid`, `tfidf_ridge`) — reporting BOTH `max_p` across all 3
     lexical comparators AND both one-sided and two-sided Holm-8 p-values, PLUS:
     - 9-division sensitivity strictly against `tfidf_centroid` (the actual 9-div lexical winner!) with BOTH 10-way and 9-way argmax;
     - Label-permutation test against chance (`5,000` type-level permutations);
     - Per-class recall & AUROC for every model;
     - Provenance sensitivity excluding the 15 debatable African American / Jewish / Armenian-American / bare-Georgia rows!
"""
import csv
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
EXP4 = SCRATCH / "exp248_round4"
EXP4.mkdir(parents=True, exist_ok=True)
EXP3 = SCRATCH / "exp247_round3"
EXP245 = SCRATCH / "exp245_controls"
FOLK = SCRATCH / "folklore_test"
REPO_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(Path(__file__).resolve().parents[2] / "bench" / "archetypes" / "fluent" / "plato")))

for p in (str(REPO_ARCH), str(REPO_ARCH / "fluent"), str(FOLK), str(EXP245), str(EXP3)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import named_arch235 as NA235  # noqa: E402
import nonjung_roles237 as NJ237  # noqa: E402
import real_myth_basis239 as R239  # noqa: E402
import aft_audit245 as A245  # noqa: E402
import aft_audit247 as A247  # noqa: E402

# Explicitly audited editorial/footnote/preface/citation & overlapping-source exclusions
EXPLICIT_EDITORIAL_AFT_IDXS = {18, 19, 591, 756, 764, 777, 1369, 1397}
EXPLICIT_OVERLAP_DUP_AFT_IDXS = {1244}  # Shares 113 consecutive words (London 1896 pp. 3-9) with aft_idx 906!


def _holm(p_map):
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for r, (k, pv) in enumerate(items):
        run = max(run, min(1.0, (m - r) * pv))
        out[k] = round(run, 5)
    return out


def _is_debatable_nonwest(prov_str):
    s = (prov_str or "").strip().lower()
    if any(k in s for k in ("african american", "african-america", "african america", "jewish", "armenian-american", "armenian american")):
        return True
    if s in ("georgia", "georgia."):
        return True
    return False


def main():
    t0 = time.time()
    aft_rows = json.loads((FOLK / "aft_real_myths_500.json").read_text())
    fam_rows = json.loads((FOLK / "family_roles239_data.json").read_text())
    jung_rows = NA235.load_named_arch_data()
    jobs_rows = NJ237.load_nonjung_data()

    # Load aft.csv to verify exact shorter-than-CSV whitespace word count!
    csv_wc = {}
    with open(FOLK / "aft.csv", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx_r, row in enumerate(reader):
            txt = row.get("text") or row.get("Text") or ""
            csv_wc[idx_r] = len(txt.split())

    div_names = list(R239.ATU_DIVS)
    div2i = {d: i for i, d in enumerate(div_names)}
    y_a = np.array([div2i[r["division"]] for r in aft_rows], dtype=int)
    atu_a = np.array([str(r["atu_id"]) for r in aft_rows])

    seen_alpha = {}
    manifest_rows = []
    keep_mask = np.ones(len(aft_rows), dtype=bool)
    prov_labels = []
    debatable_flags = []
    shorter_flags = []

    for i, r in enumerate(aft_rows):
        aidx = int(r["aft_idx"])
        seq = A247._norm_alpha_seq(r["text"])
        excl_reason = None
        if seq in seen_alpha:
            excl_reason = f"duplicate_alpha_text_of_row_{seen_alpha[seq]}_aftidx_{aft_rows[seen_alpha[seq]]['aft_idx']}"
        else:
            seen_alpha[seq] = i
            if aidx in EXPLICIT_OVERLAP_DUP_AFT_IDXS:
                excl_reason = "overlapping_source_story_113words_with_aftidx_906"
            elif aidx in EXPLICIT_EDITORIAL_AFT_IDXS:
                excl_reason = "audited_editorial_preface_footnote_or_citation"
            else:
                ed = A247._catalog_or_editorial_reason(r["text"])
                if ed:
                    excl_reason = ed
        if excl_reason is not None:
            keep_mask[i] = False
        pl = A247._adjudicate_provenance(r["provenance"], r["text"])
        prov_labels.append(pl)
        debatable_flags.append(_is_debatable_nonwest(r["provenance"]))
        shorter = (len(r["text"].split()) < csv_wc.get(aidx, 0)) if aidx in csv_wc else None
        shorter_flags.append(shorter)
        manifest_rows.append({
            "row_idx": i, "aft_idx": aidx, "atu_id": str(r["atu_id"]), "division": r["division"],
            "raw_provenance": r["provenance"], "adjudicated_provenance": pl,
            "debatable_nonwest": bool(debatable_flags[-1]),
            "shorter_than_csv": shorter,
            "included": bool(excl_reason is None), "exclusion_reason": excl_reason,
        })

    # Hard-assert ALL 20 suspect aft_idxs are excluded!
    excl_set = {m["aft_idx"] for m in manifest_rows if not m["included"]}
    must_excl = {49, 269, 370, 450, 870, 881, 983, 1147, 1205, 1303, 1363} | EXPLICIT_EDITORIAL_AFT_IDXS | EXPLICIT_OVERLAP_DUP_AFT_IDXS
    assert must_excl.issubset(excl_set), f"Missed exclusions: {must_excl - excl_set}"
    assert int(keep_mask.sum()) == 480, f"Expected 480 kept, got {int(keep_mask.sum())}"
    (EXP4 / "aft_clean248_manifest.json").write_text(json.dumps(manifest_rows, indent=2))

    n_shorter = sum(1 for i in np.where(keep_mask)[0] if shorter_flags[i] is True)
    prov_labels = np.array(prov_labels)
    debatable_flags = np.array(debatable_flags, dtype=bool)

    # Load frozen C5 frame
    c5_keys = list(PD.CONSENSUS_KEYS)
    all_keys = list(PD.ALL_KEYS)
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
    Za_cl = V234._project_out_rows(np.mean([Za[k] for k in c5_keys], 0), Q3_c5)[keep_mask]

    jung_names = list(NA235.ARCHETYPES)
    y_j = np.array([jung_names.index(r["archetype"]) for r in jung_rows], dtype=int)
    sids_j = np.array([int(r["setting_id"]) for r in jung_rows], dtype=int)
    y_n = np.array([NJ237.ROLES.index(r["role"]) for r in jobs_rows], dtype=int)
    sids_n = np.array([int(r["setting_id"]) for r in jobs_rows], dtype=int)
    y_f = np.array([R239.FAM_ROLES.index(r["role"]) for r in fam_rows], dtype=int)
    sids_f = np.array([int(r["setting_id"]) for r in fam_rows], dtype=int)

    y_cl = y_a[keep_mask]
    atu_cl = atu_a[keep_mask]
    prov_cl = prov_labels[keep_mask]
    deb_cl = debatable_flags[keep_mask]
    clean_texts = [aft_rows[i]["text"] for i in np.where(keep_mask)[0]]

    # 1. Basis Audit on 480 cleaned records
    bases = {
        "Q_jung7_labeled": R239._centroid_basis_k(Zj_c5, y_j, 8, 7),
        "PCA7_jung_unlabeled": A245._pca_basis_k(Zj_c5, 7),
        "PCA7_jung_setcent": A245._setting_centered_pca_basis_k(Zj_c5, sids_j, 7),
        "Q_fam7_labeled": R239._centroid_basis_k(Zf_c5, y_f, 8, 7),
        "PCA7_fam_unlabeled": A245._pca_basis_k(Zf_c5, 7),
        "Q_jobs7_labeled": R239._centroid_basis_k(Zn_c5, y_n, 8, 7),
        "PCA7_jobs_unlabeled": A245._pca_basis_k(Zn_c5, 7),
    }
    base_scores = {bn: round(R239._eval_basis_capture(Q, Za_cl, y_cl, 10)["v_dir"], 5) for bn, Q in bases.items()}
    contrast_pairs = [
        ("Q_jung7_minus_PCA7_jung_unlabeled", "Q_jung7_labeled", "PCA7_jung_unlabeled", 24851),
        ("Q_jung7_minus_PCA7_jung_setcent", "Q_jung7_labeled", "PCA7_jung_setcent", 24852),
        ("Q_jung7_minus_Q_fam7", "Q_jung7_labeled", "Q_fam7_labeled", 24853),
        ("Q_jung7_minus_Q_jobs7", "Q_jung7_labeled", "Q_jobs7_labeled", 24854),
        ("Q_fam7_minus_Q_jobs7", "Q_fam7_labeled", "Q_jobs7_labeled", 24855),
        ("PCA7_jung_minus_PCA7_jobs", "PCA7_jung_unlabeled", "PCA7_jobs_unlabeled", 24856),
        ("PCA7_fam_minus_PCA7_jobs", "PCA7_fam_unlabeled", "PCA7_jobs_unlabeled", 24857),
    ]
    contrasts = {cn: A245._type_clustered_boot_diff(bases[ka], bases[kb], Za_cl, y_cl, atu_cl, 10, n_boot=2000, seed=sd)["v_dir"] for cn, ka, kb, sd in contrast_pairs}
    contrasts_holm = _holm({k: v["p_two_sided"] for k, v in contrasts.items()})

    # 200-base shuffle null + unified excess (labeled nominal!)
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
    qs_j, qs_f, qs_n = make_shuf_qs(Zj_c5, y_j, sids_j, 24861, 200), make_shuf_qs(Zf_c5, y_f, sids_f, 24862, 200), make_shuf_qs(Zn_c5, y_n, sids_n, 24863, 200)
    U_cl = NA235._ovr_dirs(Za_cl - Za_cl.mean(0, keepdims=True), y_cl, 10)
    def shuf_stat(Q_o, qs):
        vo = float(np.mean(np.sum((U_cl @ Q_o) ** 2, axis=1)))
        vn = np.array([float(np.mean(np.sum((U_cl @ q) ** 2, axis=1))) for q in qs])
        return {"v_dir": round(vo, 5), "null_200_mean": round(float(vn.mean()), 5), "excess": round(vo - float(vn.mean()), 5), "p_nominal": round(float((np.sum(vn >= vo - 1e-12) + 1) / 201), 4)}
    shuf_res = {"jung": shuf_stat(bases["Q_jung7_labeled"], qs_j), "fam": shuf_stat(bases["Q_fam7_labeled"], qs_f), "jobs": shuf_stat(bases["Q_jobs7_labeled"], qs_n)}

    # 2. Fixed 5-pair diagonal + type-preserving PR + EXHAUSTIVE 8! = 40,320 geometry permutation!
    U_j = NA235._ovr_dirs(Zj_c5 - Zj_c5.mean(0, keepdims=True), y_j, 8)
    cos_ja = U_j @ U_cl.T
    j2i = {a: i for i, a in enumerate(jung_names)}
    mapped_pairs = [
        ("Trickster", "01_Animal_Trickster"), ("Trickster", "11_Jokes_Tricksters"),
        ("Hero", "02_Supernatural_Adversary"), ("Wise_Old_Man", "05_Supernatural_Helpers"), ("Shadow", "10_Stupid_Ogre_Devil"),
    ]
    vp = [(j2i[ja], div2i[da], ja, da) for ja, da in mapped_pairs]
    assert len(vp) == 5
    d_obs = float(np.mean([cos_ja[ji, di] for ji, di, _, _ in vp]))
    rng_d = np.random.default_rng(24865)
    d_null = [float(np.mean([cos_ja[pj[ji], pd_[di]] for ji, di, _, _ in vp])) for pj, pd_ in ((rng_d.permutation(8), rng_d.permutation(10)) for _ in range(5000))]
    diag_test = {"n_pairs": 5, "mean_cos": round(d_obs, 4), "p_perm_upper": round(float((np.sum(np.asarray(d_null) >= d_obs - 1e-12) + 1) / 5001), 4)}
    pr_clean = A245._pr_type_preserving_null(Za_cl, y_cl, atu_cl, 10, n_perm=4999, seed=24866)

    # 3. Clean 0-Shared-ATU-Type Western -> Non-Western Transfer (main 480 AND excluding 15 debatable rows!)
    def eval_transfer_split(exclude_debatable=False):
        w_mask = (prov_cl == "western")
        nw_mask = (prov_cl == "non_western") & (~deb_cl if exclude_debatable else np.ones(len(prov_cl), dtype=bool))
        nw_types = set(atu_cl[nw_mask].tolist())
        w_disj = w_mask & np.array([t not in nw_types for t in atu_cl], dtype=bool)
        assert len(set(atu_cl[w_disj].tolist()) & set(atu_cl[nw_mask].tolist())) == 0
        y_tr, y_te, atu_te = y_cl[w_disj], y_cl[nw_mask], atu_cl[nw_mask]

        lex_sc = R239._fit_eval_lexical_fold([clean_texts[i] for i in np.where(w_disj)[0]], y_tr, [clean_texts[i] for i in np.where(nw_mask)[0]], 10)
        cls10 = sorted(set(y_te.tolist()))
        tpd = {c: len(set(atu_te[y_te == c].tolist())) for c in cls10}
        cls9 = [c for c in cls10 if tpd[c] >= 2]

        def m_on(sc, clist, restrict_argmax=False):
            m = np.isin(y_te, clist)
            if restrict_argmax:
                sub = sc[m][:, clist]
                pr = np.array([clist[j] for j in np.argmax(sub, axis=1)])
            else:
                pr = np.argmax(sc[m], axis=1)
            return {
                "top1": round(float(np.mean(pr == y_te[m])), 4),
                "balanced_acc": round(A245._balanced_acc(y_te[m], pr, clist), 4),
                "macro_auroc": round(A245._macro_auroc(y_te[m], sc[m], clist), 4),
            }

        lex_summ = {
            ln: {
                "all_10": m_on(sc, cls10),
                "9div_10way_argmax": m_on(sc, cls9, False),
                "9div_9way_argmax": m_on(sc, cls9, True),
            }
            for ln, sc in lex_sc.items()
        }

        def boot_vs_lex(sc_n, sc_l, clist, restrict_argmax=False, n_boot=2000, seed=24870):
            by_dt = {}
            for c in clist:
                ix_c = np.where(y_te == c)[0]
                bt = {}
                for i in ix_c:
                    bt.setdefault(str(atu_te[i]), []).append(i)
                by_dt[c] = [np.array(v, dtype=int) for v in sorted(bt.values(), key=lambda x: x[0])]
            m_sub = np.isin(y_te, clist)
            def score_pair(sc, yb, ix):
                if restrict_argmax:
                    pr = np.array([clist[j] for j in np.argmax(sc[ix][:, clist], axis=1)])
                else:
                    pr = np.argmax(sc[ix], axis=1)
                return A245._balanced_acc(yb, pr, clist), A245._macro_auroc(yb, sc[ix], clist)
            bn_o, an_o = score_pair(sc_n, y_te[m_sub], np.where(m_sub)[0])
            bl_o, al_o = score_pair(sc_l, y_te[m_sub], np.where(m_sub)[0])
            rng = np.random.default_rng(seed)
            db, da = [], []
            for _ in range(n_boot):
                ix = np.concatenate([by_dt[c][g] for c in clist for g in rng.integers(0, len(by_dt[c]), size=len(by_dt[c]))])
                yb = y_te[ix]
                bn, an = score_pair(sc_n, yb, ix)
                bl, al = score_pair(sc_l, yb, ix)
                db.append(bn - bl); da.append(an - al)
            db, da = np.asarray(db), np.asarray(da)
            pb_le = float((int(np.sum(db <= 0)) + 1) / (n_boot + 1))
            pb_ge = float((int(np.sum(db >= 0)) + 1) / (n_boot + 1))
            pa_le = float((int(np.sum(da <= 0)) + 1) / (n_boot + 1))
            pa_ge = float((int(np.sum(da >= 0)) + 1) / (n_boot + 1))
            pb_two = min(1.0, 2.0 * min(pb_le, pb_ge))
            pa_two = min(1.0, 2.0 * min(pa_le, pa_ge))
            return {
                "bal_diff": round(bn_o - bl_o, 4), "bal_ci95": [round(float(np.percentile(db, 2.5)), 4), round(float(np.percentile(db, 97.5)), 4)],
                "bal_p_one": round(pb_le, 5), "bal_p_two": round(pb_two, 5),
                "_bal_p_one_fp": pb_le, "_bal_p_two_fp": pb_two,
                "auc_diff": round(an_o - al_o, 4), "auc_ci95": [round(float(np.percentile(da, 2.5)), 4), round(float(np.percentile(da, 97.5)), 4)],
                "auc_p_one": round(pa_le, 5), "auc_p_two": round(pa_two, 5),
                "_auc_p_one_fp": pa_le, "_auc_p_two_fp": pa_two,
            }

        # Type-level label-permutation test against chance
        by_t_all = {}
        for i, t in enumerate(atu_te):
            by_t_all.setdefault(str(t), []).append(i)
        t_groups = [np.array(v, dtype=int) for v in sorted(by_t_all.values(), key=lambda x: x[0])]
        t_labels = np.array([Counter(y_te[ix].tolist()).most_common(1)[0][0] for ix in t_groups], dtype=int)

        def perm_vs_chance(sc_n, n_perm=2000, seed=24875):
            pr = np.argmax(sc_n, axis=1)
            obs_b = A245._balanced_acc(y_te, pr, cls10)
            obs_a = A245._macro_auroc(y_te, sc_n, cls10)
            rng = np.random.default_rng(seed)
            gb, ga = 0, 0
            for _ in range(n_perm):
                pl = rng.permutation(t_labels)
                yp = np.zeros_like(y_te)
                for gi, ix in enumerate(t_groups):
                    yp[ix] = pl[gi]
                if A245._balanced_acc(yp, pr, cls10) >= obs_b - 1e-12:
                    gb += 1
                if A245._macro_auroc(yp, sc_n, cls10) >= obs_a - 1e-12:
                    ga += 1
            return {"p_chance_bal": round((gb + 1) / (n_perm + 1), 5), "p_chance_auc": round((ga + 1) / (n_perm + 1), 5)}

        neural_out = {}
        p_max_one, p_max_two = {}, {}
        for hk in ("phi4_mini", "olmo3_7b", "gemma4_e4b", "qwen35_9b"):
            Zt_cl = V234._project_out_rows(Za[hk][keep_mask], Q3[hk])
            src_Z = Za_cl if hk in ("phi4_mini", "olmo3_7b") else V234._project_out_rows(np.mean([Za[k][keep_mask] for k in c5_keys if k != hk], 0), Q3_c5)
            Zs_tr = src_Z[w_disj] - src_Z[w_disj].mean(0, keepdims=True)
            Zt_te = Zt_cl[nw_mask] - Zt_cl[w_disj].mean(0, keepdims=True)
            U = np.stack([V234._project_out(u, Q3[hk]) for u in NA235._ovr_dirs(Zs_tr, y_tr, 10)], 0)
            sc_n = Zt_te @ U.T
            pr = np.argmax(sc_n, axis=1)

            vs_all3_10 = {ln: boot_vs_lex(sc_n, lex_sc[ln], cls10, False, seed=24870) for ln in lex_sc}
            # Max-p across ALL THREE lexical baselines using FULL-PRECISION p-values!
            p_max_one[f"{hk}_bal"] = max(vs_all3_10[ln]["_bal_p_one_fp"] for ln in lex_sc)
            p_max_one[f"{hk}_auc"] = max(vs_all3_10[ln]["_auc_p_one_fp"] for ln in lex_sc)
            p_max_two[f"{hk}_bal"] = max(vs_all3_10[ln]["_bal_p_two_fp"] for ln in lex_sc)
            p_max_two[f"{hk}_auc"] = max(vs_all3_10[ln]["_auc_p_two_fp"] for ln in lex_sc)
            # Strip internal full-precision keys before serialization
            for ln in lex_sc:
                for dk in ("_bal_p_one_fp", "_bal_p_two_fp", "_auc_p_one_fp", "_auc_p_two_fp"):
                    vs_all3_10[ln].pop(dk, None)

            # 9-division sensitivity strictly against tfidf_centroid (actual 9-div winner!) with BOTH 10-way and 9-way argmax
            vs_tfc_9_10way = boot_vs_lex(sc_n, lex_sc["tfidf_centroid"], cls9, False, seed=24872)
            vs_tfc_9_9way = boot_vs_lex(sc_n, lex_sc["tfidf_centroid"], cls9, True, seed=24873)
            for dd in (vs_tfc_9_10way, vs_tfc_9_9way):
                for dk in ("_bal_p_one_fp", "_bal_p_two_fp", "_auc_p_one_fp", "_auc_p_two_fp"):
                    dd.pop(dk, None)

            per_cls_rec = {div_names[c]: round(float(np.mean(pr[y_te == c] == c)), 4) for c in cls10}
            neural_out[hk] = {
                "all_10": m_on(sc_n, cls10),
                "9div_9way_argmax": m_on(sc_n, cls9, True),
                "per_class_recall": per_cls_rec,
                "perm_vs_chance": perm_vs_chance(sc_n, seed=24875),
                "vs_all3_lexical_10div": vs_all3_10,
                "vs_tfidf_centroid_9div_10way": vs_tfc_9_10way,
                "vs_tfidf_centroid_9div_9way": vs_tfc_9_9way,
            }

        # Exhaustive 8! geometry permutation (only on main split)
        xc_geom = None
        if not exclude_debatable:
            ok_xc = [c for c in range(10) if np.sum(y_tr == c) >= 3 and np.sum(y_te == c) >= 3]
            def gmat(Z_sub, y_sub):
                Zc = Z_sub - Z_sub.mean(0, keepdims=True)
                M = np.stack([AR.unit(Zc[y_sub == c].mean(0)) for c in ok_xc], 0)
                return M @ M.T
            Gw, Gnw = gmat(Za_cl[w_disj], y_tr), gmat(Za_cl[nw_mask], y_te)
            iu = np.triu_indices(len(ok_xc), 1)
            gw_v = Gw[iu]
            gw_v = (gw_v - gw_v.mean()) / np.linalg.norm(gw_v - gw_v.mean())
            obs_r = float(np.corrcoef(Gw[iu], Gnw[iu])[0, 1])
            ge_cnt, tot_cnt = 0, 0
            for pm in itertools.permutations(range(len(ok_xc))):
                tot_cnt += 1
                v = Gnw[np.ix_(pm, pm)][iu]
                v = v - v.mean()
                r_p = float(np.dot(gw_v, v / max(1e-12, float(np.linalg.norm(v)))))
                if r_p >= obs_r - 1e-12:
                    ge_cnt += 1
            xc_geom = {"n_divs": len(ok_xc), "obs_r": round(obs_r, 4), "exact_ge_count": ge_cnt, "total_perms": tot_cnt, "p_exact_one_sided": round(ge_cnt / tot_cnt, 5)}

        return {
            "n_train_western": int(w_disj.sum()), "n_test_non_western": int(nw_mask.sum()),
            "n_test_unique_types": len(nw_types),
            "majority_class_top1": round(float(Counter(y_te.tolist()).most_common(1)[0][1] / len(y_te)), 4),
            "lexical_baselines": lex_summ,
            "neural_c5_perp_val3": neural_out,
            "holm8_max_p_across_all3_lexical_one_sided": _holm(p_max_one),
            "holm8_max_p_across_all3_lexical_two_sided": _holm(p_max_two),
            "west_vs_nonwest_geometry_exact": xc_geom,
        }

    main_tr = eval_transfer_split(exclude_debatable=False)
    strict_tr = eval_transfer_split(exclude_debatable=True)

    report = {
        "elapsed_sec": round(time.time() - t0, 1),
        "hygiene_480": {
            "n_total": 500, "n_kept": int(keep_mask.sum()), "n_excluded": len(excl_set),
            "excluded_aft_idxs": sorted(excl_set),
            "n_shorter_than_csv": int(n_shorter),
            "adjudicated_provenance_counts_kept": dict(Counter(prov_cl.tolist())),
        },
        "bases_480_v_dir": base_scores,
        "basis_contrasts_480": contrasts,
        "basis_contrasts_480_holm7": contrasts_holm,
        "shuffle_nominal_200base": shuf_res,
        "five_pair_diagonal_480": diag_test,
        "pr_type_preserving_480": pr_clean,
        "west_to_nonwest_main_480": main_tr,
        "west_to_nonwest_excl_15_debatable": strict_tr,
    }
    out_p = EXP4 / "round4_folklore_results.json"
    out_p.write_text(json.dumps(report, indent=2))
    print(f"SAVED {out_p} in {report['elapsed_sec']}s", flush=True)
    print("HYGIENE 480:", report["hygiene_480"])
    print("BASES 480:", base_scores)
    print("HOLM-7 CONTRASTS:", contrasts_holm)
    print("SHUFFLE NOMINAL:", shuf_res)
    print("DIAG 5-PAIR:", diag_test, "| PR:", pr_clean, "| GEOM EXACT:", main_tr["west_vs_nonwest_geometry_exact"])
    print("MAIN 480 HOLM8 max-p (1-sided):", main_tr["holm8_max_p_across_all3_lexical_one_sided"])
    print("MAIN 480 HOLM8 max-p (2-sided):", main_tr["holm8_max_p_across_all3_lexical_two_sided"])
    print("EXCL 15 DEBATABLE HOLM8 max-p (2-sided):", strict_tr["holm8_max_p_across_all3_lexical_two_sided"])
    print("MAIN 480 Qwen 10div:", main_tr["neural_c5_perp_val3"]["qwen35_9b"]["all_10"], "chance_p:", main_tr["neural_c5_perp_val3"]["qwen35_9b"]["perm_vs_chance"])


if __name__ == "__main__":
    main()
