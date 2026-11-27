"""Round-5 Final Polish Audit for Claim C (480 & 476 Cleaned Folklore Records, 9-Bin & 10-Bin Like-for-Like All-3-Lexical Holm-8, Joint Source+Target Bootstrap & Full Hygiene Serialization).

Resolves EVERY remaining Round-4 finding on Claim C from BOTH Fable 5.2 and Astra 6.2:
  1. Serializes exact CSV comparison counts (442 shorter, 38 equal, 0 longer), exact >=25-word consecutive
     normalized overlap check across retained EXCERPTS (and train-vs-test maximum), plus full-CSV overlap check (379/383).
  2. Serializes exact 10 ATU-derived bin definitions (showing 07_Supernatural_Power pools 650-699 + 700-749)
     and exact per-bin Western train / Non-Western test record and unique-type counts (showing 10_Stupid_Ogre_Devil
     is 39 train / 2 test [1 type] and Western-train majority scores 2/137 = 1.46% vs test-majority oracle 29.93%).
  3. Runs BOTH fixed-corpus Holm-7 basis contrasts AND 2,000-draw JOINT bootstrap resampling BOTH the 40 Jung
     settings (refitting Q_jung7, raw PCA7, setting-centered PCA7) AND target ATU types within bin!
  4. Runs like-for-like ALL-3-LEXICAL `max_p` + TWO-SIDED Holm-8 on BOTH:
     - 9-bin split (excluding the 2-record / 1-type singleton 10_Stupid_Ogre_Devil) with BOTH 9-way and 10-way argmax;
     - 10-bin split (all 10 ATU-derived bins);
     across ALL THREE corpora: (a) main 480 records, (b) excluding 15 debatable Non-Western provenance rows,
     and (c) 476 records also excluding the 4 framing-heavy excerpts identified by Astra 6.2 [277, 356, 383, 1390]
     (both main 476 and 476-excl-15-debatable)!
  5. Uses 2,000 type-level label permutations against chance (with Holm-8 adjustment) and saves per-class recall
     (plus exact hit/total counts per bin, e.g., Qwen 1/2 on 10_Stupid_Ogre_Devil).
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
EXP5 = SCRATCH / "exp249_round5"
EXP5.mkdir(parents=True, exist_ok=True)
EXP4 = SCRATCH / "exp248_round4"
EXP3 = SCRATCH / "exp247_round3"
EXP245 = SCRATCH / "exp245_controls"
FOLK = SCRATCH / "folklore_test"
REPO_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(Path(__file__).resolve().parents[2] / "bench" / "archetypes" / "fluent" / "plato")))

for p in (str(REPO_ARCH), str(REPO_ARCH / "fluent"), str(FOLK), str(EXP245), str(EXP3), str(EXP4)):
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
import round4_folklore_audit as R4F  # noqa: E402

FRAMING_SENS_AFT_IDXS = {277, 356, 383, 1390}


def _holm(p_map):
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for r, (k, pv) in enumerate(items):
        run = max(run, min(1.0, (m - r) * pv))
        out[k] = round(run, 5)
    return out


def _max_consecutive_overlap(word_lists, idxs_a, idxs_b=None):
    """Exact max consecutive normalized-word overlap between pairs in word_lists."""
    # Check >=25 shingles first, plus exact longest common contiguous run across pairs
    shingles = {}
    pairs_ge25 = set()
    for i in idxs_a:
        ws = word_lists[i]
        for p in range(len(ws) - 24):
            sh = tuple(ws[p:p + 25])
            if sh in shingles and shingles[sh] != i:
                pairs_ge25.add(tuple(sorted((shingles[sh], i))))
            else:
                shingles[sh] = i

    # Exact max contiguous run (using rolling length on 4-gram index)
    def longest_pair_run(u, v):
        if not u or not v:
            return 0
        # DP over common words via index map
        pos = {}
        for j, w in enumerate(v):
            pos.setdefault(w, []).append(j)
        best = 0
        prev = {}
        for w in u:
            curr = {}
            for j in pos.get(w, ()):
                val = prev.get(j - 1, 0) + 1
                curr[j] = val
                if val > best:
                    best = val
            prev = curr
        return best

    if idxs_b is None:
        # Filter candidate pairs sharing at least one 5-gram
        gram5 = {}
        cands = set()
        for i in idxs_a:
            ws = word_lists[i]
            for p in range(len(ws) - 4):
                g = tuple(ws[p:p + 5])
                for prev_i in gram5.get(g, ()):
                    if prev_i != i:
                        cands.add((min(prev_i, i), max(prev_i, i)))
                gram5.setdefault(g, set()).add(i)
        max_run, max_pair = 0, None
        for a, b in cands:
            r = longest_pair_run(word_lists[a], word_lists[b])
            if r > max_run:
                max_run, max_pair = r, (a, b)
        return len(pairs_ge25), max_run, max_pair
    else:
        gram5 = {}
        for i in idxs_a:
            ws = word_lists[i]
            for p in range(len(ws) - 4):
                gram5.setdefault(tuple(ws[p:p + 5]), set()).add(i)
        cands = set()
        for j in idxs_b:
            ws = word_lists[j]
            for p in range(len(ws) - 4):
                for i in gram5.get(tuple(ws[p:p + 5]), ()):
                    cands.add((i, j))
        max_run, max_pair = 0, None
        for a, b in cands:
            r = longest_pair_run(word_lists[a], word_lists[b])
            if r > max_run:
                max_run, max_pair = r, (a, b)
        return max_run, max_pair


def main():
    t0 = time.time()
    aft_rows = json.loads((FOLK / "aft_real_myths_500.json").read_text())
    m4 = json.loads((EXP4 / "aft_clean248_manifest.json").read_text())
    jung_rows = NA235.load_named_arch_data()
    fam_rows = json.loads((FOLK / "family_roles239_data.json").read_text())
    jobs_rows = NJ237.load_nonjung_data()

    # Load aft.csv & atu_df.csv for exact CSV comparison and ATU-bin mapping audit
    with open(FOLK / "aft.csv", newline="", encoding="utf-8", errors="replace") as f:
        csv_rows = list(csv.DictReader(f))
    atu_cat_map = {}
    if (FOLK / "atu_df.csv").exists():
        with open(FOLK / "atu_df.csv", newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                tid = str(row.get("atu_id") or "").strip()
                cat = str(row.get("division") or "").strip()
                if not cat or cat == "NA":
                    cat = str(row.get("chapter") or "").strip()
                if tid and cat:
                    atu_cat_map[tid] = cat

    div_names = list(R239.ATU_DIVS)
    div2i = {d: i for i, d in enumerate(div_names)}
    y_a = np.array([div2i[r["division"]] for r in aft_rows], dtype=int)
    atu_a = np.array([str(r["atu_id"]) for r in aft_rows])
    aidx_a = np.array([int(r["aft_idx"]) for r in aft_rows], dtype=int)

    keep480 = np.array([bool(m["included"]) for m in m4], dtype=bool)
    keep476 = keep480 & (~np.isin(aidx_a, list(FRAMING_SENS_AFT_IDXS)))
    assert int(keep480.sum()) == 480 and int(keep476.sum()) == 476

    prov_a = np.array([m["adjudicated_provenance"] for m in m4])
    deb_a = np.array([R4F._is_debatable_nonwest(r["provenance"]) for r in aft_rows], dtype=bool)

    # 1. Exact CSV word-count breakdown & excerpt overlap audit on 480 kept records
    n_short = n_eq = n_long = 0
    for i in np.where(keep480)[0]:
        rw = len(aft_rows[i]["text"].split())
        cw = len((csv_rows[int(aidx_a[i])].get("text") or "").split())
        if rw < cw:
            n_short += 1
        elif rw == cw:
            n_eq += 1
        else:
            n_long += 1
    assert n_short + n_eq + n_long == 480

    nword_lists = [re.findall(r"[a-z]+", r["text"].lower()) for r in aft_rows]
    kept_ix = np.where(keep480)[0].tolist()
    n_ge25_ex, max_ex_run, max_ex_pair = _max_consecutive_overlap(nword_lists, kept_ix)

    # Full-CSV overlap check specifically for 379 vs 383 noted by Astra 6.2
    w379 = re.findall(r"[a-z]+", (csv_rows[379].get("text") or "").lower())
    w383 = re.findall(r"[a-z]+", (csv_rows[383].get("text") or "").lower())
    csv_379_383_overlap = 0
    pos383 = {}
    for j, w in enumerate(w383):
        pos383.setdefault(w, []).append(j)
    prev = {}
    for w in w379:
        curr = {}
        for j in pos383.get(w, ()):
            v = prev.get(j - 1, 0) + 1
            curr[j] = v
            if v > csv_379_383_overlap:
                csv_379_383_overlap = v
        prev = curr

    # Train-vs-test excerpt max overlap on main 480 split
    w_m = keep480 & (prov_a == "western")
    nw_m = keep480 & (prov_a == "non_western")
    nw_types_480 = set(atu_a[nw_m].tolist())
    w_disj_480 = w_m & np.array([t not in nw_types_480 for t in atu_a], dtype=bool)
    max_tr_te_run, max_tr_te_pair = _max_consecutive_overlap(nword_lists, np.where(w_disj_480)[0].tolist(), np.where(nw_m)[0].tolist())

    # Exact ATU-derived bin mapping audit on 480 kept records
    bin_catalog_breakdown = {}
    for d in div_names:
        ix = [i for i in kept_ix if aft_rows[i]["division"] == d]
        atu_nums = []
        for i in ix:
            m_num = re.match(r"(\d+)", str(aft_rows[i]["atu_id"]))
            if m_num:
                atu_nums.append(int(m_num.group(1)))
        cats = Counter(atu_cat_map.get(str(aft_rows[i]["atu_id"]), "unknown") for i in ix)
        bin_catalog_breakdown[d] = {
            "n_kept_480": len(ix),
            "atu_numeric_min_max": [min(atu_nums), max(atu_nums)] if atu_nums else None,
            "n_train_western_main": int(np.sum(w_disj_480 & (y_a == div2i[d]))),
            "n_test_nonwest_main": int(np.sum(nw_m & (y_a == div2i[d]))),
            "n_test_unique_types_main": len(set(atu_a[nw_m & (y_a == div2i[d])].tolist())),
            "source_catalog_subcategories": dict(cats),
        }

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
    Za_c5_all = V234._project_out_rows(np.mean([Za[k] for k in c5_keys], 0), Q3_c5)

    jung_names = list(NA235.ARCHETYPES)
    y_j = np.array([jung_names.index(r["archetype"]) for r in jung_rows], dtype=int)
    sids_j = np.array([int(r["setting_id"]) for r in jung_rows], dtype=int)
    uniq_sj = sorted(set(int(s) for s in sids_j))
    sj_ix = [np.where(sids_j == s)[0] for s in uniq_sj]

    # 2. Joint 40-Jung-setting + target-ATU-type bootstrap (2,000 draws) on 480 records!
    def joint_source_target_boot(keep_mask, n_boot=2000, seed=24950):
        Za_cl = Za_c5_all[keep_mask]
        y_cl = y_a[keep_mask]
        atu_cl = atu_a[keep_mask]
        # Pre-group target ATU types within each bin
        by_dt = {}
        for c in range(10):
            bt = {}
            for i in np.where(y_cl == c)[0]:
                bt.setdefault(str(atu_cl[i]), []).append(i)
            by_dt[c] = [np.array(v, dtype=int) for v in sorted(bt.values(), key=lambda x: x[0])]
        # Observed
        Qj_o = R239._centroid_basis_k(Zj_c5, y_j, 8, 7)
        Qp_o = A245._pca_basis_k(Zj_c5, 7)
        Qs_o = A245._setting_centered_pca_basis_k(Zj_c5, sids_j, 7)
        vj_o = R239._eval_basis_capture(Qj_o, Za_cl, y_cl, 10)["v_dir"]
        vp_o = R239._eval_basis_capture(Qp_o, Za_cl, y_cl, 10)["v_dir"]
        vs_o = R239._eval_basis_capture(Qs_o, Za_cl, y_cl, 10)["v_dir"]

        rng = np.random.default_rng(seed)
        d_raw, d_sc = [], []
        for _ in range(n_boot):
            # Resample 40 Jung settings
            ch_s = rng.integers(0, len(uniq_sj), size=len(uniq_sj))
            ix_j = np.concatenate([sj_ix[c] for c in ch_s])
            sids_b = np.concatenate([np.full(len(sj_ix[c]), b_i, dtype=int) for b_i, c in enumerate(ch_s)])
            Zb, yb = Zj_c5[ix_j], y_j[ix_j]
            Qj_b = R239._centroid_basis_k(Zb, yb, 8, 7)
            Qp_b = A245._pca_basis_k(Zb, 7)
            Qs_b = A245._setting_centered_pca_basis_k(Zb, sids_b, 7)
            # Resample target ATU types within bin
            ix_t = np.concatenate([by_dt[c][g] for c in range(10) for g in rng.integers(0, len(by_dt[c]), size=len(by_dt[c]))])
            Zt_b, yt_b = Za_cl[ix_t], y_cl[ix_t]
            U_b = NA235._ovr_dirs(Zt_b - Zt_b.mean(0, keepdims=True), yt_b, 10)
            vj_b = float(np.mean(np.sum((U_b @ Qj_b) ** 2, axis=1)))
            vp_b = float(np.mean(np.sum((U_b @ Qp_b) ** 2, axis=1)))
            vs_b = float(np.mean(np.sum((U_b @ Qs_b) ** 2, axis=1)))
            d_raw.append(vp_b - vj_b)
            d_sc.append(vs_b - vj_b)
        d_raw, d_sc = np.asarray(d_raw), np.asarray(d_sc)
        def summ(obs_d, arr):
            p_le = float((int(np.sum(arr <= 0)) + 1) / (n_boot + 1))
            p_ge = float((int(np.sum(arr >= 0)) + 1) / (n_boot + 1))
            return {
                "obs_diff": round(obs_d, 5),
                "ci95": [round(float(np.percentile(arr, 2.5)), 5), round(float(np.percentile(arr, 97.5)), 5)],
                "p_two_sided_nominal": round(min(1.0, 2.0 * min(p_le, p_ge)), 5),
            }
        return {
            "n_records": int(keep_mask.sum()),
            "v_dir_obs": {"Q_jung7": round(vj_o, 5), "PCA7_unlabeled": round(vp_o, 5), "PCA7_setcent": round(vs_o, 5)},
            "PCA7_unlabeled_minus_Q_jung7_joint": summ(vp_o - vj_o, d_raw),
            "PCA7_setcent_minus_Q_jung7_joint": summ(vs_o - vj_o, d_sc),
        }

    joint_480 = joint_source_target_boot(keep480, n_boot=2000, seed=24950)
    joint_476 = joint_source_target_boot(keep476, n_boot=2000, seed=24951)

    # 3. Full like-for-like ALL-3-LEXICAL max_p + TWO-SIDED Holm-8 across 9-bin (9-way & 10-way) AND 10-bin!
    def eval_regime(keep_mask, exclude_debatable=False):
        y_cl, atu_cl, prov_cl, deb_cl = y_a[keep_mask], atu_a[keep_mask], prov_a[keep_mask], deb_a[keep_mask]
        clean_texts = [aft_rows[i]["text"] for i in np.where(keep_mask)[0]]
        Za_cl = Za_c5_all[keep_mask]

        w_mask = (prov_cl == "western")
        nw_mask = (prov_cl == "non_western") & (~deb_cl if exclude_debatable else np.ones(len(prov_cl), dtype=bool))
        nw_types = set(atu_cl[nw_mask].tolist())
        w_disj = w_mask & np.array([t not in nw_types for t in atu_cl], dtype=bool)
        assert len(set(atu_cl[w_disj].tolist()) & set(atu_cl[nw_mask].tolist())) == 0
        y_tr, y_te, atu_te = y_cl[w_disj], y_cl[nw_mask], atu_cl[nw_mask]

        tr_maj_cls, tr_maj_cnt = Counter(y_tr.tolist()).most_common(1)[0]
        te_maj_cls, te_maj_cnt = Counter(y_te.tolist()).most_common(1)[0]

        lex_sc = R239._fit_eval_lexical_fold([clean_texts[i] for i in np.where(w_disj)[0]], y_tr, [clean_texts[i] for i in np.where(nw_mask)[0]], 10)
        cls10 = sorted(set(y_te.tolist()))
        tpd = {c: len(set(atu_te[y_te == c].tolist())) for c in cls10}
        cls9 = [c for c in cls10 if tpd[c] >= 2]

        def m_on(sc, clist, restrict_argmax=False):
            m = np.isin(y_te, clist)
            pr = np.array([clist[j] for j in np.argmax(sc[m][:, clist], axis=1)]) if restrict_argmax else np.argmax(sc[m], axis=1)
            return {
                "top1": round(float(np.mean(pr == y_te[m])), 4),
                "balanced_acc": round(A245._balanced_acc(y_te[m], pr, clist), 4),
                "macro_auroc": round(A245._macro_auroc(y_te[m], sc[m], clist), 4),
            }

        def boot_vs_lex(sc_n, sc_l, clist, restrict_argmax=False, n_boot=2000, seed=24870):
            by_dt = {}
            for c in clist:
                bt = {}
                for i in np.where(y_te == c)[0]:
                    bt.setdefault(str(atu_te[i]), []).append(i)
                by_dt[c] = [np.array(v, dtype=int) for v in sorted(bt.values(), key=lambda x: x[0])]
            m_sub = np.isin(y_te, clist)
            def score_p(sc, yb, ix):
                pr = np.array([clist[j] for j in np.argmax(sc[ix][:, clist], axis=1)]) if restrict_argmax else np.argmax(sc[ix], axis=1)
                return A245._balanced_acc(yb, pr, clist), A245._macro_auroc(yb, sc[ix], clist)
            bn_o, an_o = score_p(sc_n, y_te[m_sub], np.where(m_sub)[0])
            bl_o, al_o = score_p(sc_l, y_te[m_sub], np.where(m_sub)[0])
            rng = np.random.default_rng(seed)
            db, da = [], []
            for _ in range(n_boot):
                ix = np.concatenate([by_dt[c][g] for c in clist for g in rng.integers(0, len(by_dt[c]), size=len(by_dt[c]))])
                yb = y_te[ix]
                bn, an = score_p(sc_n, yb, ix)
                bl, al = score_p(sc_l, yb, ix)
                db.append(bn - bl); da.append(an - al)
            db, da = np.asarray(db), np.asarray(da)
            pb_le = float((int(np.sum(db <= 0)) + 1) / (n_boot + 1))
            pb_ge = float((int(np.sum(db >= 0)) + 1) / (n_boot + 1))
            pa_le = float((int(np.sum(da <= 0)) + 1) / (n_boot + 1))
            pa_ge = float((int(np.sum(da >= 0)) + 1) / (n_boot + 1))
            pb_two, pa_two = min(1.0, 2.0 * min(pb_le, pb_ge)), min(1.0, 2.0 * min(pa_le, pa_ge))
            return {
                "bal_diff": round(bn_o - bl_o, 4), "bal_ci95": [round(float(np.percentile(db, 2.5)), 4), round(float(np.percentile(db, 97.5)), 4)],
                "bal_p_two": round(pb_two, 5), "_bal_p_two_fp": pb_two,
                "auc_diff": round(an_o - al_o, 4), "auc_ci95": [round(float(np.percentile(da, 2.5)), 4), round(float(np.percentile(da, 97.5)), 4)],
                "auc_p_two": round(pa_two, 5), "_auc_p_two_fp": pa_two,
            }

        # Type-level chance permutation (2,000 draws) on BOTH 10-bin and 9-bin!
        def perm_chance(sc_n, clist, restrict_argmax=False, n_perm=2000, seed=24875):
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
            obs_b = A245._balanced_acc(ym, pr, clist)
            obs_a = A245._macro_auroc(ym, sc_m, clist)
            rng = np.random.default_rng(seed)
            gb = ga = 0
            for _ in range(n_perm):
                pl = rng.permutation(tl)
                yp = np.zeros_like(ym)
                for gi, ix in enumerate(tg):
                    yp[ix] = pl[gi]
                if A245._balanced_acc(yp, pr, clist) >= obs_b - 1e-12:
                    gb += 1
                if A245._macro_auroc(yp, sc_m, clist) >= obs_a - 1e-12:
                    ga += 1
            pb, pa = (gb + 1) / (n_perm + 1), (ga + 1) / (n_perm + 1)
            return {"p_chance_bal": round(pb, 5), "p_chance_auc": round(pa, 5), "_pb": pb, "_pa": pa}

        lex_summ = {ln: {"all_10": m_on(sc, cls10, False), "9bin_10way": m_on(sc, cls9, False), "9bin_9way": m_on(sc, cls9, True)} for ln, sc in lex_sc.items()}
        neural = {}
        p10, p9_10w, p9_9w = {}, {}, {}
        ch10, ch9 = {}, {}

        for hk in ("phi4_mini", "olmo3_7b", "gemma4_e4b", "qwen35_9b"):
            Zt_cl = V234._project_out_rows(Za[hk][keep_mask], Q3[hk])
            src_Z = Za_cl if hk in ("phi4_mini", "olmo3_7b") else V234._project_out_rows(np.mean([Za[k][keep_mask] for k in c5_keys if k != hk], 0), Q3_c5)
            Zs_tr = src_Z[w_disj] - src_Z[w_disj].mean(0, keepdims=True)
            Zt_te = Zt_cl[nw_mask] - Zt_cl[w_disj].mean(0, keepdims=True)
            U = np.stack([V234._project_out(u, Q3[hk]) for u in NA235._ovr_dirs(Zs_tr, y_tr, 10)], 0)
            sc_n = Zt_te @ U.T
            pr10 = np.argmax(sc_n, axis=1)

            vs10 = {ln: boot_vs_lex(sc_n, lex_sc[ln], cls10, False, seed=24870) for ln in lex_sc}
            vs9_10 = {ln: boot_vs_lex(sc_n, lex_sc[ln], cls9, False, seed=24872) for ln in lex_sc}
            vs9_9 = {ln: boot_vs_lex(sc_n, lex_sc[ln], cls9, True, seed=24873) for ln in lex_sc}

            for d_out, vs_d in ((p10, vs10), (p9_10w, vs9_10), (p9_9w, vs9_9)):
                d_out[f"{hk}_bal"] = max(vs_d[ln]["_bal_p_two_fp"] for ln in lex_sc)
                d_out[f"{hk}_auc"] = max(vs_d[ln]["_auc_p_two_fp"] for ln in lex_sc)
                for ln in lex_sc:
                    vs_d[ln].pop("_bal_p_two_fp", None)
                    vs_d[ln].pop("_auc_p_two_fp", None)

            c10 = perm_chance(sc_n, cls10, False, seed=24875)
            c9 = perm_chance(sc_n, cls9, True, seed=24876)
            ch10[f"{hk}_bal"], ch10[f"{hk}_auc"] = c10.pop("_pb"), c10.pop("_pa")
            ch9[f"{hk}_bal"], ch9[f"{hk}_auc"] = c9.pop("_pb"), c9.pop("_pa")

            rec_counts = {div_names[c]: f"{int(np.sum(pr10[y_te == c] == c))}/{int(np.sum(y_te == c))}" for c in cls10}
            neural[hk] = {
                "all_10": m_on(sc_n, cls10, False),
                "9bin_10way": m_on(sc_n, cls9, False),
                "9bin_9way": m_on(sc_n, cls9, True),
                "per_bin_hits_over_total": rec_counts,
                "perm_vs_chance_10bin": c10,
                "perm_vs_chance_9bin_9way": c9,
                "vs_all3_lex_10bin": vs10,
                "vs_all3_lex_9bin_10way": vs9_10,
                "vs_all3_lex_9bin_9way": vs9_9,
            }

        return {
            "n_train_western": int(w_disj.sum()),
            "n_test_non_western": int(nw_mask.sum()),
            "n_test_unique_types": len(nw_types),
            "test_majority_oracle_top1": {"bin": div_names[te_maj_cls], "hits": f"{te_maj_cnt}/{len(y_te)}", "rate": round(te_maj_cnt / len(y_te), 4)},
            "western_train_majority_on_test": {"bin": div_names[tr_maj_cls], "train_hits": f"{tr_maj_cnt}/{len(y_tr)}", "test_hits": f"{int(np.sum(y_te == tr_maj_cls))}/{len(y_te)}", "test_rate": round(float(np.mean(y_te == tr_maj_cls)), 4)},
            "lexical_baselines": lex_summ,
            "neural_c5_perp_val3": neural,
            "holm8_two_sided_maxp_all3_10bin": _holm(p10),
            "holm8_two_sided_maxp_all3_9bin_10way": _holm(p9_10w),
            "holm8_two_sided_maxp_all3_9bin_9way": _holm(p9_9w),
            "holm8_perm_vs_chance_10bin": _holm(ch10),
            "holm8_perm_vs_chance_9bin_9way": _holm(ch9),
        }

    reg_480_main = eval_regime(keep480, exclude_debatable=False)
    reg_480_excl15 = eval_regime(keep480, exclude_debatable=True)
    reg_476_main = eval_regime(keep476, exclude_debatable=False)
    reg_476_excl15 = eval_regime(keep476, exclude_debatable=True)

    out = {
        "elapsed_sec": round(time.time() - t0, 1),
        "hygiene_and_overlap_480": {
            "n_kept": 480,
            "csv_wordcount_breakdown": {"shorter": n_short, "equal": n_eq, "longer": n_long},
            "excerpt_overlap_ge25_pairs": n_ge25_ex,
            "excerpt_max_consecutive_overlap_words": max_ex_run,
            "excerpt_max_overlap_aft_idxs": [int(aidx_a[max_ex_pair[0]]), int(aidx_a[max_ex_pair[1]])] if max_ex_pair else None,
            "train_vs_test_excerpt_max_overlap_words": max_tr_te_run,
            "train_vs_test_max_overlap_aft_idxs": [int(aidx_a[max_tr_te_pair[0]]), int(aidx_a[max_tr_te_pair[1]])] if max_tr_te_pair else None,
            "full_csv_overlap_379_vs_383_words": csv_379_383_overlap,
            "framing_sensitivity_excluded_aft_idxs": sorted(FRAMING_SENS_AFT_IDXS),
        },
        "atu_derived_10bin_mapping": bin_catalog_breakdown,
        "joint_source_target_boot_480": joint_480,
        "joint_source_target_boot_476": joint_476,
        "transfer_480_main": reg_480_main,
        "transfer_480_excl15_debatable": reg_480_excl15,
        "transfer_476_excl4_framing_main": reg_476_main,
        "transfer_476_excl4_framing_excl15_debatable": reg_476_excl15,
    }
    out_p = EXP5 / "round5_folklore_results.json"
    out_p.write_text(json.dumps(out, indent=2))
    print(f"SAVED {out_p} in {out['elapsed_sec']}s", flush=True)
    print("HYGIENE:", out["hygiene_and_overlap_480"])
    print("JOINT 480:", joint_480)
    print("JOINT 476:", joint_476)
    for name, reg in [("480_main", reg_480_main), ("480_excl15", reg_480_excl15), ("476_main", reg_476_main), ("476_excl15", reg_476_excl15)]:
        print(f"=== {name} (tr={reg['n_train_western']}, te={reg['n_test_non_western']}) ===")
        print("  10-bin Holm8 all3:", reg["holm8_two_sided_maxp_all3_10bin"])
        print("  9-bin 10way Holm8 all3:", reg["holm8_two_sided_maxp_all3_9bin_10way"])
        print("  9-bin 9way  Holm8 all3:", reg["holm8_two_sided_maxp_all3_9bin_9way"])
        print("  9-bin 9way chance Holm8:", reg["holm8_perm_vs_chance_9bin_9way"])


if __name__ == "__main__":
    main()
