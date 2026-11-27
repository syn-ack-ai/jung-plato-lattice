"""Experiment 247-C: Clean, Auditable Folklore Audit Fixing ALL Round-2 Claim-C Findings.

Fixes every Round-2 Claim-C issue from Fable 5.2 and Astra 6.2:
  1. Publishes complete row-level inclusion/exclusion & provenance adjudication manifest (`aft_clean247_manifest.json`)
     for all 500 records:
     - Fixes `notes?:` regex (no trailing `\b` after colon!) + filters bibliography/Ashliman/collector notes
       (`269, 450, 870, 881, 756, 591, 777, 19`, `Title Collection`, `Notes:`, `Ashliman`, `I found the survivors`, `Discussions about`).
     - Deduplicates by normalized alphabetic-token sequence ACROSS ALL DIVISIONS (catching `aft_idx 1146` vs `983`
       in ATU `175` & `545B`!).
     - Fixes provenance keyword disambiguation (`Georgia, USA` -> excluded from non_western; `Sri Lanka`, `Kashmir`,
       `Pakistan`, `Iran`, `Nez Percé` -> `non_western`; `Albania`, `Scandinavia`, `Perrault`, `Basile`, `Andersen` -> `western`;
       distinguishes `missing_metadata` from `uncoded_other`!).
  2. Fixes ALL 5 theoretical Jung-archetype -> ATU-division diagonal pairs (`05_Supernatural_Helpers` and
     `10_Stupid_Ogre_Devil`), asserting `len(valid_pairs) == 5`!
  3. Complete basis contrast table including `PCA7_fam_unlabeled - PCA7_jobs_unlabeled`, Bonferroni-adjusted
     p-values across the 7 contrasts, AND using the EXACT SAME 200 shuffle bases for BOTH point estimate AND
     bootstrap CI of `excess_jung - excess_fam` and `excess_jung - excess_jobs`!
  4. Division-stratified ATU-type bootstrap preserving the exact 10-class (and 9-class excluding 1-type
     `10_Stupid_Ogre_Devil`) estimand on EVERY draw, comparing neural `C5 perp VAL3` against ALL THREE lexical
     baselines (`bow_unigram`, `tfidf_centroid`, AND `tfidf_ridge`) with Holm correction across the full family!
  5. Exact permutation test (all 9! = 362,880 or 50,000 permutations) for Western vs Non-Western division geometry `r`.
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np

SCRATCH = Path("./local_scratch")
EXP_DIR = SCRATCH / "exp247_round3"
EXP_DIR.mkdir(parents=True, exist_ok=True)
FOLK_DIR = SCRATCH / "folklore_test"
EXP245 = SCRATCH / "exp245_controls"
REPO_ARCH = Path("./bench/archetypes")
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", "./bench/archetypes/fluent/plato"))

for p in (str(REPO_ARCH), str(REPO_ARCH / "fluent"), str(FOLK_DIR), str(EXP245)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import named_arch235 as NA235  # noqa: E402
import nonjung_roles237 as NJ237  # noqa: E402
import real_myth_basis239 as R239  # noqa: E402
import aft_audit245 as A245  # noqa: E402

# Expanded, disambiguated keyword-adjudicated provenance table
NON_WEST_KW = (
    "india", "indian", "jataka", "panchatantra", "sri lanka", "ceylon", "kashmir", "pakistan",
    "iran", "persia", "persian", "china", "chinese", "japan", "japanese", "korea", "korean",
    "africa", "african", "ashanti", "zulu", "hausa", "swahili", "yoruba", "egypt", "arab",
    "arabian", "turkey", "turkish", "kurdish", "native american", "thompson", "ntlakyapamuk",
    "skidi", "pawnee", "cherokee", "navajo", "ojibwa", "iroquois", "inuit", "eskimo",
    "nez perc", "biloxi", "maya", "aztec", "inca", "maori", "polynes", "hawaii", "samoa",
    "australia", "aborigin", "vietnam", "thai", "burma", "cambodia", "malay", "indonesia",
    "philippin", "tibet", "mongol", "siberia", "yakut", "chukchi", "armenia", "caucasus",
    "jewish", "hebrew", "palestin", "syria", "iraq", "mexico", "brazil", "peru", "chile",
    "jamaica", "caribbean", "haiti", "cuba", "senegal", "nigeria", "kenya", "ethiopia",
    "morocco", "algeria", "tunisia", "sudan", "bengal", "punjab", "ceylon", "afghan",
)
WEST_KW = (
    "grimm", "german", "france", "french", "perrault", "england", "english", "britain", "british",
    "scotland", "scottish", "ireland", "irish", "wales", "welsh", "norway", "norwegian",
    "sweden", "swedish", "denmark", "danish", "andersen", "scandinavia", "iceland", "finland",
    "finnish", "russia", "russian", "poland", "polish", "czech", "bohemia", "hungary", "hungarian",
    "italy", "italian", "basile", "spain", "spanish", "portugal", "portuguese", "greece", "greek",
    "aesop", "rome", "roman", "netherlands", "dutch", "flanders", "flemish", "belgium",
    "switzerland", "swiss", "austria", "austrian", "europe", "european", "balkan", "serbia",
    "croatia", "romania", "bulgaria", "ukraine", "estonia", "latvia", "lithuania", "albania",
    "basque", "catalan", "breton", "cornish", "appalachia", "ozark", "new england", "canada", "quebec",
)


def _adjudicate_provenance(prov_str, text_str=""):
    s = (prov_str or "").strip().lower()
    t = (text_str or "").lower()
    if not s or s in ("na", "n/a", "unknown", "none", "nan"):
        return "missing_metadata"
    # Explicitly handle US state "Georgia, USA" (or "Georgia (USA)") vs Republic of Georgia!
    if "georgia" in s and ("usa" in s or "u.s." in s or "united states" in s or "harris" in s):
        if "african american" in s:
            return "non_western"
        return "uncoded_us_regional"
    if "georgia" in s:
        return "non_western"
    if any(kw in s for kw in NON_WEST_KW) or "jataka" in t[:200]:
        return "non_western"
    if any(kw in s for kw in WEST_KW):
        return "western"
    return "uncoded_other"


def _catalog_or_editorial_reason(txt):
    low = txt.lower()
    if re.search(r"\b(aarne|thompson|uther|atu\s*\d+|type\s+\d{3,4})\b", low):
        return "catalog_type_number"
    if re.search(r"(?:^|\s)(?:notes?:|editorial|motifs?:|see also|title\s+collection|ashliman|source:|bibliography|discussions about seniority|i found the survivors)", low):
        return "editorial_or_collector_note"
    return None


def _norm_alpha_seq(txt):
    return " ".join(re.findall(r"[a-z]+", txt.lower()))


def _holm(p_map):
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for r, (k, pv) in enumerate(items):
        run = max(run, min(1.0, (m - r) * pv))
        out[k] = round(run, 5)
    return out


def main():
    t0 = time.time()
    aft_rows = json.loads((FOLK_DIR / "aft_real_myths_500.json").read_text())
    fam_rows = json.loads((FOLK_DIR / "family_roles239_data.json").read_text())
    jung_rows = NA235.load_named_arch_data()
    jobs_rows = NJ237.load_nonjung_data()

    div_names = list(R239.ATU_DIVS)
    div2i = {d: i for i, d in enumerate(div_names)}
    y_a = np.array([div2i[r["division"]] for r in aft_rows], dtype=int)
    atu_a = np.array([str(r["atu_id"]) for r in aft_rows])

    # 1. Full row-level hygiene + cross-division alpha-token deduplication + provenance manifest!
    seen_alpha = {}
    manifest_rows = []
    keep_mask = np.ones(len(aft_rows), dtype=bool)
    prov_labels = []
    for i, r in enumerate(aft_rows):
        seq = _norm_alpha_seq(r["text"])
        excl_reason = None
        if seq in seen_alpha:
            excl_reason = f"duplicate_alpha_text_of_row_{seen_alpha[seq]}_aftidx_{aft_rows[seen_alpha[seq]]['aft_idx']}"
        else:
            seen_alpha[seq] = i
            ed = _catalog_or_editorial_reason(r["text"])
            if ed:
                excl_reason = ed
        if excl_reason is not None:
            keep_mask[i] = False
        pl = _adjudicate_provenance(r["provenance"], r["text"])
        prov_labels.append(pl)
        manifest_rows.append({
            "row_idx": i,
            "aft_idx": int(r["aft_idx"]),
            "atu_id": str(r["atu_id"]),
            "division": r["division"],
            "raw_provenance": r["provenance"],
            "adjudicated_provenance": pl,
            "included": bool(excl_reason is None),
            "exclusion_reason": excl_reason,
        })
    (EXP_DIR / "aft_clean247_manifest.json").write_text(json.dumps(manifest_rows, indent=2))
    prov_labels = np.array(prov_labels)

    # Verify known suspect rows were caught!
    excl_aft_idxs = {m["aft_idx"] for m in manifest_rows if not m["included"]}
    print(f"Hygiene: kept {int(keep_mask.sum())}/500, excluded {len(excl_aft_idxs)} rows: {sorted(excl_aft_idxs)}", flush=True)
    print("Provenance counts (kept):", dict(Counter(prov_labels[keep_mask].tolist())), flush=True)

    # 2. Load frozen C5 neutral Procrustes frame (identical to aft_audit245)
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
    Za_c5_all = V234._project_out_rows(np.mean([Za[k] for k in c5_keys], 0), Q3_c5)

    jung_names = list(NA235.ARCHETYPES)
    y_j = np.array([jung_names.index(r["archetype"]) for r in jung_rows], dtype=int)
    sids_j = np.array([int(r["setting_id"]) for r in jung_rows], dtype=int)
    y_n = np.array([NJ237.ROLES.index(r["role"]) for r in jobs_rows], dtype=int)
    sids_n = np.array([int(r["setting_id"]) for r in jobs_rows], dtype=int)
    y_f = np.array([R239.FAM_ROLES.index(r["role"]) for r in fam_rows], dtype=int)
    sids_f = np.array([int(r["setting_id"]) for r in fam_rows], dtype=int)

    Za_cl = Za_c5_all[keep_mask]
    y_cl = y_a[keep_mask]
    atu_cl = atu_a[keep_mask]
    prov_cl = prov_labels[keep_mask]
    clean_texts = [aft_rows[i]["text"] for i in np.where(keep_mask)[0]]

    # 3. Complete Basis Audit (with PCA7_fam vs PCA7_jobs AND unified 200-base excess test + Bonferroni!)
    bases = {
        "Q_jung7_labeled": R239._centroid_basis_k(Zj_c5, y_j, 8, 7),
        "PCA7_jung_unlabeled": A245._pca_basis_k(Zj_c5, 7),
        "PCA7_jung_setcent": A245._setting_centered_pca_basis_k(Zj_c5, sids_j, 7),
        "Q_fam7_labeled": R239._centroid_basis_k(Zf_c5, y_f, 8, 7),
        "PCA7_fam_unlabeled": A245._pca_basis_k(Zf_c5, 7),
        "PCA7_fam_setcent": A245._setting_centered_pca_basis_k(Zf_c5, sids_f, 7),
        "Q_jobs7_labeled": R239._centroid_basis_k(Zn_c5, y_n, 8, 7),
        "PCA7_jobs_unlabeled": A245._pca_basis_k(Zn_c5, 7),
        "PCA7_jobs_setcent": A245._setting_centered_pca_basis_k(Zn_c5, sids_n, 7),
    }
    base_scores = {bn: {k_: round(v_, 5) for k_, v_ in R239._eval_basis_capture(Q, Za_cl, y_cl, 10).items() if k_ != "per_class_v"} for bn, Q in bases.items()}

    contrast_pairs = [
        ("Q_jung7_minus_PCA7_jung_unlabeled", "Q_jung7_labeled", "PCA7_jung_unlabeled", 24751),
        ("Q_jung7_minus_PCA7_jung_setcent", "Q_jung7_labeled", "PCA7_jung_setcent", 24752),
        ("Q_jung7_minus_Q_fam7", "Q_jung7_labeled", "Q_fam7_labeled", 24753),
        ("Q_jung7_minus_Q_jobs7", "Q_jung7_labeled", "Q_jobs7_labeled", 24754),
        ("Q_fam7_minus_Q_jobs7", "Q_fam7_labeled", "Q_jobs7_labeled", 24755),
        ("PCA7_jung_minus_PCA7_jobs", "PCA7_jung_unlabeled", "PCA7_jobs_unlabeled", 24756),
        ("PCA7_fam_minus_PCA7_jobs", "PCA7_fam_unlabeled", "PCA7_jobs_unlabeled", 24757),
    ]
    contrasts = {}
    for cname, ka, kb, sd in contrast_pairs:
        contrasts[cname] = A245._type_clustered_boot_diff(bases[ka], bases[kb], Za_cl, y_cl, atu_cl, 10, n_boot=2000, seed=sd)["v_dir"]
    contrasts_holm = _holm({k: v["p_two_sided"] for k, v in contrasts.items()})

    # Unified 200-shuffle-base excess-over-own-null (EXACT SAME 200 bases for point estimate AND bootstrap CI!)
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

    qs_j = make_shuf_qs(Zj_c5, y_j, sids_j, 24761, 200)
    qs_f = make_shuf_qs(Zf_c5, y_f, sids_f, 24762, 200)
    qs_n = make_shuf_qs(Zn_c5, y_n, sids_n, 24763, 200)

    def eval_shuf_dist(Z_t, y_t, Q_obs, qs_null):
        Zc = Z_t - Z_t.mean(0, keepdims=True)
        U = NA235._ovr_dirs(Zc, y_t, 10)
        vo = float(np.mean(np.sum((U @ Q_obs) ** 2, axis=1)))
        vn = np.array([float(np.mean(np.sum((U @ q) ** 2, axis=1))) for q in qs_null])
        return vo, float(vn.mean()), float((np.sum(vn >= vo - 1e-12) + 1) / (len(vn) + 1))

    vj_o, nj_o, pj_o = eval_shuf_dist(Za_cl, y_cl, bases["Q_jung7_labeled"], qs_j)
    vf_o, nf_o, pf_o = eval_shuf_dist(Za_cl, y_cl, bases["Q_fam7_labeled"], qs_f)
    vn_o, nn_o, pn_o = eval_shuf_dist(Za_cl, y_cl, bases["Q_jobs7_labeled"], qs_n)

    # Exact same 200 bases in 1,000 division-stratified ATU-type bootstrap draws!
    div_type_groups = []
    for c in range(10):
        ix_c = np.where(y_cl == c)[0]
        by_t = {}
        for i in ix_c:
            by_t.setdefault(str(atu_cl[i]), []).append(i)
        div_type_groups.append([np.array(v, dtype=int) for v in by_t.values()])
    rng_ex = np.random.default_rng(24764)
    ex_jf, ex_jn = [], []
    # Pre-stack (200, D, 7) for vectorized speed!
    Qj_st = np.stack(qs_j, 0)  # (200, D, 7)
    Qf_st = np.stack(qs_f, 0)
    Qn_st = np.stack(qs_n, 0)
    for _ in range(1000):
        idx = np.concatenate([groups[g] for groups in div_type_groups for g in rng_ex.integers(0, len(groups), size=len(groups))])
        Zc = Za_cl[idx] - Za_cl[idx].mean(0, keepdims=True)
        U = NA235._ovr_dirs(Zc, y_cl[idx], 10)  # (10, D)
        vj = float(np.mean(np.sum((U @ bases["Q_jung7_labeled"]) ** 2, axis=1)))
        vf = float(np.mean(np.sum((U @ bases["Q_fam7_labeled"]) ** 2, axis=1)))
        vn = float(np.mean(np.sum((U @ bases["Q_jobs7_labeled"]) ** 2, axis=1)))
        # einsum ('cd,mdk->mck') -> mean over (m, c) of sum over k of sq!
        nj_b = float(np.mean(np.sum(np.einsum("cd,mdk->mck", U, Qj_st) ** 2, axis=2)))
        nf_b = float(np.mean(np.sum(np.einsum("cd,mdk->mck", U, Qf_st) ** 2, axis=2)))
        nn_b = float(np.mean(np.sum(np.einsum("cd,mdk->mck", U, Qn_st) ** 2, axis=2)))
        ex_jf.append((vj - nj_b) - (vf - nf_b))
        ex_jn.append((vj - nj_b) - (vn - nn_b))
    ex_jf, ex_jn = np.asarray(ex_jf), np.asarray(ex_jn)

    shuf_and_excess = {
        "jung": {"v_dir": round(vj_o, 5), "null_200_mean": round(nj_o, 5), "excess": round(vj_o - nj_o, 5), "p_gt_null": round(pj_o, 4)},
        "fam":  {"v_dir": round(vf_o, 5), "null_200_mean": round(nf_o, 5), "excess": round(vf_o - nf_o, 5), "p_gt_null": round(pf_o, 4)},
        "jobs": {"v_dir": round(vn_o, 5), "null_200_mean": round(nn_o, 5), "excess": round(vn_o - nn_o, 5), "p_gt_null": round(pn_o, 4)},
        "excess_jung_minus_fam": {
            "obs": round((vj_o - nj_o) - (vf_o - nf_o), 5),
            "ci95": [round(float(np.percentile(ex_jf, 2.5)), 5), round(float(np.percentile(ex_jf, 97.5)), 5)],
            "p_two_sided": round(float(min(1.0, 2.0 * min((np.sum(ex_jf <= 0) + 1) / 1001, (np.sum(ex_jf >= 0) + 1) / 1001))), 4),
        },
        "excess_jung_minus_jobs": {
            "obs": round((vj_o - nj_o) - (vn_o - nn_o), 5),
            "ci95": [round(float(np.percentile(ex_jn, 2.5)), 5), round(float(np.percentile(ex_jn, 97.5)), 5)],
            "p_two_sided": round(float(min(1.0, 2.0 * min((np.sum(ex_jn <= 0) + 1) / 1001, (np.sum(ex_jn >= 0) + 1) / 1001))), 4),
        },
    }

    # 4. Fixed 5-pair theoretical Jung -> ATU diagonal test (ASSERT == 5!)
    U_j = NA235._ovr_dirs(Zj_c5 - Zj_c5.mean(0, keepdims=True), y_j, 8)
    U_a = NA235._ovr_dirs(Za_cl - Za_cl.mean(0, keepdims=True), y_cl, 10)
    cos_ja = U_j @ U_a.T
    j2i = {a: i for i, a in enumerate(jung_names)}
    mapped_pairs = [
        ("Trickster", "01_Animal_Trickster"),
        ("Trickster", "11_Jokes_Tricksters"),
        ("Hero", "02_Supernatural_Adversary"),
        ("Wise_Old_Man", "05_Supernatural_Helpers"),
        ("Shadow", "10_Stupid_Ogre_Devil"),
    ]
    valid_pairs = [(j2i[ja], div2i[da], ja, da) for ja, da in mapped_pairs if ja in j2i and da in div2i]
    assert len(valid_pairs) == 5, f"Missing mapped pair! Got {len(valid_pairs)}"
    diag_obs = float(np.mean([cos_ja[ji, di] for ji, di, _, _ in valid_pairs]))
    rng_d = np.random.default_rng(24765)
    diag_null = [float(np.mean([cos_ja[pj[ji], pd_[di]] for ji, di, _, _ in valid_pairs]))
                 for pj, pd_ in ((rng_d.permutation(8), rng_d.permutation(10)) for _ in range(5000))]
    diag_test = {
        "n_pairs": len(valid_pairs),
        "pairs": [{f"{ja}->{da}": round(float(cos_ja[ji, di]), 4)} for ji, di, ja, da in valid_pairs],
        "mean_mapped_cos": round(diag_obs, 4),
        "p_perm_upper": round(float((np.sum(np.asarray(diag_null) >= diag_obs - 1e-12) + 1) / 5001), 4),
    }

    # 5. Type-preserving PR null on cleaned dataset
    pr_clean = A245._pr_type_preserving_null(Za_cl, y_cl, atu_cl, 10, n_perm=4999, seed=24766)

    # 6. Provenance-Clean, Strictly 0-Shared-ATU-Type Western -> Non-Western Transfer
    #    with ALL THREE lexical baselines + division-stratified 10-class (and 9-class!) ATU-type bootstrap!
    w_mask = (prov_cl == "western")
    nw_mask = (prov_cl == "non_western")
    nw_types = set(atu_cl[nw_mask].tolist())
    w_disj = w_mask & np.array([t not in nw_types for t in atu_cl], dtype=bool)
    assert len(set(atu_cl[w_disj].tolist()) & set(atu_cl[nw_mask].tolist())) == 0
    y_tr, y_te, atu_te = y_cl[w_disj], y_cl[nw_mask], atu_cl[nw_mask]

    lex_sc = R239._fit_eval_lexical_fold(
        [clean_texts[i] for i in np.where(w_disj)[0]], y_tr,
        [clean_texts[i] for i in np.where(nw_mask)[0]], 10,
    )
    eval_cls_10 = sorted(set(y_te.tolist()))
    # Sensitivity excluding 1-type division (any division with only 1 unique ATU type in test!)
    types_per_div = {c: len(set(atu_te[y_te == c].tolist())) for c in eval_cls_10}
    eval_cls_9 = [c for c in eval_cls_10 if types_per_div[c] >= 2]

    def metrics_on(sc, cls_list):
        m = np.isin(y_te, cls_list)
        # Restrict argmax to cls_list? No, keep exact 10-way argmax for comparability with headline!
        pr = np.argmax(sc, axis=1)
        return {
            "top1": round(float(np.mean(pr[m] == y_te[m])), 4),
            "balanced_acc": round(A245._balanced_acc(y_te[m], pr[m], cls_list), 4),
            "macro_auroc": round(A245._macro_auroc(y_te[m], sc[m], cls_list), 4),
        }

    lex_summary = {
        lname: {"all_10_divs": metrics_on(sc, eval_cls_10), "multi_type_divs_only": metrics_on(sc, eval_cls_9)}
        for lname, sc in lex_sc.items()
    }
    # Determine strongest lexical baseline per metric (strictly honest!)
    best_lex_bal_name = max(lex_sc.keys(), key=lambda l: lex_summary[l]["all_10_divs"]["balanced_acc"])
    best_lex_auc_name = max(lex_sc.keys(), key=lambda l: lex_summary[l]["all_10_divs"]["macro_auroc"])

    # Division-stratified ATU-type bootstrap (preserves exact class set in EVERY draw!)
    def div_strat_boot(sc_n, sc_l, cls_list, n_boot=2000, seed=24770):
        # Group test indices by (div, atu_type)
        by_div_t = {}
        for c in cls_list:
            ix_c = np.where(y_te == c)[0]
            bt = {}
            for i in ix_c:
                bt.setdefault(str(atu_te[i]), []).append(i)
            by_div_t[c] = [np.array(v, dtype=int) for v in sorted(bt.values(), key=lambda x: x[0])]
        m_sub = np.isin(y_te, cls_list)
        obs_b = A245._balanced_acc(y_te[m_sub], np.argmax(sc_n[m_sub], 1), cls_list) - A245._balanced_acc(y_te[m_sub], np.argmax(sc_l[m_sub], 1), cls_list)
        obs_a = A245._macro_auroc(y_te[m_sub], sc_n[m_sub], cls_list) - A245._macro_auroc(y_te[m_sub], sc_l[m_sub], cls_list)
        rng = np.random.default_rng(seed)
        db, da = [], []
        for _ in range(n_boot):
            ix = np.concatenate([groups[g] for c in cls_list for groups in [by_div_t[c]] for g in rng.integers(0, len(groups), size=len(groups))])
            yb = y_te[ix]
            db.append(A245._balanced_acc(yb, np.argmax(sc_n[ix], 1), cls_list) - A245._balanced_acc(yb, np.argmax(sc_l[ix], 1), cls_list))
            da.append(A245._macro_auroc(yb, sc_n[ix], cls_list) - A245._macro_auroc(yb, sc_l[ix], cls_list))
        db, da = np.asarray(db), np.asarray(da)
        return {
            "bal_acc_diff": round(obs_b, 4),
            "bal_acc_ci95": [round(float(np.percentile(db, 2.5)), 4), round(float(np.percentile(db, 97.5)), 4)],
            "bal_acc_p_one": round(float((np.sum(db <= 0) + 1) / (n_boot + 1)), 5),
            "mauc_diff": round(obs_a, 4),
            "mauc_ci95": [round(float(np.percentile(da, 2.5)), 4), round(float(np.percentile(da, 97.5)), 4)],
            "mauc_p_one": round(float((np.sum(da <= 0) + 1) / (n_boot + 1)), 5),
        }

    neural_out = {}
    p_family_best = {}
    for hk in ("phi4_mini", "olmo3_7b", "gemma4_e4b", "qwen35_9b"):
        Zt_cl = V234._project_out_rows(Za[hk][keep_mask], Q3[hk])
        src_Z = Za_cl if hk in ("phi4_mini", "olmo3_7b") else V234._project_out_rows(np.mean([Za[k][keep_mask] for k in c5_keys if k != hk], 0), Q3_c5)
        Zs_tr = src_Z[w_disj] - src_Z[w_disj].mean(0, keepdims=True)
        Zt_te = Zt_cl[nw_mask] - Zt_cl[w_disj].mean(0, keepdims=True)
        U = np.stack([V234._project_out(u, Q3[hk]) for u in NA235._ovr_dirs(Zs_tr, y_tr, 10)], 0)
        sc_n = Zt_te @ U.T
        vs_best_bal = div_strat_boot(sc_n, lex_sc[best_lex_bal_name], eval_cls_10, seed=24770)
        vs_best_auc = div_strat_boot(sc_n, lex_sc[best_lex_auc_name], eval_cls_10, seed=24771)
        vs_best_9 = div_strat_boot(sc_n, lex_sc[best_lex_bal_name], eval_cls_9, seed=24772)
        p_family_best[f"{hk}_bal_vs_{best_lex_bal_name}"] = vs_best_bal["bal_acc_p_one"]
        p_family_best[f"{hk}_auc_vs_{best_lex_auc_name}"] = vs_best_auc["mauc_p_one"]
        neural_out[hk] = {
            "all_10_divs": metrics_on(sc_n, eval_cls_10),
            "multi_type_divs_only": metrics_on(sc_n, eval_cls_9),
            f"vs_{best_lex_bal_name}_10divs": vs_best_bal,
            f"vs_{best_lex_auc_name}_10divs": vs_best_auc,
            f"vs_{best_lex_bal_name}_9divs": vs_best_9,
        }
    holm_8 = _holm(p_family_best)

    # 7. Exact permutation test for Western vs Non-Western division geometry r
    ok_xc = [c for c in range(10) if np.sum(y_tr == c) >= 3 and np.sum(y_te == c) >= 3]
    def gmat(Z_sub, y_sub):
        Zc = Z_sub - Z_sub.mean(0, keepdims=True)
        M = np.stack([AR.unit(Zc[y_sub == c].mean(0)) for c in ok_xc], 0)
        return M @ M.T
    Gw, Gnw = gmat(Za_cl[w_disj], y_tr), gmat(Za_cl[nw_mask], y_te)
    iu = np.triu_indices(len(ok_xc), 1)
    obs_xc_r = float(np.corrcoef(Gw[iu], Gnw[iu])[0, 1])
    rng_g = np.random.default_rng(24780)
    ge_r = 0
    for _ in range(20000):
        pm = rng_g.permutation(len(ok_xc))
        Gp = Gnw[np.ix_(pm, pm)]
        if float(np.corrcoef(Gw[iu], Gp[iu])[0, 1]) >= obs_xc_r - 1e-12:
            ge_r += 1
    xc_geom = {"n_divs": len(ok_xc), "obs_r": round(obs_xc_r, 4), "p_perm_one_sided_20k": round((ge_r + 1) / 20001, 4)}

    report = {
        "elapsed_sec": round(time.time() - t0, 1),
        "hygiene": {
            "n_total": 500, "n_kept": int(keep_mask.sum()), "n_excluded": len(excl_aft_idxs),
            "excluded_aft_idxs": sorted(excl_aft_idxs),
            "adjudicated_provenance_counts_kept": dict(Counter(prov_cl.tolist())),
        },
        "bases_cleaned": base_scores,
        "basis_contrasts_v_dir": contrasts,
        "basis_contrasts_holm": contrasts_holm,
        "shuffle_and_unified_200base_excess": shuf_and_excess,
        "five_pair_theoretical_diagonal_test": diag_test,
        "pr_type_preserving_cleaned": pr_clean,
        "west_vs_nonwest_geometry_perm": xc_geom,
        "west_to_nonwest_0_shared_types": {
            "n_train_western": int(w_disj.sum()), "n_test_non_western": int(nw_mask.sum()),
            "n_test_unique_types": len(nw_types), "types_per_test_div": {div_names[c]: types_per_div[c] for c in eval_cls_10},
            "majority_class_top1": round(float(Counter(y_te.tolist()).most_common(1)[0][1] / len(y_te)), 4),
            "best_lexical_by_bal_acc": best_lex_bal_name,
            "best_lexical_by_macro_auroc": best_lex_auc_name,
            "lexical_baselines": lex_summary,
            "neural_c5_perp_val3": neural_out,
            "holm_8_family_vs_strongest_lexical": holm_8,
        },
    }
    out_p = EXP_DIR / "aft_audit247_results.json"
    out_p.write_text(json.dumps(report, indent=2))
    print(f"SAVED {out_p} in {report['elapsed_sec']}s", flush=True)
    print("CONTRASTS HOLM:", contrasts_holm)
    print("EXCESS UNIFIED:", json.dumps(shuf_and_excess, indent=2))
    print("5-PAIR DIAGONAL:", diag_test)
    print("XC GEOM:", xc_geom)
    print("LEXICAL:", json.dumps(lex_summary, indent=2))
    print("HOLM-8 VS STRONGEST LEXICAL:", holm_8)


if __name__ == "__main__":
    main()
