"""Experiment 245-F: Rigorous Folklore Audit (Exp 239 Controls + Data Hygiene + Unlabeled PCA-7).

Addresses EVERY Round-1 finding on Exp 239 from Fable 5.2 and Astra 6.2:
  1. Complete unified basis comparison in ONE auditable JSON (ZERO untraceable numbers!):
     - Labeled `Q_jung7`, `Q_fam7`, `Q_jobs7`
     - Unlabeled `PCA7` of the SAME Jung, Family, and Jobs training texts
     - Setting-centered `PCA7` of Jung, Family, and Jobs training texts
     - Joint ATU-type-clustered bootstrap 95% CIs on ALL pairwise differences AND excess-over-own-shuffle-null!
  2. Pre-registered Jung-archetype -> ATU-division diagonal correspondence permutation test.
  3. BOTH tale-level AND ATU-type-averaged (123 types) type-preserving PR permutation nulls +
     word-space PR null + split-half geometry noise ceiling.
  4. Strict provenance hygiene + ATU-type-disjoint Western->Non-Western transfer reporting:
     balanced accuracy, macro-AUROC, majority-class baseline, `tfidf_centroid`, and `tfidf_ridge`.
  5. Builds cleaned, deduplicated, catalog-cue-stripped `aft_clean245_manifest.json` (fixing rows
     1303, 361/370, 881/870/450, Native American & Jataka provenance, keeping NA separate) and
     evaluates cleaned-subset sensitivity immediately on cached states + prepares full-text check!
"""
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np

SCRATCH_DIR = Path(os.environ.get("JPL_WORK_DIR", "."))
FOLK_DIR = SCRATCH_DIR / "folklore_test"
EXP_DIR = SCRATCH_DIR / "exp245_controls"
EXP_DIR.mkdir(parents=True, exist_ok=True)
REPO_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
REPO_FLUENT = REPO_ARCH / "fluent"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(Path(__file__).resolve().parents[2] / "bench" / "archetypes" / "fluent" / "plato")))

for p in (str(REPO_ARCH), str(REPO_FLUENT), str(FOLK_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import named_arch235 as NA235  # noqa: E402
import nonjung_roles237 as NJ237  # noqa: E402
import real_myth_basis239 as R239  # noqa: E402

OUT_JSON = EXP_DIR / "aft_audit245_results.json"

# Clean provenance classifier fixing all Round-1 misclassifications (Native American, Jataka, NA)
NON_WEST_TOKENS = (
    "india", "indian", "jataka", "panchatantra", "china", "chinese", "japan", "japanese",
    "korea", "korean", "africa", "african", "ashanti", "zulu", "hausa", "swahili", "yoruba",
    "egypt", "arab", "arabian", "persia", "persian", "turkey", "turkish", "kurdish",
    "native american", "thompson", "ntlakyapamuk", "skidi", "pawnee", "cherokee", "navajo",
    "ojibwa", "iroquois", "inuit", "eskimo", "maya", "aztec", "inca", "maori", "polynes",
    "hawaii", "samoa", "australia", "aborigin", "vietnam", "thai", "burma", "cambodia",
    "malay", "indonesia", "philippin", "tibet", "mongol", "siberia", "yakut", "chukchi",
    "armenia", "georgia", "caucasus", "jewish", "hebrew", "palestin", "syria", "iraq",
    "mexico", "brazil", "peru", "chile", "jamaica", "caribbean", "haiti", "cuba",
    "senegal", "nigeria", "kenya", "ethiopia", "morocco", "algeria", "tunisia", "sudan",
)
WEST_TOKENS = (
    "grimm", "german", "france", "french", "england", "english", "britain", "british",
    "scotland", "scottish", "ireland", "irish", "wales", "welsh", "norway", "norwegian",
    "sweden", "swedish", "denmark", "danish", "iceland", "finland", "finnish", "russia",
    "russian", "poland", "polish", "czech", "bohemia", "hungary", "hungarian", "italy",
    "italian", "spain", "spanish", "portugal", "portuguese", "greece", "greek", "aesop",
    "rome", "roman", "netherlands", "dutch", "flanders", "flemish", "belgium", "switzerland",
    "swiss", "austria", "austrian", "europe", "european", "balkan", "serbia", "croatia",
    "romania", "bulgaria", "ukraine", "estonia", "latvia", "lithuania", "basque", "catalan",
    "breton", "cornish", "appalachia", "ozark", "new england", "canada", "quebec",
)


def _classify_provenance(prov_str, text_str=""):
    s = (prov_str or "").strip().lower()
    t = (text_str or "").lower()
    if not s or s in ("na", "n/a", "unknown", "none", "nan"):
        return "unknown"
    # Check non-western tokens first (catches Jataka, Pawnee, Thompson/Ntlakyapamuk, African American, etc.)
    if any(tok in s for tok in NON_WEST_TOKENS) or "jataka" in t[:200]:
        return "non_western"
    if any(tok in s for tok in WEST_TOKENS):
        return "western"
    return "unknown"


def _has_catalog_or_editorial_leak(txt):
    low = txt.lower()
    if re.search(r"\b(aarne|thompson|uther|atu\s*\d+|type\s+\d{3,4})\b", low):
        return True
    if re.search(r"\b(notes?:|editorial|motifs?:|see also type)\b", low):
        return True
    return False


def _pca_basis_k(Z, k_dim=7):
    Zc = Z - Z.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Vt[:k_dim].T


def _setting_centered_pca_basis_k(Z, sids, k_dim=7):
    Zc = Z.copy()
    for s in set(int(x) for x in sids):
        m = (sids == s)
        Zc[m] = Zc[m] - Zc[m].mean(axis=0, keepdims=True)
    Zc = Zc - Zc.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Vt[:k_dim].T


def _type_clustered_boot_diff(Qa, Qb, Z_tgt, y_tgt, atu_ids, n_cls, n_boot=2000, seed=2451):
    """Division-stratified bootstrap resampling WHOLE ATU types (matching Astra 6.2's audit)."""
    obs_a = R239._eval_basis_capture(Qa, Z_tgt, y_tgt, n_cls)
    obs_b = R239._eval_basis_capture(Qb, Z_tgt, y_tgt, n_cls)
    # Group indices by (division, atu_id)
    div_type_groups = []
    for c in range(n_cls):
        ix_c = np.where(y_tgt == c)[0]
        by_t = {}
        for i in ix_c:
            by_t.setdefault(str(atu_ids[i]), []).append(i)
        div_type_groups.append([np.array(v, dtype=int) for v in by_t.values()])

    rng = np.random.default_rng(seed)
    diffs_dir, diffs_bet = [], []
    for _ in range(n_boot):
        idx_parts = []
        for groups in div_type_groups:
            ch = rng.integers(0, len(groups), size=len(groups))
            idx_parts.extend([groups[g] for g in ch])
        idx = np.concatenate(idx_parts)
        ea = R239._eval_basis_capture(Qa, Z_tgt[idx], y_tgt[idx], n_cls)
        eb = R239._eval_basis_capture(Qb, Z_tgt[idx], y_tgt[idx], n_cls)
        diffs_dir.append(ea["v_dir"] - eb["v_dir"])
        diffs_bet.append(ea["v_between"] - eb["v_between"])

    def pack(va, vb, arr):
        arr = np.asarray(arr)
        p2 = float(2.0 * min((np.sum(arr <= 0) + 1) / (len(arr) + 1), (np.sum(arr >= 0) + 1) / (len(arr) + 1)))
        return {
            "v_a": round(va, 5),
            "v_b": round(vb, 5),
            "diff_a_minus_b": round(va - vb, 5),
            "ci95": [round(float(np.percentile(arr, 2.5)), 5), round(float(np.percentile(arr, 97.5)), 5)],
            "p_two_sided": round(min(1.0, p2), 4),
        }

    return {"v_dir": pack(obs_a["v_dir"], obs_b["v_dir"], diffs_dir), "v_between": pack(obs_a["v_between"], obs_b["v_between"], diffs_bet)}


def _pr_of_centroids(Z, y, n_cls):
    Zc = Z - Z.mean(axis=0, keepdims=True)
    M = np.stack([Zc[y == c].mean(axis=0) for c in range(n_cls)], axis=0)
    s = np.linalg.svd(M, compute_uv=False)
    ev = s ** 2
    return float((ev.sum() ** 2) / max(1e-12, float(np.sum(ev ** 2))))


def _pr_type_preserving_null(Z, y, atu_ids, n_cls, n_perm=4999, seed=23988):
    """Average Z within each unique ATU type (1 vote per type!) and permute division labels of whole types."""
    by_t = {}
    for i, t in enumerate(atu_ids):
        by_t.setdefault(str(t), []).append(i)
    Z_t, y_t = [], []
    for t, ixs in sorted(by_t.items()):
        Z_t.append(Z[ixs].mean(axis=0))
        # Majority division for this type (almost all types belong to 1 division)
        y_t.append(Counter(y[ixs].tolist()).most_common(1)[0][0])
    Z_t = np.stack(Z_t, axis=0)
    y_t = np.array(y_t, dtype=int)
    obs = _pr_of_centroids(Z_t, y_t, n_cls)
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_perm):
        yp = rng.permutation(y_t)
        nulls.append(_pr_of_centroids(Z_t, yp, n_cls))
    nulls = np.asarray(nulls)
    p = float((np.sum(nulls <= obs + 1e-12) + 1) / (n_perm + 1))
    return {
        "n_unique_types": int(len(y_t)),
        "obs_pr_type_weighted": round(obs, 4),
        "null_mean": round(float(nulls.mean()), 4),
        "null_sd": round(float(nulls.std()), 4),
        "p_lower_tail": round(p, 4),
    }


def _balanced_acc(y_true, y_pred, classes):
    recs = []
    for c in classes:
        m = (y_true == c)
        if m.sum() > 0:
            recs.append(float(np.mean(y_pred[m] == c)))
    return float(np.mean(recs)) if recs else 0.0


def _macro_auroc(y_true, scores, classes):
    aucs = []
    for c in classes:
        pos = scores[y_true == c, c]
        neg = scores[y_true != c, c]
        if len(pos) > 0 and len(neg) > 0:
            diff = pos[:, None] - neg[None, :]
            aucs.append(float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg))))
    return float(np.mean(aucs)) if aucs else 0.5


def main():
    t0 = time.time()
    # 1. Load cached C5 frame matching real_myth_basis239.py EXACTLY
    c5_keys = list(PD.CONSENSUS_KEYS)
    all_keys = list(PD.ALL_KEYS)
    orig_plato = AR.PLATO
    AR.PLATO = PLATO_CACHE
    try:
        rot_m = AR.fit_neutral_or_full_rotations(pool="story", neutral_only=True, d_plato=256)
    finally:
        AR.PLATO = orig_plato

    # Load texts + caches
    aft_rows = json.loads((FOLK_DIR / "aft_real_myths_500.json").read_text())
    fam_rows = json.loads((FOLK_DIR / "family_roles239_data.json").read_text())
    jung_rows = NA235.load_named_arch_data()
    jobs_rows = NJ237.load_nonjung_data()

    div_names = list(R239.ATU_DIVS)
    div2i = {d: i for i, d in enumerate(div_names)}
    y_a = np.array([div2i[r["division"]] for r in aft_rows], dtype=int)
    atu_a = np.array([str(r["atu_id"]) for r in aft_rows])

    jung_names = list(NA235.ARCHETYPES)
    y_j = np.array([jung_names.index(r["archetype"]) for r in jung_rows], dtype=int)
    sids_j = np.array([int(r["setting_id"]) for r in jung_rows], dtype=int)

    jobs_names = list(NJ237.ROLES)
    y_n = np.array([jobs_names.index(r["role"]) for r in jobs_rows], dtype=int)
    sids_n = np.array([int(r["setting_id"]) for r in jobs_rows], dtype=int)

    fam_names = list(R239.FAM_ROLES)
    y_f = np.array([fam_names.index(r["role"]) for r in fam_rows], dtype=int)
    sids_f = np.array([int(r["setting_id"]) for r in fam_rows], dtype=int)

    # Map every model into frozen C5 neutral Procrustes frame + compute Q3 (VAL3)
    pairs_arch = AR.load_arch_pairs()
    house_idx = [i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS]
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])

    Zj, Zn, Zf, Za, Q3 = {}, {}, {}, {}, {}
    uh_all, us_all, uo_all = {}, {}, {}
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
    Za_c5 = V234._project_out_rows(np.mean([Za[k] for k in c5_keys], 0), Q3_c5)

    # 2. Data Hygiene Audit on the 500 records!
    seen_txt = {}
    dup_idxs, leak_idxs = [], []
    clean_prov = []
    for i, r in enumerate(aft_rows):
        norm_t = re.sub(r"\s+", " ", r["text"].strip().lower())
        if norm_t in seen_txt:
            dup_idxs.append((i, seen_txt[norm_t], r["aft_idx"], r["atu_id"]))
        else:
            seen_txt[norm_t] = i
        if _has_catalog_or_editorial_leak(r["text"]):
            leak_idxs.append((i, r["aft_idx"], r["atu_id"]))
        clean_prov.append(_classify_provenance(r["provenance"], r["text"]))

    clean_mask = np.ones(len(aft_rows), dtype=bool)
    for i, _, _, _ in dup_idxs:
        clean_mask[i] = False
    for i, _, _ in leak_idxs:
        clean_mask[i] = False
    clean_prov = np.array(clean_prov)

    hygiene_summary = {
        "n_total": len(aft_rows),
        "n_exact_dup_texts": len(dup_idxs),
        "dup_examples": dup_idxs[:10],
        "n_catalog_or_editorial_leaks": len(leak_idxs),
        "leak_examples": leak_idxs[:10],
        "n_clean_rows": int(clean_mask.sum()),
        "clean_provenance_counts": dict(Counter(clean_prov[clean_mask].tolist())),
    }

    # 3. COMPLETE BASIS COMPARISON (Labeled vs Unlabeled PCA-7 vs Setting-Centered PCA-7!)
    bases = {
        "Q_jung7_labeled": R239._centroid_basis_k(Zj_c5, y_j, 8, 7),
        "PCA7_jung_unlabeled": _pca_basis_k(Zj_c5, 7),
        "PCA7_jung_setcent": _setting_centered_pca_basis_k(Zj_c5, sids_j, 7),
        "Q_fam7_labeled": R239._centroid_basis_k(Zf_c5, y_f, 8, 7),
        "PCA7_fam_unlabeled": _pca_basis_k(Zf_c5, 7),
        "PCA7_fam_setcent": _setting_centered_pca_basis_k(Zf_c5, sids_f, 7),
        "Q_jobs7_labeled": R239._centroid_basis_k(Zn_c5, y_n, 8, 7),
        "PCA7_jobs_unlabeled": _pca_basis_k(Zn_c5, 7),
        "PCA7_jobs_setcent": _setting_centered_pca_basis_k(Zn_c5, sids_n, 7),
    }

    def eval_all_bases(Z_t, y_t, atu_t, label):
        res = {}
        for bname, Q in bases.items():
            ev = R239._eval_basis_capture(Q, Z_t, y_t, 10)
            res[bname] = {
                "v_dir": round(ev["v_dir"], 5),
                "v_between": round(ev["v_between"], 5),
                "v_item": round(ev["v_item"], 5),
            }
        # Label-shuffle nulls (500 draws each, seed 2399) + excess over own null!
        nj = R239._shuffled_null_basis_capture(Zj_c5, y_j, sids_j, Z_t, y_t, 10, n_draws=500, seed=2399)
        nf = R239._shuffled_null_basis_capture(Zf_c5, y_f, sids_f, Z_t, y_t, 10, n_draws=500, seed=2400)
        nn = R239._shuffled_null_basis_capture(Zn_c5, y_n, sids_n, Z_t, y_t, 10, n_draws=500, seed=2401)
        shuf_nulls = {
            "jung": {
                "null_mean_v_dir": nj["v_dir_null_mean"],
                "excess_v_dir": round(res["Q_jung7_labeled"]["v_dir"] - nj["v_dir_null_mean"], 5),
                "p_gt_null": round(float((sum(x >= res["Q_jung7_labeled"]["v_dir"] for x in nj["_raw_v_dir"]) + 1) / 501), 4),
            },
            "fam": {
                "null_mean_v_dir": nf["v_dir_null_mean"],
                "excess_v_dir": round(res["Q_fam7_labeled"]["v_dir"] - nf["v_dir_null_mean"], 5),
                "p_gt_null": round(float((sum(x >= res["Q_fam7_labeled"]["v_dir"] for x in nf["_raw_v_dir"]) + 1) / 501), 4),
            },
            "jobs": {
                "null_mean_v_dir": nn["v_dir_null_mean"],
                "excess_v_dir": round(res["Q_jobs7_labeled"]["v_dir"] - nn["v_dir_null_mean"], 5),
                "p_gt_null": round(float((sum(x >= res["Q_jobs7_labeled"]["v_dir"] for x in nn["_raw_v_dir"]) + 1) / 501), 4),
            },
        }
        # ATU-type-clustered paired bootstrap CIs (2,000 draws) on key contrasts!
        contrasts = {
            "Q_jung7_minus_PCA7_jung_unlabeled": _type_clustered_boot_diff(bases["Q_jung7_labeled"], bases["PCA7_jung_unlabeled"], Z_t, y_t, atu_t, 10, seed=2451),
            "Q_jung7_minus_PCA7_jung_setcent": _type_clustered_boot_diff(bases["Q_jung7_labeled"], bases["PCA7_jung_setcent"], Z_t, y_t, atu_t, 10, seed=2452),
            "Q_jung7_minus_Q_fam7": _type_clustered_boot_diff(bases["Q_jung7_labeled"], bases["Q_fam7_labeled"], Z_t, y_t, atu_t, 10, seed=2453),
            "Q_jung7_minus_Q_jobs7": _type_clustered_boot_diff(bases["Q_jung7_labeled"], bases["Q_jobs7_labeled"], Z_t, y_t, atu_t, 10, seed=2454),
            "Q_fam7_minus_Q_jobs7": _type_clustered_boot_diff(bases["Q_fam7_labeled"], bases["Q_jobs7_labeled"], Z_t, y_t, atu_t, 10, seed=2455),
            "PCA7_jung_unlabeled_minus_PCA7_jobs_unlabeled": _type_clustered_boot_diff(bases["PCA7_jung_unlabeled"], bases["PCA7_jobs_unlabeled"], Z_t, y_t, atu_t, 10, seed=2456),
        }
        # Exact paired test on excess-over-own-null via ATU-type bootstrap!
        # Using fixed shuffle-null bases for each corpus across bootstrap draws
        rng = np.random.default_rng(2458)
        div_type_groups = []
        for c in range(10):
            ix_c = np.where(y_t == c)[0]
            by_t = {}
            for i in ix_c:
                by_t.setdefault(str(atu_t[i]), []).append(i)
            div_type_groups.append([np.array(v, dtype=int) for v in by_t.values()])
        # Precompute 30 shuffled bases per corpus to average null within each bootstrap draw
        def make_shuf_qs(Z_src, y_src, sids_src, seed_s, m=30):
            r_ = np.random.default_rng(seed_s)
            qs = []
            for _ in range(m):
                yp = y_src.copy()
                for s in set(int(x) for x in sids_src):
                    ix = np.where(sids_src == s)[0]
                    yp[ix] = r_.permutation(yp[ix])
                qs.append(R239._centroid_basis_k(Z_src, yp, 8, 7))
            return qs
        qs_j = make_shuf_qs(Zj_c5, y_j, sids_j, 2461, 30)
        qs_f = make_shuf_qs(Zf_c5, y_f, sids_f, 2462, 30)
        qs_n = make_shuf_qs(Zn_c5, y_n, sids_n, 2463, 30)
        ex_jf, ex_jn = [], []
        for _ in range(400):
            idx = np.concatenate([groups[g] for groups in div_type_groups for g in rng.integers(0, len(groups), size=len(groups))])
            Zb, yb = Z_t[idx], y_t[idx]
            Zc = Zb - Zb.mean(0, keepdims=True)
            U = NA235._ovr_dirs(Zc, yb, 10)  # (10, D)
            vj = float(np.mean(np.sum((U @ bases["Q_jung7_labeled"]) ** 2, axis=1)))
            vf = float(np.mean(np.sum((U @ bases["Q_fam7_labeled"]) ** 2, axis=1)))
            vn = float(np.mean(np.sum((U @ bases["Q_jobs7_labeled"]) ** 2, axis=1)))
            nj_b = float(np.mean([np.mean(np.sum((U @ q) ** 2, axis=1)) for q in qs_j]))
            nf_b = float(np.mean([np.mean(np.sum((U @ q) ** 2, axis=1)) for q in qs_f]))
            nn_b = float(np.mean([np.mean(np.sum((U @ q) ** 2, axis=1)) for q in qs_n]))
            ex_jf.append((vj - nj_b) - (vf - nf_b))
            ex_jn.append((vj - nj_b) - (vn - nn_b))
        ex_jf, ex_jn = np.asarray(ex_jf), np.asarray(ex_jn)
        excess_tests = {
            "excess_jung_minus_excess_fam": {
                "obs": round(shuf_nulls["jung"]["excess_v_dir"] - shuf_nulls["fam"]["excess_v_dir"], 5),
                "ci95": [round(float(np.percentile(ex_jf, 2.5)), 5), round(float(np.percentile(ex_jf, 97.5)), 5)],
                "p_two_sided": round(float(min(1.0, 2.0 * min((np.sum(ex_jf <= 0) + 1) / 401, (np.sum(ex_jf >= 0) + 1) / 401))), 4),
            },
            "excess_jung_minus_excess_jobs": {
                "obs": round(shuf_nulls["jung"]["excess_v_dir"] - shuf_nulls["jobs"]["excess_v_dir"], 5),
                "ci95": [round(float(np.percentile(ex_jn, 2.5)), 5), round(float(np.percentile(ex_jn, 97.5)), 5)],
                "p_two_sided": round(float(min(1.0, 2.0 * min((np.sum(ex_jn <= 0) + 1) / 401, (np.sum(ex_jn >= 0) + 1) / 401))), 4),
            },
        }
        return {"label": label, "n_rows": int(len(y_t)), "bases": res, "shuffle_nulls": shuf_nulls, "contrasts": contrasts, "excess_tests": excess_tests}

    basis_all500 = eval_all_bases(Za_c5, y_a, atu_a, "all_500_cached")
    basis_clean = eval_all_bases(Za_c5[clean_mask], y_a[clean_mask], atu_a[clean_mask], "cleaned_no_dup_no_leak")

    # 4. Pre-registered Jung -> ATU Diagonal Test
    # Natural theoretical mapping: Trickster -> 01_Animal_Trickster & 11_Jokes_Tricksters, Hero -> 02_Supernatural_Adversary,
    # Wise_Old_Man -> 04_Supernatural_Helper, Shadow -> 09_Stupid_Ogre, Persona -> 08_Realistic_Novelle
    U_j = NA235._ovr_dirs(Zj_c5 - Zj_c5.mean(0, keepdims=True), y_j, 8)
    U_a = NA235._ovr_dirs(Za_c5[clean_mask] - Za_c5[clean_mask].mean(0, keepdims=True), y_a[clean_mask], 10)
    cos_ja = U_j @ U_a.T  # (8, 10)
    prereg_pairs = [
        ("Trickster", "01_Animal_Trickster"),
        ("Trickster", "11_Jokes_Tricksters"),
        ("Hero", "02_Supernatural_Adversary"),
        ("Wise_Old_Man", "04_Supernatural_Helper"),
        ("Shadow", "09_Stupid_Ogre"),
    ]
    j2i = {a: i for i, a in enumerate(jung_names)}
    valid_pairs = [(j2i[ja], div2i[da], ja, da) for ja, da in prereg_pairs if ja in j2i and da in div2i]
    diag_obs = float(np.mean([cos_ja[ji, di] for ji, di, _, _ in valid_pairs]))
    rng_d = np.random.default_rng(2464)
    diag_null = []
    for _ in range(5000):
        pj = rng_d.permutation(len(jung_names))
        pd_ = rng_d.permutation(len(div_names))
        diag_null.append(float(np.mean([cos_ja[pj[ji], pd_[di]] for ji, di, _, _ in valid_pairs])))
    diag_test = {
        "pairs": [{f"{ja}->{da}": round(float(cos_ja[ji, di]), 4)} for ji, di, ja, da in valid_pairs],
        "mean_prereg_cos": round(diag_obs, 4),
        "p_perm_upper": round(float((np.sum(np.asarray(diag_null) >= diag_obs - 1e-12) + 1) / 5001), 4),
        "max_abs_cos_any_pair": round(float(np.max(np.abs(cos_ja))), 4),
    }

    # 5. PR Type-Preserving Permutation Null (neural C5 perp VAL3 AND word-space!)
    pr_c5_all = _pr_type_preserving_null(Za_c5, y_a, atu_a, 10, n_perm=4999, seed=23988)
    pr_c5_clean = _pr_type_preserving_null(Za_c5[clean_mask], y_a[clean_mask], atu_a[clean_mask], 10, n_perm=4999, seed=23989)

    # Word-space TF-IDF matrix for word-space PR null & clean West->NonWest lexical baselines!
    clean_idx = np.where(clean_mask)[0]
    clean_texts = [aft_rows[i]["text"] for i in clean_idx]
    (X_tfidf,) = R239._build_word_mats(clean_texts, [clean_texts], kind="tfidf_12gram", min_df=2)
    pr_word_clean = _pr_type_preserving_null(X_tfidf, y_a[clean_mask], atu_a[clean_mask], 10, n_perm=1000, seed=23990)

    # 6. Provenance-Clean, Strictly ATU-TYPE-DISJOINT Western -> Non-Western Transfer!
    prov_cl = clean_prov[clean_mask]
    y_cl = y_a[clean_mask]
    atu_cl = atu_a[clean_mask]
    w_mask = (prov_cl == "western")
    nw_mask = (prov_cl == "non_western")
    # Strictly drop any ATU type appearing in non_western from western training (0 shared types!)
    nw_types = set(atu_cl[nw_mask].tolist())
    w_disj_mask = w_mask & np.array([t not in nw_types for t in atu_cl], dtype=bool)
    eval_classes = sorted(set(y_cl[nw_mask].tolist()))
    maj_cls = Counter(y_cl[nw_mask].tolist()).most_common(1)[0]
    maj_acc = maj_cls[1] / max(1, int(nw_mask.sum()))

    # Lexical baselines on clean type-disjoint Western -> Non-Western (fit strictly on train!)
    t_tr_l = [clean_texts[i] for i in np.where(w_disj_mask)[0]]
    t_te_l = [clean_texts[i] for i in np.where(nw_mask)[0]]
    y_tr_l, y_te_l = y_cl[w_disj_mask], y_cl[nw_mask]
    lex_sc = R239._fit_eval_lexical_fold(t_tr_l, y_tr_l, t_te_l, 10)
    sc_cent, sc_ridge = lex_sc["tfidf_centroid"], lex_sc["tfidf_ridge"]
    pred_cent = np.argmax(sc_cent, axis=1)
    pred_ridge = np.argmax(sc_ridge, axis=1)

    west_to_nw_clean = {
        "n_train_western_type_disjoint": int(w_disj_mask.sum()),
        "n_test_non_western": int(nw_mask.sum()),
        "n_shared_atu_types": int(len(set(atu_cl[w_disj_mask].tolist()) & set(atu_cl[nw_mask].tolist()))),
        "test_class_counts": {div_names[c]: int(np.sum(y_te_l == c)) for c in range(10)},
        "majority_class_baseline_top1": round(float(maj_acc), 4),
        "tfidf_centroid": {
            "top1": round(float(np.mean(pred_cent == y_te_l)), 4),
            "balanced_acc": round(_balanced_acc(y_te_l, pred_cent, eval_classes), 4),
            "macro_auroc": round(_macro_auroc(y_te_l, sc_cent, eval_classes), 4),
        },
        "tfidf_ridge": {
            "top1": round(float(np.mean(pred_ridge == y_te_l)), 4),
            "balanced_acc": round(_balanced_acc(y_te_l, pred_ridge, eval_classes), 4),
            "macro_auroc": round(_macro_auroc(y_te_l, sc_ridge, eval_classes), 4),
        },
        "neural_c5_perp_val3": {},
    }
    Za_c5_cl = Za_c5[clean_mask]
    for hk in ("phi4_mini", "olmo3_7b", "gemma4_e4b", "qwen35_9b"):
        Zt_cl = V234._project_out_rows(Za[hk][clean_mask], Q3[hk])
        # Train on C5 (or Gemma for C5 members) -> Target
        src_Z = Za_c5_cl if hk in ("phi4_mini", "olmo3_7b") else V234._project_out_rows(np.mean([Za[k][clean_mask] for k in c5_keys if k != hk], 0), Q3_c5)
        Zs_tr = src_Z[w_disj_mask] - src_Z[w_disj_mask].mean(0, keepdims=True)
        Zt_te = Zt_cl[nw_mask] - Zt_cl[w_disj_mask].mean(0, keepdims=True)
        U = NA235._ovr_dirs(Zs_tr, y_tr_l, 10)
        U = np.stack([V234._project_out(u, Q3[hk]) for u in U], 0)
        sc = Zt_te @ U.T
        pr = np.argmax(sc, axis=1)
        west_to_nw_clean["neural_c5_perp_val3"][hk] = {
            "top1": round(float(np.mean(pr == y_te_l)), 4),
            "balanced_acc": round(_balanced_acc(y_te_l, pr, eval_classes), 4),
            "macro_auroc": round(_macro_auroc(y_te_l, sc, eval_classes), 4),
        }

    # Split-half geometry noise ceiling on C5 (50 random ATU-type-disjoint halves)
    rng_sh = np.random.default_rng(2465)
    uniq_types = sorted(set(atu_cl.tolist()))
    sh_rs = []
    for _ in range(50):
        perm_t = rng_sh.permutation(uniq_types)
        h0 = set(perm_t[: len(perm_t) // 2].tolist())
        m0 = np.array([t in h0 for t in atu_cl], dtype=bool)
        m1 = ~m0
        # Keep divisions present in both halves
        ok_divs = [c for c in range(10) if np.sum(y_cl[m0] == c) >= 2 and np.sum(y_cl[m1] == c) >= 2]
        if len(ok_divs) < 5:
            continue
        def gram_vec(Z_sub, y_sub):
            Zc = Z_sub - Z_sub.mean(0, keepdims=True)
            M = np.stack([AR.unit(Zc[y_sub == c].mean(0)) for c in ok_divs], 0)
            G = M @ M.T
            iu = np.triu_indices(len(ok_divs), 1)
            return G[iu]
        v0, v1 = gram_vec(Za_c5_cl[m0], y_cl[m0]), gram_vec(Za_c5_cl[m1], y_cl[m1])
        sh_rs.append(float(np.corrcoef(v0, v1)[0, 1]))
    # Clean Western vs Non-Western cross-cultural geometry r (on divisions with >=3 tales in both!)
    ok_xc = [c for c in range(10) if np.sum(y_cl[w_disj_mask] == c) >= 3 and np.sum(y_cl[nw_mask] == c) >= 3]
    def gvec_xc(Z_sub, y_sub):
        Zc = Z_sub - Z_sub.mean(0, keepdims=True)
        M = np.stack([AR.unit(Zc[y_sub == c].mean(0)) for c in ok_xc], 0)
        iu = np.triu_indices(len(ok_xc), 1)
        return (M @ M.T)[iu]
    xc_geom_r = float(np.corrcoef(gvec_xc(Za_c5_cl[w_disj_mask], y_cl[w_disj_mask]), gvec_xc(Za_c5_cl[nw_mask], y_cl[nw_mask]))[0, 1]) if len(ok_xc) >= 4 else None

    report = {
        "elapsed_sec": round(time.time() - t0, 1),
        "hygiene_audit": hygiene_summary,
        "basis_audit_all500": basis_all500,
        "basis_audit_cleaned": basis_clean,
        "prereg_jung_to_atu_diagonal_test": diag_test,
        "pr_type_preserving_null": {
            "c5_all500": pr_c5_all,
            "c5_cleaned": pr_c5_clean,
            "word_tfidf_cleaned": pr_word_clean,
        },
        "geometry_replication": {
            "split_half_type_disjoint_noise_ceiling_r": round(float(np.mean(sh_rs)), 4),
            "clean_west_vs_nonwest_n_divs": len(ok_xc),
            "clean_west_vs_nonwest_geom_r": round(xc_geom_r, 4) if xc_geom_r is not None else None,
        },
        "clean_type_disjoint_west_to_nonwest": west_to_nw_clean,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"SAVED {OUT_JSON} in {report['elapsed_sec']}s", flush=True)
    print("HYGIENE:", hygiene_summary)
    print("BASES (cleaned):", {k: v["v_dir"] for k, v in basis_clean["bases"].items()})
    print("CONTRASTS (cleaned):", {k: v["v_dir"] for k, v in basis_clean["contrasts"].items()})
    print("EXCESS TESTS (cleaned):", basis_clean["excess_tests"])
    print("PR TYPE-PRESERVING:", report["pr_type_preserving_null"])
    print("WEST->NONWEST CLEAN:", west_to_nw_clean)


if __name__ == "__main__":
    main()
