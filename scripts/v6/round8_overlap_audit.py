#!/usr/bin/env python3
"""Round-8 Claim C Overlap & Neutral-Only Verification Audit.

Verifies two independence properties for the historical ATU folktale audit (Claim C):
  1. Exact normalized-word overlap scan between `passages.jsonl` (all 3,598 passages and
     the 2,394-passage neutral `gut+proc+chat` subset used by `neutral_only=True` rotations)
     and both the 500 scored excerpts AND all 1,518 full tales in SHA-256-verified `aft.csv`.
     Fails hard if `aft.csv` is missing, truncated, or fails SHA-256 verification.
  2. Sensitivity rerun of the structural Claim C suite (all 7 bases, all 7 fixed-source
     Holm-7 contrasts with nominal & Holm-adjusted percentile/basic p-values, plus 10,000-draw
     Jung shuffle & 200-draw Family/Workplace shuffles) excluding the 3 records whose full
     public-domain source tales share editions with Gutenberg passages in `passages.jsonl`
     (`aft_idx` 109, 1464, 1494 -> `n=477`, and with the 4 audit-motivated framing exclusions
     -> `n=473`).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PLATO = Path(os.environ.get("JPL_PLATO_DIR", str(REPO / "bench" / "archetypes" / "fluent" / "plato")))
FOLK_DIR = Path(os.environ.get("JPL_FOLK_DIR", "./folklore_local_data"))
V6_DIR = Path(__file__).resolve().parent

for p in (str(REPO / "bench" / "archetypes"), str(REPO / "bench" / "archetypes" / "fluent"), str(V6_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

EXPECTED_AFT_SHA256 = "5214f213b2ef99dded16fcd4f5fca9b1e2c0675aff99ea475b1839c21f441ba1"
EXPECTED_AFT_ROWS = 1518
OVERLAP_FULLTALE_3 = {109, 1464, 1494}
FRAMING_4 = {277, 356, 383, 1390}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _max_contiguous_overlap(pset, aset):
    idx = {}
    for pid, w in pset:
        for i in range(len(w) - 5):
            idx.setdefault(tuple(w[i : i + 6]), []).append((pid, i, w))
    best = (0, None, None)
    for aid, aw in aset:
        for j in range(len(aw) - 5):
            g = tuple(aw[j : j + 6])
            if g in idx:
                for pid, i, pw in idx[g]:
                    k = 6
                    while i + k < len(pw) and j + k < len(aw) and pw[i + k] == aw[j + k]:
                        k += 1
                    if k > best[0]:
                        best = (k, pid, aid)
    return best


def _find_ge12_overlaps(pset, csv_rows):
    """Find every (passage_id, aft_idx) pair sharing >= 12 contiguous normalized words,
    computing the exact maximum contiguous match length for each matching pair."""
    idx12 = {}
    for pid, pw in pset:
        for i in range(len(pw) - 11):
            idx12.setdefault(tuple(pw[i : i + 12]), []).append((pid, i, pw))
    pair_max = {}
    for aft_idx, row in enumerate(csv_rows):
        aw = _toks(row["text"])
        for j in range(len(aw) - 11):
            g = tuple(aw[j : j + 12])
            hits = idx12.get(g)
            if hits:
                for pid, i, pw in hits:
                    k = 12
                    while i + k < len(pw) and j + k < len(aw) and pw[i + k] == aw[j + k]:
                        k += 1
                    key = (pid, aft_idx)
                    if k > pair_max.get(key, 0):
                        pair_max[key] = k
    return sorted(
        [{"passage_id": pid, "aft_idx": aid, "overlap_words": wlen} for (pid, aid), wlen in pair_max.items()],
        key=lambda x: (x["passage_id"], x["aft_idx"]),
    )


def main() -> int:
    t0 = time.time()
    import plato_data as PD
    import arch_rot229 as AR
    import valence_ctrl234 as V234
    import named_arch235 as NA235
    import nonjung_roles237 as NJ237
    import real_myth_basis239 as R239
    import aft_audit245 as A245
    import round7_claim_c_seal as R7

    aft_csv = FOLK_DIR / "aft.csv"
    if not aft_csv.exists():
        raise FileNotFoundError(
            f"Missing required upstream CSV {aft_csv}. Run `python scripts/v6/fetch_trilogy_aft.py --out-dir {FOLK_DIR}` first."
        )
    got_sha = _sha256_file(aft_csv)
    if got_sha != EXPECTED_AFT_SHA256:
        raise RuntimeError(f"aft.csv SHA-256 mismatch: got {got_sha}, expected {EXPECTED_AFT_SHA256}")

    csv.field_size_limit(10**7)
    with aft_csv.open(newline="", encoding="utf-8", errors="replace") as f:
        csv_rows = list(csv.DictReader(f))
    if len(csv_rows) != EXPECTED_AFT_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_AFT_ROWS} rows in aft.csv, found {len(csv_rows)}")

    passages = [json.loads(l) for l in (REPO / "bench/archetypes/fluent/plato/passages.jsonl").read_text().splitlines()]
    aft_rows = json.loads((FOLK_DIR / "aft_real_myths_500.json").read_text())
    m4 = json.loads((REPO / "bench/archetypes/selfshape/aft_clean248_manifest.json").read_text())

    # Join aft_rows and manifest strictly by unique aft_idx (never by position!)
    m4_by_idx = {}
    for m in m4:
        aid = int(m["aft_idx"])
        if aid in m4_by_idx:
            raise RuntimeError(f"Duplicate aft_idx {aid} in manifest")
        m4_by_idx[aid] = m
    if len(aft_rows) != 500 or len(m4_by_idx) != 500:
        raise RuntimeError(f"Expected 500 aft_rows and 500 manifest rows, got {len(aft_rows)} and {len(m4_by_idx)}")
    seen_aft = set()
    for r in aft_rows:
        aid = int(r["aft_idx"])
        if aid in seen_aft or aid not in m4_by_idx:
            raise RuntimeError(f"Invalid or missing aft_idx {aid} in aft_real_myths_500.json")
        seen_aft.add(aid)
        if str(r["atu_id"]).strip() != str(m4_by_idx[aid]["atu_id"]).strip():
            raise RuntimeError(f"ATU ID mismatch at aft_idx={aid}")

    p_all = [(p["id"], _toks(p["text"])) for p in passages]
    p_neut = [(p["id"], _toks(p["text"])) for p in passages if p["domain"] != "myth"]
    a_500 = [(int(r["aft_idx"]), _toks(r["text"])) for r in aft_rows]
    mo_all = _max_contiguous_overlap(p_all, a_500)
    mo_neut = _max_contiguous_overlap(p_neut, a_500)

    neut_hits = _find_ge12_overlaps(p_neut, csv_rows)
    all_hits = _find_ge12_overlaps(p_all, csv_rows)

    # Check whether any >=12-word full-tale hit occurs inside the scored 500 excerpts
    scored_500_toks = {aid: w for aid, w in a_500}
    p_by_id = {pid: pw for pid, pw in p_all}
    enriched_hits = []
    for h in all_hits:
        aid = h["aft_idx"]
        in_500 = aid in m4_by_idx
        inc_480 = bool(m4_by_idx[aid]["included"]) if in_500 else False
        in_scored = False
        if in_500:
            ex_w = scored_500_toks[aid]
            ex_ng = {tuple(ex_w[i : i + 12]) for i in range(len(ex_w) - 11)}
            pw = p_by_id[h["passage_id"]]
            in_scored = any(tuple(pw[i : i + 12]) in ex_ng for i in range(len(pw) - 11))
        enriched_hits.append({
            "passage_id": h["passage_id"],
            "aft_idx": aid,
            "overlap_words": h["overlap_words"],
            "in_500_sample": in_500,
            "included_480": inc_480,
            "in_scored_excerpt": in_scored,
        })

    # Separate the 6 shared public-domain editions (>=20 words) from any shorter proverbial overlap (12-19 words)
    edition_hits = [h for h in enriched_hits if h["overlap_words"] >= 20]
    derived_overlap_3 = {h["aft_idx"] for h in enriched_hits if h["included_480"]}
    if derived_overlap_3 != OVERLAP_FULLTALE_3:
        raise RuntimeError(f"Derived included-480 full-tale overlap set {derived_overlap_3} != expected {OVERLAP_FULLTALE_3}")

    jung_rows = NA235.load_named_arch_data()
    fam_rows = json.loads((REPO / "bench/archetypes/selfshape/family_roles239_data.json").read_text())
    jobs_rows = NJ237.load_nonjung_data()
    div_names = list(R239.ATU_DIVS)
    div2i = {d: i for i, d in enumerate(div_names)}
    y_a = np.array([div2i[r["division"]] for r in aft_rows], dtype=int)
    atu_a = np.array([str(r["atu_id"]) for r in aft_rows])
    aidx_a = np.array([int(r["aft_idx"]) for r in aft_rows], dtype=int)
    keep480 = np.array([bool(m4_by_idx[int(r["aft_idx"])]["included"]) for r in aft_rows], dtype=bool)
    keep477 = keep480 & (~np.isin(aidx_a, list(derived_overlap_3)))
    keep473 = keep477 & (~np.isin(aidx_a, list(FRAMING_4)))

    if (int(keep480.sum()), int(keep477.sum()), int(keep473.sum())) != (480, 477, 473):
        raise RuntimeError(f"Unexpected subset sizes: {(int(keep480.sum()), int(keep477.sum()), int(keep473.sum()))}")

    AR.PLATO = PLATO
    rot_m = AR.fit_neutral_or_full_rotations(pool="story", neutral_only=True, d_plato=256)
    c5_keys, all_keys = list(PD.CONSENSUS_KEYS), list(PD.ALL_KEYS)
    pairs_arch = AR.load_arch_pairs()
    house_idx = [i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS]
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])
    Zj, Zn, Zf, Za, uh_all, us_all, uo_all = {}, {}, {}, {}, {}, {}, {}
    for k in all_keys:
        Zj[k] = NA235._map_stories(AR._load_npz_dict(PLATO / f"arch235_{k}.npz")["mean"], rot_m, k, frame="to_c5")
        Zn[k] = NA235._map_stories(AR._load_npz_dict(PLATO / f"nj237_{k}.npz")["mean"], rot_m, k, frame="to_c5")
        Zf[k] = NA235._map_stories(AR._load_npz_dict(PLATO / f"fam239_{k}.npz")["mean"], rot_m, k, frame="to_c5")
        Za[k] = NA235._map_stories(AR._load_npz_dict(PLATO / f"aft239_{k}.npz")["mean"], rot_m, k, frame="to_c5")
        a_d = AR._load_npz_dict(PLATO / f"arch229_{k}.npz")
        v_d = AR._load_npz_dict(PLATO / f"val234_{k}.npz")
        _, _, Dh = AR.map_diffs(a_d["done_mean"][house_idx], a_d["neg_mean"][house_idx], rot_m, k, frame="to_c5")
        _, _, Dv = AR.map_diffs(v_d["pos_mean"], v_d["neg_mean"], rot_m, k, frame="to_c5")
        groups = v_d["groups"]
        uh_all[k] = AR.house_dir(Dh, house_cats)
        us_all[k] = AR.unit(Dv[groups == "sentiment_gen"].mean(0))
        uo_all[k] = AR.unit(Dv[groups == "outcome_valence"].mean(0))

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
        ("Q_jung7_minus_PCA7_raw", "Q_jung7_labeled", "PCA7_jung_unlabeled", 24851),
        ("Q_jung7_minus_PCA7_setcent", "Q_jung7_labeled", "PCA7_jung_setcent", 24852),
        ("Q_jung7_minus_Q_fam7", "Q_jung7_labeled", "Q_fam7_labeled", 24853),
        ("Q_jung7_minus_Q_jobs7", "Q_jung7_labeled", "Q_jobs7_labeled", 24854),
        ("Q_fam7_minus_Q_jobs7", "Q_fam7_labeled", "Q_jobs7_labeled", 24855),
        ("PCA7_jung_minus_PCA7_jobs", "PCA7_jung_unlabeled", "PCA7_jobs_unlabeled", 24856),
        ("PCA7_fam_minus_PCA7_jobs", "PCA7_fam_unlabeled", "PCA7_jobs_unlabeled", 24857),
    ]

    # 10,000-draw Jung shuffle (matching round7 seed 24980) + 200-draw Family/Workplace shuffles for exact Holm-3
    def _make_shuf_qs(Z_s, y_s, sids_s, seed_s, m=200):
        r_ = np.random.default_rng(seed_s)
        qs = []
        s_grps = [np.where(sids_s == s)[0] for s in sorted(set(int(x) for x in sids_s))]
        for _ in range(m):
            yp = y_s.copy()
            for ix in s_grps:
                yp[ix] = r_.permutation(yp[ix])
            qs.append(R239._centroid_basis_k(Z_s, yp, 8, 7))
        return qs

    qs_f200 = _make_shuf_qs(Zf_c5, y_f, sids_f, 24862, 200)
    qs_n200 = _make_shuf_qs(Zn_c5, y_n, sids_n, 24863, 200)

    rng_10k = np.random.default_rng(24980)
    s_groups_j = [np.where(sids_j == s)[0] for s in sorted(set(int(x) for x in sids_j))]
    U_477 = NA235._ovr_dirs(Za_c5_all[keep477] - Za_c5_all[keep477].mean(0, keepdims=True), y_a[keep477], 10)
    U_473 = NA235._ovr_dirs(Za_c5_all[keep473] - Za_c5_all[keep473].mean(0, keepdims=True), y_a[keep473], 10)
    vo_477 = float(np.mean(np.sum((U_477 @ bases["Q_jung7_labeled"]) ** 2, axis=1)))
    vo_473 = float(np.mean(np.sum((U_473 @ bases["Q_jung7_labeled"]) ** 2, axis=1)))
    ge_477 = ge_473 = 0
    for _ in range(10000):
        yp = y_j.copy()
        for ix in s_groups_j:
            yp[ix] = rng_10k.permutation(yp[ix])
        qb = R239._centroid_basis_k(Zj_c5, yp, 8, 7)
        if float(np.mean(np.sum((U_477 @ qb) ** 2, axis=1))) >= vo_477 - 1e-12:
            ge_477 += 1
        if float(np.mean(np.sum((U_473 @ qb) ** 2, axis=1))) >= vo_473 - 1e-12:
            ge_473 += 1

    out = {
        "rotation_fit_corpus": "neutral_only=True (gut+proc+chat, zero myth passages)",
        "aft_csv_sha256": got_sha,
        "aft_csv_rows_scanned": len(csv_rows),
        "passages_scanned_all": len(p_all),
        "passages_scanned_neutral": len(p_neut),
        "max_contiguous_word_overlap_all_passages_vs_scored_500": {"max_words": mo_all[0], "passage_id": mo_all[1], "aft_idx": mo_all[2]},
        "max_contiguous_word_overlap_neutral_passages_vs_scored_500": {"max_words": mo_neut[0], "passage_id": mo_neut[1], "aft_idx": mo_neut[2]},
        "neutral_passages_ge12_word_overlaps_vs_all_1518_aft_csv": len(neut_hits),
        "all_myth_pool_ge12_word_overlaps_vs_all_1518_aft_csv": enriched_hits,
        "full_tale_shared_public_domain_editions_in_gutenberg_myth_pool": edition_hits,
        "sensitivity_excluding_3_full_tale_edition_matches": {},
    }

    for lbl, km, U_c, ge in (
        ("n477_excl_3_fulltale", keep477, U_477, ge_477),
        ("n473_excl_3_and_4framing", keep473, U_473, ge_473),
    ):
        vdirs = {bn: round(R239._eval_basis_capture(Q, Za_c5_all[km], y_a[km], 10)["v_dir"], 5) for bn, Q in bases.items()}
        det = {}
        p_pct_map, p_basic_map = {}, {}
        for cn, ka, kb, sd in cpairs:
            c = R7._fixed_source_div_stratified_pct_and_basic(bases[ka], bases[kb], Za_c5_all[km], y_a[km], atu_a[km], 10, 2000, sd)
            p_pct_map[cn] = c.pop("_p_pct_fp")
            p_basic_map[cn] = c.pop("_p_basic_fp")
            det[cn] = c
        h7_pct = R7._holm(p_pct_map)
        h7_basic = R7._holm(p_basic_map)

        # Exact Holm-3 using 10k Jung + 200-draw Fam/Jobs
        p_j = (ge + 1) / 10001.0
        vo_f = float(np.mean(np.sum((U_c @ bases["Q_fam7_labeled"]) ** 2, axis=1)))
        vn_f = np.array([float(np.mean(np.sum((U_c @ q) ** 2, axis=1))) for q in qs_f200])
        p_f = float((int(np.sum(vn_f >= vo_f - 1e-12)) + 1) / (len(qs_f200) + 1))
        vo_n = float(np.mean(np.sum((U_c @ bases["Q_jobs7_labeled"]) ** 2, axis=1)))
        vn_n = np.array([float(np.mean(np.sum((U_c @ q) ** 2, axis=1))) for q in qs_n200])
        p_n = float((int(np.sum(vn_n >= vo_n - 1e-12)) + 1) / (len(qs_n200) + 1))
        h3 = R7._holm({"jung": p_j, "fam": p_f, "jobs": p_n})

        block = {
            "n_records": int(km.sum()),
            "v_dir": vdirs,
            "jung_10k_shuffle": {
                "ge_count": ge,
                "p_upper": round(p_j, 5),
                "holm3_upper": h3["jung"],
                "fam_200_p_nominal": round(p_f, 5),
                "jobs_200_p_nominal": round(p_n, 5),
                "holm3_all": h3,
            },
            "holm7_percentile": h7_pct,
            "holm7_basic": h7_basic,
            "all_7_contrasts": det,
            # Preserve direct top-level aliases for the 3 primary contrasts
            "Q_jung7_minus_PCA7_raw": det["Q_jung7_minus_PCA7_raw"],
            "Q_jung7_minus_PCA7_setcent": det["Q_jung7_minus_PCA7_setcent"],
            "Q_jung7_minus_Q_jobs7": det["Q_jung7_minus_Q_jobs7"],
        }
        out["sensitivity_excluding_3_full_tale_edition_matches"][lbl] = block

    out["elapsed_sec"] = round(time.time() - t0, 1)
    dst = REPO / "bench/archetypes/fluent/plato/v6_round8_overlap_audit.json"
    dst.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {dst} in {out['elapsed_sec']}s (all_hits={len(enriched_hits)}, edition_hits={len(edition_hits)}, neut_hits={len(neut_hits)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
