"""Add ATU-type-clustered bootstrap 95% CIs on the clean 0-shared-type Western->Non-Western
neural (C5 perp VAL3) minus best-lexical (tfidf_centroid) gain (balanced_acc & macro_auroc)."""
import json
import os
import sys
from pathlib import Path
import numpy as np

SCRATCH = Path("./local_scratch")
EXP = SCRATCH / "exp245_controls"
FOLK = SCRATCH / "folklore_test"
REPO_ARCH = Path("./bench/archetypes")
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", "./bench/archetypes/fluent/plato"))

for p in (str(REPO_ARCH), str(REPO_ARCH / "fluent"), str(FOLK), str(EXP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import named_arch235 as NA235  # noqa: E402
import real_myth_basis239 as R239  # noqa: E402
import aft_audit245 as A245  # noqa: E402


def main():
    aft_rows = json.loads((FOLK / "aft_real_myths_500.json").read_text())
    div_names = list(R239.ATU_DIVS)
    div2i = {d: i for i, d in enumerate(div_names)}
    y_a = np.array([div2i[r["division"]] for r in aft_rows], dtype=int)
    atu_a = np.array([str(r["atu_id"]) for r in aft_rows])

    seen_txt, dup_set, leak_set, clean_prov = {}, set(), set(), []
    import re
    for i, r in enumerate(aft_rows):
        nt = re.sub(r"\s+", " ", r["text"].strip().lower())
        if nt in seen_txt:
            dup_set.add(i)
        else:
            seen_txt[nt] = i
        if A245._has_catalog_or_editorial_leak(r["text"]):
            leak_set.add(i)
        clean_prov.append(A245._classify_provenance(r["provenance"], r["text"]))
    clean_mask = np.array([i not in dup_set and i not in leak_set for i in range(len(aft_rows))], dtype=bool)
    clean_prov = np.array(clean_prov)[clean_mask]
    y_cl = y_a[clean_mask]
    atu_cl = atu_a[clean_mask]
    clean_texts = [aft_rows[i]["text"] for i in np.where(clean_mask)[0]]

    w_mask = (clean_prov == "western")
    nw_mask = (clean_prov == "non_western")
    nw_types = set(atu_cl[nw_mask].tolist())
    w_disj = w_mask & np.array([t not in nw_types for t in atu_cl], dtype=bool)
    y_tr, y_te, atu_te = y_cl[w_disj], y_cl[nw_mask], atu_cl[nw_mask]
    eval_cls = sorted(set(y_te.tolist()))

    # Lexical scores (strictly train-fit)
    lex_sc = R239._fit_eval_lexical_fold([clean_texts[i] for i in np.where(w_disj)[0]], y_tr, [clean_texts[i] for i in np.where(nw_mask)[0]], 10)
    sc_tfc = lex_sc["tfidf_centroid"]

    # Neural scores (exact same frozen C5 frame as aft_audit245)
    orig_plato = AR.PLATO
    AR.PLATO = PLATO_CACHE
    try:
        rot_m = AR.fit_neutral_or_full_rotations(pool="story", neutral_only=True, d_plato=256)
    finally:
        AR.PLATO = orig_plato
    pairs_arch = AR.load_arch_pairs()
    house_idx = [i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS]
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])
    Za, Q3, uh_all, us_all, uo_all = {}, {}, {}, {}, {}
    for k in PD.ALL_KEYS:
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
    c5_keys = list(PD.CONSENSUS_KEYS)
    Q3_c5 = V234._orthonormal_basis([
        AR.unit(np.mean([us_all[k] for k in c5_keys], 0)),
        AR.unit(np.mean([uo_all[k] for k in c5_keys], 0)),
        AR.unit(np.mean([uh_all[k] for k in c5_keys], 0)),
    ])
    Za_c5_cl = V234._project_out_rows(np.mean([Za[k] for k in c5_keys], 0), Q3_c5)[clean_mask]

    # Group test indices by ATU type for type-clustered bootstrap!
    by_t = {}
    for i, t in enumerate(atu_te):
        by_t.setdefault(str(t), []).append(i)
    type_groups = [np.array(v, dtype=int) for v in sorted(by_t.values(), key=lambda x: x[0])]

    out = {"n_test_unique_atu_types": len(type_groups), "per_model_vs_tfidf_centroid": {}}
    for hk in ("phi4_mini", "olmo3_7b", "gemma4_e4b", "qwen35_9b"):
        Zt_cl = V234._project_out_rows(Za[hk][clean_mask], Q3[hk])
        src_Z = Za_c5_cl if hk in ("phi4_mini", "olmo3_7b") else V234._project_out_rows(np.mean([Za[k][clean_mask] for k in c5_keys if k != hk], 0), Q3_c5)
        Zs_tr = src_Z[w_disj] - src_Z[w_disj].mean(0, keepdims=True)
        Zt_te = Zt_cl[nw_mask] - Zt_cl[w_disj].mean(0, keepdims=True)
        U = np.stack([V234._project_out(u, Q3[hk]) for u in NA235._ovr_dirs(Zs_tr, y_tr, 10)], 0)
        sc_n = Zt_te @ U.T

        obs_b_n = A245._balanced_acc(y_te, np.argmax(sc_n, 1), eval_cls)
        obs_b_l = A245._balanced_acc(y_te, np.argmax(sc_tfc, 1), eval_cls)
        obs_a_n = A245._macro_auroc(y_te, sc_n, eval_cls)
        obs_a_l = A245._macro_auroc(y_te, sc_tfc, eval_cls)

        rng = np.random.default_rng(24599)
        db, da = [], []
        for _ in range(2000):
            ch = rng.integers(0, len(type_groups), size=len(type_groups))
            ix = np.concatenate([type_groups[g] for g in ch])
            yb = y_te[ix]
            cls_b = sorted(set(yb.tolist()))
            db.append(A245._balanced_acc(yb, np.argmax(sc_n[ix], 1), cls_b) - A245._balanced_acc(yb, np.argmax(sc_tfc[ix], 1), cls_b))
            da.append(A245._macro_auroc(yb, sc_n[ix], cls_b) - A245._macro_auroc(yb, sc_tfc[ix], cls_b))
        db, da = np.asarray(db), np.asarray(da)
        out["per_model_vs_tfidf_centroid"][hk] = {
            "bal_acc_diff": round(obs_b_n - obs_b_l, 4),
            "bal_acc_ci95": [round(float(np.percentile(db, 2.5)), 4), round(float(np.percentile(db, 97.5)), 4)],
            "bal_acc_p_one": round(float((np.sum(db <= 0) + 1) / 2001), 4),
            "mauc_diff": round(obs_a_n - obs_a_l, 4),
            "mauc_ci95": [round(float(np.percentile(da, 2.5)), 4), round(float(np.percentile(da, 97.5)), 4)],
            "mauc_p_one": round(float((np.sum(da <= 0) + 1) / 2001), 4),
        }
    (EXP / "aft_type_disj_ci245.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
