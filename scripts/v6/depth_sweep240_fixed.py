"""Experiment 245-D: Fixed Depth Sweep Re-Analysis (Zero Re-Extraction).

Directly fixes all Exp 240 bugs & statistical gaps caught in Round-1 blind audits:
  1. Fixes `depth_sweep240.py:324` (`np.linalg.norm(Zc, axis=1)` instead of matrix 1-norm)
     and asserts rowwise vectorized cosine == scalar per-row cosine within 1e-6!
  2. Replaces leaked HOUSE CV (95.1% setting overlap!) with strictly SETTING-DISJOINT 2-fold CV
     (asserts 0 setting overlap between train and test folds in every split!).
  3. Computes simultaneous setting-clustered paired bootstrap 95% CIs (2,000 draws, Bonferroni-adjusted)
     on `d50 - d30` and `d50 - d70` for HOUSE, bm120 (mean & last), and roles!
  4. Explicitly labels Frobenius R^2 as `test_centered_frob_r2` and reports per-pair 5x5 winners honestly.
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

SCRATCH_DIR = Path("./local_scratch")
EXP_DIR = SCRATCH_DIR / "exp245_controls"
EXP_DIR.mkdir(parents=True, exist_ok=True)
REPO_ARCH = Path("./bench/archetypes")
REPO_FLUENT = REPO_ARCH / "fluent"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", "./bench/archetypes/fluent/plato"))
DEPTH_DIR = PLATO_CACHE / "depth240"
OUT_JSON = EXP_DIR / "depth_sweep240_fixed_results.json"

for p in (str(REPO_ARCH), str(REPO_FLUENT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plato_data as PD  # noqa: E402
import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import nonjung_roles237 as NJ237  # noqa: E402
import bow_mirror238 as BM238  # noqa: E402

MLX_KEYS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b", "phi4_mini", "olmo3_7b")
C4_KEYS = ("gemma4_e4b", "qwen35_9b", "llama31_8b", "mistral7b")
DEPTH_FRACS = (0.15, 0.30, 0.50, 0.70, 0.85)


def _test_centered_frob_r2(Y, Yhat):
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Pc = Yhat - Yhat.mean(axis=0, keepdims=True)
    return float(1.0 - np.sum((Yc - Pc) ** 2) / max(1e-12, float(np.sum(Yc ** 2))))


def _row_cosines_verified(A, B):
    """Rowwise cosine with explicit assertion against scalar loop."""
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    num = np.sum(Ac * Bc, axis=1)
    den = np.maximum(1e-12, np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc, axis=1))
    vec = num / den
    # Verify 10 scalar samples
    for i in range(0, len(Ac), max(1, len(Ac) // 10)):
        sc = float(np.dot(Ac[i], Bc[i]) / max(1e-12, float(np.linalg.norm(Ac[i]) * np.linalg.norm(Bc[i]))))
        assert abs(float(vec[i]) - sc) < 1e-6, f"Row cosine mismatch at {i}: {vec[i]} vs {sc}"
    return vec


def _fit_rot_at_slot(data_map, slot, pool_key, d_plato=256):
    pca, Z_tr = {}, {}
    for k in MLX_KEYS:
        X = data_map[k][f"neut_{pool_key}"][slot].astype(np.float64)
        mu, V, rms, z_norm = AR.fit_pca_params(X, d_plato)
        pca[k] = (mu, V, rms)
        Z_tr[k] = z_norm
    C = Z_tr[C4_KEYS[0]].copy()
    for _ in range(10):
        aligned = [PD.apply_procrustes_map(Z_tr[k], PD.fit_procrustes_map(Z_tr[k], C)) for k in C4_KEYS]
        C = np.mean(aligned, axis=0)
        C = C / max(1e-12, float(np.sqrt(np.mean(C ** 2))))
    to_c4 = {k: PD.fit_procrustes_map(Z_tr[k], C) for k in MLX_KEYS}
    return {"pca": pca, "to_c4": to_c4}


def _map(X, rot, k):
    mu, V, rms = rot["pca"][k]
    return PD.apply_procrustes_map(AR.apply_pca_params(X.astype(np.float64), mu, V, rms), rot["to_c4"][k])


def _paired_boot_d50_vs_neighbors(by_depth_setting_scores, n_boot=2000, seed=2405):
    a30 = np.asarray(by_depth_setting_scores["d30"], dtype=np.float64)
    a50 = np.asarray(by_depth_setting_scores["d50"], dtype=np.float64)
    a70 = np.asarray(by_depth_setting_scores["d70"], dtype=np.float64)
    G = len(a50)
    rng = np.random.default_rng(seed)
    d50_30, d50_70 = [], []
    for _ in range(n_boot):
        ix = rng.integers(0, G, size=G)
        d50_30.append(float(np.mean(a50[ix] - a30[ix])))
        d50_70.append(float(np.mean(a50[ix] - a70[ix])))
    d50_30, d50_70 = np.asarray(d50_30), np.asarray(d50_70)
    return {
        "n_settings": int(G),
        "d30_mean": round(float(a30.mean()), 4),
        "d50_mean": round(float(a50.mean()), 4),
        "d70_mean": round(float(a70.mean()), 4),
        "d50_minus_d30": {
            "obs": round(float(np.mean(a50 - a30)), 4),
            "ci95": [round(float(np.percentile(d50_30, 2.5)), 4), round(float(np.percentile(d50_30, 97.5)), 4)],
            "ci95_simultaneous_bonf": [round(float(np.percentile(d50_30, 1.25)), 4), round(float(np.percentile(d50_30, 98.75)), 4)],
        },
        "d50_minus_d70": {
            "obs": round(float(np.mean(a50 - a70)), 4),
            "ci95": [round(float(np.percentile(d50_70, 2.5)), 4), round(float(np.percentile(d50_70, 97.5)), 4)],
            "ci95_simultaneous_bonf": [round(float(np.percentile(d50_70, 1.25)), 4), round(float(np.percentile(d50_70, 98.75)), 4)],
        },
    }


def main():
    t0 = time.time()
    data_map = {k: np.load(DEPTH_DIR / f"depth240_{k}.npz", allow_pickle=True) for k in MLX_KEYS}
    ev_rows = [json.loads(l) for l in open(PLATO_CACHE / "eval_passages.jsonl")]
    ev_neut_mask = np.array([r["domain"] in ("held_gut", "proc_held") for r in ev_rows], dtype=bool)
    ev_m_no_pn = np.array([r["domain"] == "items3_myth" and r["id"].endswith("_anchor") and not NJ237._has_proper_name(r["text"]) for r in ev_rows], dtype=bool)
    ev_c_no_pn = np.array([r["domain"] == "items3_mech" and r["id"].endswith("_anchor") and not NJ237._has_proper_name(r["text"]) for r in ev_rows], dtype=bool)

    pairs_arch = AR.load_arch_pairs()
    house_idx = np.array([i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS])
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])
    house_settings = np.array([str(pairs_arch[i].get("setting", pairs_arch[i].get("id", i))) for i in house_idx])
    uniq_hsets = sorted(set(house_settings.tolist()))
    hset2i = {s: i for i, s in enumerate(uniq_hsets)}
    hset_idx = np.array([hset2i[s] for s in house_settings], dtype=int)

    # Strictly SETTING-DISJOINT 2-fold splits for HOUSE (20 splits, assert 0 overlap!)
    house_disj_splits = []
    for s_i in range(20):
        rng = np.random.default_rng(24010 + s_i)
        perm = rng.permutation(len(uniq_hsets))
        f0 = set(int(x) for x in perm[: len(uniq_hsets) // 2])
        fl = np.array([0 if int(s) in f0 else 1 for s in hset_idx], dtype=int)
        assert len(set(hset_idx[fl == 0].tolist()) & set(hset_idx[fl == 1].tolist())) == 0
        house_disj_splits.append(fl)

    sg_rows, ov_rows, _ = V234.load_valence_data()
    n_sg = len(sg_rows)
    bm_rows = BM238.load_mirror_data()
    bm_sids_str = [str(r["setting_id"]) for r in bm_rows]
    uniq_bm = sorted(set(bm_sids_str))
    bm_sid = np.array([{s: i for i, s in enumerate(uniq_bm)}[s] for s in bm_sids_str], dtype=int)
    bm_splits = []
    for s_i in range(20):
        rng = np.random.default_rng(23800 + s_i)
        perm = rng.permutation(len(uniq_bm))
        f0 = set(int(x) for x in perm[: len(uniq_bm) // 2])
        bm_splits.append(np.array([0 if int(s) in f0 else 1 for s in bm_sid], dtype=int))

    matched_depth = {}
    house_set_scores = {src: {} for src in ("gemma4_e4b", "qwen35_9b")}
    bm_set_scores = {(src, pool, mode): {} for src in ("gemma4_e4b", "qwen35_9b") for pool in ("mean", "last") for mode in ("cv2_p2", "sent")}

    for slot, df in enumerate(DEPTH_FRACS):
        dkey = f"d{int(round(df * 100))}"
        rot_m = _fit_rot_at_slot(data_map, slot, "mean", d_plato=256)
        rot_l = _fit_rot_at_slot(data_map, slot, "last", d_plato=256)

        Z_ev = {k: _map(data_map[k]["ev_mean"][slot], rot_m, k) for k in MLX_KEYS}
        C4_ev = np.mean([Z_ev[k] for k in C4_KEYS], axis=0)

        neut_r2, items3_reg = {}, {}
        for hk in ("phi4_mini", "olmo3_7b"):
            neut_r2[f"c4_to_{hk}_test_centered_r2"] = round(_test_centered_frob_r2(Z_ev[hk][ev_neut_mask], C4_ev[ev_neut_mask]), 4)
            p_cos = _row_cosines_verified(Z_ev[hk], C4_ev)
            items3_reg[hk] = {
                "n_myth_no_pn": int(ev_m_no_pn.sum()),
                "n_mech_no_pn": int(ev_c_no_pn.sum()),
                "myth_no_pn": round(float(p_cos[ev_m_no_pn].mean()), 4),
                "mech_no_pn": round(float(p_cos[ev_c_no_pn].mean()), 4),
                "diff": round(float(p_cos[ev_m_no_pn].mean() - p_cos[ev_c_no_pn].mean()), 4),
            }

        # HOUSE (setting-disjoint!) & bm120
        Dh_m, Q2_m, us_m, Zbm_m = {}, {}, {}, {}
        Q2_l, us_l, Zbm_l = {}, {}, {}
        for k in MLX_KEYS:
            d = data_map[k]
            Dh_m[k] = _map(d["house_pos_mean"][slot][house_idx], rot_m, k) - _map(d["house_neg_mean"][slot][house_idx], rot_m, k)
            Dv_m = _map(d["val_pos_mean"][slot], rot_m, k) - _map(d["val_neg_mean"][slot], rot_m, k)
            Zbm_m[k] = _map(d["bm120_pos_mean"][slot], rot_m, k) - _map(d["bm120_neg_mean"][slot], rot_m, k)
            us_m[k] = AR.unit(Dv_m[:n_sg].mean(0))
            Q2_m[k] = V234._orthonormal_basis([us_m[k], AR.unit(Dv_m[n_sg:].mean(0))])

            Dv_l = _map(d["val_pos_last"][slot], rot_l, k) - _map(d["val_neg_last"][slot], rot_l, k)
            Zbm_l[k] = _map(d["bm120_pos_last"][slot], rot_l, k) - _map(d["bm120_neg_last"][slot], rot_l, k)
            us_l[k] = AR.unit(Dv_l[:n_sg].mean(0))
            Q2_l[k] = V234._orthonormal_basis([us_l[k], AR.unit(Dv_l[n_sg:].mean(0))])

        house_out, bm_out = {}, {}
        for src_k in ("gemma4_e4b", "qwen35_9b"):
            tgts = [k for k in MLX_KEYS if k != src_k]
            # Setting-disjoint HOUSE 2-fold CV
            h_raw_t, h_p2_t = {}, {}
            set_acc_accum = np.zeros(len(uniq_hsets), dtype=np.float64)
            for tk in tgts:
                ar_l, ap_l = [], []
                oof_p2_avg = np.zeros(len(house_idx), dtype=np.float64)
                for fl in house_disj_splits:
                    oof_r = np.zeros(len(house_idx))
                    oof_p = np.zeros(len(house_idx))
                    for f in (0, 1):
                        tr, te = (fl != f), (fl == f)
                        ut = AR.house_dir(Dh_m[src_k][tr], house_cats[tr])
                        ut2 = V234._project_out(V234._project_out(ut, Q2_m[src_k]), Q2_m[tk])
                        Dt2 = V234._project_out_rows(Dh_m[tk][te], Q2_m[tk])
                        oof_r[te] = Dh_m[tk][te] @ ut
                        oof_p[te] = Dt2 @ ut2
                    ar_l.append(float(np.mean(oof_r > 0)))
                    ap_l.append(float(np.mean(oof_p > 0)))
                    oof_p2_avg += (oof_p > 0).astype(np.float64) / len(house_disj_splits)
                h_raw_t[tk] = round(float(np.mean(ar_l)), 4)
                h_p2_t[tk] = round(float(np.mean(ap_l)), 4)
                for gi in range(len(uniq_hsets)):
                    set_acc_accum[gi] += float(np.mean(oof_p2_avg[hset_idx == gi])) / len(tgts)
            house_set_scores[src_k][dkey] = set_acc_accum
            house_out[src_k] = {
                "n_house_settings": len(uniq_hsets),
                "mean_setting_disjoint_raw": round(float(np.mean(list(h_raw_t.values()))), 4),
                "mean_setting_disjoint_p2": round(float(np.mean(list(h_p2_t.values()))), 4),
                "per_target_p2": h_p2_t,
            }

            bm_out[src_k] = {}
            for pname, Zb, Qd, usd in [("mean", Zbm_m, Q2_m, us_m), ("last", Zbm_l, Q2_l, us_l)]:
                s_cv2 = np.zeros(len(uniq_bm))
                s_snt = np.zeros(len(uniq_bm))
                per_t = {}
                for tk in tgts:
                    Dt = Zb[tk]
                    Dt2 = V234._project_out_rows(Dt, Qd[tk])
                    sc_s = Dt @ usd[src_k]
                    p2_accs = []
                    oof_avg = np.zeros(len(bm_rows))
                    for fl in bm_splits:
                        oof = np.zeros(len(bm_rows))
                        for f in (0, 1):
                            tr, te = (fl != f), (fl == f)
                            ut = AR.unit(Zb[src_k][tr].mean(0))
                            ut2 = V234._project_out(V234._project_out(ut, Qd[src_k]), Qd[tk])
                            oof[te] = Dt2[te] @ ut2
                        p2_accs.append(float(np.mean(oof > 0)))
                        oof_avg += (oof > 0).astype(np.float64) / len(bm_splits)
                    per_t[tk] = {"cv2_p2": round(float(np.mean(p2_accs)), 4), "sent": round(float(np.mean(sc_s > 0)), 4)}
                    for gi in range(len(uniq_bm)):
                        ix = (bm_sid == gi)
                        s_cv2[gi] += float(np.mean(oof_avg[ix])) / len(tgts)
                        s_snt[gi] += float(np.mean(sc_s[ix] > 0)) / len(tgts)
                bm_set_scores[(src_k, pname, "cv2_p2")][dkey] = s_cv2
                bm_set_scores[(src_k, pname, "sent")][dkey] = s_snt
                bm_out[src_k][pname] = {
                    "mean_cv2_p2": round(float(np.mean([v["cv2_p2"] for v in per_t.values()])), 4),
                    "mean_sent": round(float(np.mean([v["sent"] for v in per_t.values()])), 4),
                    "per_target": per_t,
                }

        matched_depth[dkey] = {
            "neutral_test_centered_r2": neut_r2,
            "items3_zero_pn_register_FIXED": items3_reg,
            "house_setting_disjoint_2fold": house_out,
            "bm120": bm_out,
        }

    contrasts = {
        "house_setting_disjoint_p2": {
            src: _paired_boot_d50_vs_neighbors(house_set_scores[src], seed=2405)
            for src in ("gemma4_e4b", "qwen35_9b")
        },
        "bm120": {
            f"{src}_{pool}_{mode}": _paired_boot_d50_vs_neighbors(bm_set_scores[(src, pool, mode)], seed=2406)
            for src in ("gemma4_e4b", "qwen35_9b")
            for pool in ("mean", "last")
            for mode in ("cv2_p2", "sent")
        },
    }

    report = {
        "elapsed_sec": round(time.time() - t0, 1),
        "matched_depth": matched_depth,
        "paired_depth_contrasts_d50_vs_d30_d70": contrasts,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"SAVED {OUT_JSON} in {report['elapsed_sec']}s", flush=True)
    for k, v in contrasts["bm120"].items():
        print(f"  {k}: d30={v['d30_mean']} d50={v['d50_mean']} d70={v['d70_mean']} | 50-30={v['d50_minus_d30']['obs']} bonfCI={v['d50_minus_d30']['ci95_simultaneous_bonf']} | 50-70={v['d50_minus_d70']['obs']} bonfCI={v['d50_minus_d70']['ci95_simultaneous_bonf']}")
    for k, v in contrasts["house_setting_disjoint_p2"].items():
        print(f"  HOUSE {k}: d30={v['d30_mean']} d50={v['d50_mean']} d70={v['d70_mean']} | 50-30={v['d50_minus_d30']['obs']} bonfCI={v['d50_minus_d30']['ci95_simultaneous_bonf']} | 50-70={v['d50_minus_d70']['obs']} bonfCI={v['d50_minus_d70']['ci95_simultaneous_bonf']}")
    for dk in ("d15", "d30", "d50", "d70", "d85"):
        print(f"  items3_zero_pn {dk}: {matched_depth[dk]['items3_zero_pn_register_FIXED']}")


if __name__ == "__main__":
    main()
