"""Strictly Held-Out Evaluative Erasure Verification (Fixes full-set mean-zeroing anti-correlation!).
Fits Q_val2 / Q_val5 / Q_val5_house / Q_eval_svd_k STRICTLY on Fold A (50%) of val234 + HOUSE
(group-disjoint by setting/domain-item), applies that frozen projection to the COMPLETELY HELD-OUT
Fold B (50%) of val234 + HOUSE, and trains a fresh 2-fold CV decoder strictly on erased Fold B!
Also evaluates bm_actor2x2_247 under the EXACT SAME half-fit frozen erasure basis."""
import json
import os
import sys
from pathlib import Path
import numpy as np

SCRATCH = Path(os.environ.get("JPL_WORK_DIR", "."))
EXP = SCRATCH / "exp247_round3"
EXP245 = SCRATCH / "exp245_controls"
REPO_ARCH = Path(__file__).resolve().parents[2] / "bench" / "archetypes"
PLATO_CACHE = Path(os.environ.get("JPL_PLATO_DIR", str(Path(__file__).resolve().parents[2] / "bench" / "archetypes" / "fluent" / "plato")))
DEPTH_DIR = PLATO_CACHE / "depth240"

for p in (str(REPO_ARCH), str(REPO_ARCH / "fluent"), str(EXP245), str(EXP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import arch_rot229 as AR  # noqa: E402
import valence_ctrl234 as V234  # noqa: E402
import exp245_binding as E245  # noqa: E402
import exp247_actor_binding as E247  # noqa: E402

MLX_KEYS = E247.MLX_KEYS


def main():
    pairs_arch = AR.load_arch_pairs()
    house_idx = np.array([i for i, p in enumerate(pairs_arch) if p["cat"] in AR.HOUSE_CATS])
    house_cats = np.array([pairs_arch[i]["cat"] for i in house_idx])
    hsets = [str(pairs_arch[i].get("setting", pairs_arch[i].get("id", i))) for i in house_idx]
    uniq_hs = sorted(set(hsets))
    rng = np.random.default_rng(24790)
    hs_fit = set(uniq_hs[i] for i in rng.permutation(len(uniq_hs))[: len(uniq_hs) // 2])
    h_fit_m = np.array([s in hs_fit for s in hsets], dtype=bool)
    h_eval_m = ~h_fit_m

    sg_rows, ov_rows, _ = V234.load_valence_data()
    sg_doms = np.array([r["domain"] for r in sg_rows])
    n_sg = len(sg_rows)
    n_ov = len(ov_rows)
    # Stratified 50/50 split of sg_rows by domain and ov_rows
    sg_fit_m = np.zeros(n_sg, dtype=bool)
    for d in sorted(set(sg_doms.tolist())):
        ix = np.where(sg_doms == d)[0]
        sg_fit_m[rng.permutation(ix)[: len(ix) // 2]] = True
    ov_fit_m = np.zeros(n_ov, dtype=bool)
    ov_fit_m[rng.permutation(n_ov)[: n_ov // 2]] = True
    v_fit_m = np.concatenate([sg_fit_m, ov_fit_m])

    sg_eval_groups = [f"{sg_doms[i]}_{i}" for i in np.where(~sg_fit_m)[0]]
    h_eval_groups = [hsets[i] for i in np.where(h_eval_m)[0]]

    actor_rows = json.loads((EXP / "bm_actor2x2_247_data.json").read_text())["bm_actor2x2_240"]
    ben_m = np.array([r["actor_type"] == "benign" for r in actor_rows], dtype=bool)
    sid_idx, splits = E245._sid_and_splits(actor_rows, 24700)

    slot = 2  # d50
    out = {}
    for pool_key in ("mean", "last"):
        rot = E245._fit_neutral_rot_at_slot(slot, pool_key, d_plato=256)
        Dv_k, Dh_k, Za_k, bases_fit = {}, {}, {}, {}
        for k in MLX_KEYS:
            d240 = np.load(DEPTH_DIR / f"depth240_{k}.npz")
            dr = np.load(DEPTH_DIR / f"bmactor247_{k}.npz")
            Za_k[k] = E245._map_mat(dr[f"pos_{pool_key}"][slot], rot, k) - E245._map_mat(dr[f"neg_{pool_key}"][slot], rot, k)
            Dh = E245._map_mat(d240[f"house_pos_{pool_key}"][slot][house_idx], rot, k) - E245._map_mat(d240[f"house_neg_{pool_key}"][slot][house_idx], rot, k)
            Dv = E245._map_mat(d240[f"val_pos_{pool_key}"][slot], rot, k) - E245._map_mat(d240[f"val_neg_{pool_key}"][slot], rot, k)
            Dv_k[k], Dh_k[k] = Dv, Dh
            # Fit erasure bases STRICTLY on Fold A (v_fit_m / h_fit_m)!
            Dv_fit = Dv[v_fit_m]
            Dh_fit = Dh[h_fit_m]
            bases_fit[k], _, _ = E245._build_valence_erasure_bases(
                Dv_fit, Dh_fit, sg_doms[sg_fit_m], house_cats[h_fit_m]
            )

        res_p = {}
        for ek in ("raw", "val2", "val5", "val5_house", "eval_svd_12", "eval_svd_24", "eval_svd_40"):
            sent_b_accs, house_b_accs = {}, {}
            for k in MLX_KEYS:
                Qe = bases_fit[k][ek]
                Dv_eval_e = V234._project_out_rows(Dv_k[k][:n_sg][~sg_fit_m], Qe) if Qe.shape[1] > 0 else Dv_k[k][:n_sg][~sg_fit_m]
                Dh_eval_e = V234._project_out_rows(Dh_k[k][h_eval_m], Qe) if Qe.shape[1] > 0 else Dh_k[k][h_eval_m]
                sent_b_accs[k] = E247._refit_cv_eval_probe(Dv_eval_e, sg_eval_groups, n_splits=10, seed=24791)
                house_b_accs[k] = E247._refit_cv_eval_probe(Dh_eval_e, h_eval_groups, n_splits=10, seed=24792)

            # Gemma -> 5 bm_actor2x2_247 under this strictly Fold-A-fit erasure basis!
            tgts = [k for k in MLX_KEYS if k != "gemma4_e4b"]
            Qse = bases_fit["gemma4_e4b"][ek]
            Dse = V234._project_out_rows(Za_k["gemma4_e4b"], Qse) if Qse.shape[1] > 0 else Za_k["gemma4_e4b"]
            c_all, c_ben, c_adv = [], [], []
            for tk in tgts:
                Qte = bases_fit[tk][ek]
                Dte = V234._project_out_rows(Za_k[tk], Qte) if Qte.shape[1] > 0 else Za_k[tk]
                sp_a, sp_b, sp_ad = [], [], []
                for fl in splits:
                    oo = np.zeros(len(actor_rows))
                    for f in (0, 1):
                        tr, te = (fl != f), (fl == f)
                        ut = AR.unit(Dse[tr].mean(0))
                        if Qte.shape[1] > 0:
                            ut = V234._project_out(ut, Qte)
                        oo[te] = Dte[te] @ ut
                    sp_a.append(float(np.mean(oo > 0)))
                    sp_b.append(float(np.mean(oo[ben_m] > 0)))
                    sp_ad.append(float(np.mean(oo[~ben_m] > 0)))
                c_all.append(float(np.mean(sp_a)))
                c_ben.append(float(np.mean(sp_b)))
                c_adv.append(float(np.mean(sp_ad)))
            res_p[ek] = {
                "rank": int(bases_fit["gemma4_e4b"][ek].shape[1]),
                "heldout_foldB_sent_cv_mean": round(float(np.mean(list(sent_b_accs.values()))), 4),
                "heldout_foldB_house_cv_mean": round(float(np.mean(list(house_b_accs.values()))), 4),
                "gemma5_actor2x2_cv2": round(float(np.mean(c_all)), 4),
                "gemma5_actor2x2_benign": round(float(np.mean(c_ben)), 4),
                "gemma5_actor2x2_adverse": round(float(np.mean(c_adv)), 4),
            }
        out[pool_key] = res_p
    (EXP / "erasure_heldout_check247.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
